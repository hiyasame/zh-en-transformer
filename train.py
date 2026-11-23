from __future__ import annotations

import argparse
from contextlib import nullcontext
import inspect
import math
import os
from pathlib import Path
import shutil
import time
import yaml
import sacrebleu
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import ParallelDataset, TokenBucketBatchSampler, make_bucketed_collate_fn
from src.decode import greedy_decode
from src.model import Transformer
from src.optim import NoamScheduler
from src.tokenizer import SentencePieceTokenizer


def resolve_device(preferred: str) -> torch.device:
    """按配置优先选择设备，若当前机器不支持则平滑回退。"""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        print(f"[提示] 请求设备 '{preferred}' 不可用，已自动切换至 'cuda'")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print(f"[提示] 请求设备 '{preferred}' 不可用，已自动切换至 'mps'")
        return torch.device("mps")
    print(f"[提示] 请求设备 '{preferred}' 不可用，已自动切换至 'cpu'")
    return torch.device("cpu")


def unwrap_model(model: nn.Module) -> Transformer:
    """兼容 DDP (.module) 与 torch.compile (._orig_mod) 包装，确保存档与解码接口一致。"""
    m = model.module if isinstance(model, DDP) else model
    return getattr(m, "_orig_mod", m)  # type: ignore[return-value]


@torch.no_grad()
def evaluate_bleu(
    model: nn.Module,
    val_pairs: list[tuple[list[int], list[int]]],
    val_raw: list[tuple[str, str]],
    en_tok: SentencePieceTokenizer,
    device: torch.device,
    pad_id: int = 0,
    bos_id: int = 2,
    eos_id: int = 3,
    max_extra_len: int = 50,
    max_eval_samples: int = 500,
    batch_size: int = 128,
    pad_to_multiple: int = 1,
    amp_dtype: torch.dtype | None = None,
    dist_eval: bool = False,
) -> tuple[float, list[tuple[str, str, str]]]:
    """在验证集上批量运行 Greedy 解码并计算标准 SacreBLEU（支持多卡 DDP 分片并行评测，消除单卡串行空等）。"""
    raw_model = unwrap_model(model)
    was_training = raw_model.training
    raw_model.eval()

    n = min(len(val_pairs), len(val_raw), max_eval_samples)
    sub_pairs = val_pairs[:n]
    sub_raw = val_raw[:n]

    use_dist = dist_eval and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if use_dist:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        per_rank = math.ceil(n / world_size)
        r_start = min(n, rank * per_rank)
        r_end = min(n, (rank + 1) * per_rank)
        my_pairs = sub_pairs[r_start:r_end]
    else:
        my_pairs = sub_pairs

    collate = make_bucketed_collate_fn(
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_to_multiple=pad_to_multiple,
    )
    local_hyps: list[str] = []
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if (amp_dtype is not None and device.type == "cuda")
        else nullcontext()
    )

    for start in range(0, len(my_pairs), batch_size):
        batch = collate(my_pairs[start : start + batch_size])
        src = batch["src"].to(device)
        src_mask = batch["src_mask"].to(device)
        max_len = min(src.size(1) + max_extra_len, 256)

        with amp_ctx:
            pred_ids = greedy_decode(
                raw_model,
                src,
                src_mask,
                max_len=max_len,
                bos_id=bos_id,
                eos_id=eos_id,
                pad_id=pad_id,
            )
        for row in pred_ids.cpu().tolist():
            local_hyps.append(en_tok.decode(row))

    if use_dist:
        gathered: list[list[str]] = [[] for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_hyps)
        hyps = [h for part in gathered for h in part]
    else:
        hyps = local_hyps

    refs: list[str] = [en for _, en in sub_raw]
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score if hyps else 0.0
    examples: list[tuple[str, str, str]] = []
    for idx in range(min(3, len(hyps))):
        examples.append((sub_raw[idx][0], hyps[idx], refs[idx]))

    if was_training:
        raw_model.train()
    return float(bleu), examples


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamScheduler,
    step: int,
    best_bleu: float,
    cfg: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = unwrap_model(model)
    torch.save(
        {
            "step": step,
            "best_bleu": best_bleu,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": cfg,
        },
        path,
    )


def infinite_micro_batches(loader: DataLoader):
    """跨 Epoch 源源不断产出微批次，供按真实更新步数 (max_steps) 驱动训练主循环。"""
    while True:
        for batch in loader:
            yield batch


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 Zh->En Transformer 模型 (支持单卡 MPS/CUDA 与多卡 H100 DDP)")
    parser.add_argument("--config", type=str, default="configs/local_small.yaml", help="YAML 配置文件路径")
    parser.add_argument("--data-dir", type=str, default=None, help="覆盖预处理数据目录")
    parser.add_argument("--save-dir", type=str, default=None, help="覆盖模型保存目录")
    parser.add_argument("--resume", type=str, default=None, help="断点续训 checkpoint 路径")
    parser.add_argument("--device", type=str, default=None, help="覆盖配置文件中的训练设备 (mps/cuda/cpu)")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖训练总步数（方便快速试验）")
    parser.add_argument("--overfit32", action="store_true", help="30 秒极速过拟合 32 条真实中英样本验证全链路")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_ddp = world_size > 1
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        rank = 0

    is_main = rank == 0

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.data_dir:
        cfg["data"]["processed_dir"] = args.data_dir
    if args.save_dir:
        cfg["train"]["save_dir"] = args.save_dir

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    decode_cfg = cfg.get("decode", {})

    if not is_ddp:
        device = resolve_device(args.device or train_cfg.get("device", "mps"))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    data_dir = Path(data_cfg["processed_dir"])
    dataset_path = data_dir / "dataset.pt"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"未找到预处理数据包 {dataset_path}，请先运行:\n"
            f"  python scripts/prepare_data.py --config {args.config}"
        )

    zh_tok = SentencePieceTokenizer(data_dir / "zh_spm.model")
    en_tok = SentencePieceTokenizer(data_dir / "en_spm.model")
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)

    train_data = payload["train"]
    val_pairs: list[tuple[list[int], list[int]]] = payload["val"]
    val_raw: list[tuple[str, str]] = payload["val_raw"]

    if args.overfit32:
        if is_main:
            print("[Overfit-32 模式] 抽取前 32 条真实中英样本做极速记忆验证...")
        tmp_ds = ParallelDataset(train_data)
        train_data = [tmp_ds[i] for i in range(min(32, len(tmp_ds)))]
        val_pairs = train_data
        val_raw = payload.get("train_sample_raw", val_raw)[: len(train_data)]
        train_cfg["grad_accum_steps"] = 1
        train_cfg["max_tokens_per_batch"] = 4096
        train_cfg["warmup_steps"] = 100
        train_cfg["lr_factor"] = 0.16
        train_cfg["max_steps"] = args.max_steps or 300
        train_cfg["eval_every"] = 150
        train_cfg["save_dir"] = "checkpoints/overfit32"
        model_cfg["dropout"] = 0.0

    pad_id = int(model_cfg.get("pad_id", 0))
    bos_id = int(model_cfg.get("bos_id", 2))
    eos_id = int(model_cfg.get("eos_id", 3))
    tie_all = bool(model_cfg.get("tie_all_embeddings", False))
    use_flash_attn = bool(model_cfg.get("use_flash_attn", True))
    use_flash_causal = bool(model_cfg.get("use_flash_causal", True)) and use_flash_attn
    unpad_loss = bool(train_cfg.get("unpad_loss", True))
    want_fused_adam = bool(train_cfg.get("fused_adam", True))
    dist_eval = bool(train_cfg.get("dist_eval", True))
    eval_batch_size = int(train_cfg.get("eval_batch_size", 128))
    bucket_pool_size = int(train_cfg.get("bucket_pool_size", 2048))

    cfg["model"]["src_vocab_size"] = zh_tok.vocab_size
    cfg["model"]["tgt_vocab_size"] = en_tok.vocab_size
    base_model = Transformer(
        src_vocab_size=zh_tok.vocab_size,
        tgt_vocab_size=en_tok.vocab_size,
        d_model=int(model_cfg["d_model"]),
        num_heads=int(model_cfg["num_heads"]),
        num_encoder_layers=int(model_cfg["num_encoder_layers"]),
        num_decoder_layers=int(model_cfg["num_decoder_layers"]),
        d_ff=int(model_cfg["d_ff"]),
        max_len=int(model_cfg.get("max_len", 512)),
        dropout=float(model_cfg["dropout"]),
        pad_id=pad_id,
        tie_tgt_weights=bool(model_cfg.get("tie_tgt_weights", True)),
        tie_all_embeddings=tie_all,
        norm_first=bool(model_cfg.get("norm_first", True)),
        use_flash_attn=use_flash_attn,
    ).to(device)

    can_fuse_adam = (
        want_fused_adam
        and device.type == "cuda"
        and "fused" in inspect.signature(torch.optim.Adam).parameters
    )
    adam_kwargs: dict = {
        "lr": 0.0,
        "betas": tuple(train_cfg.get("adam_betas", [0.9, 0.98])),
        "eps": float(train_cfg.get("adam_eps", 1e-9)),
    }
    if can_fuse_adam:
        adam_kwargs["fused"] = True

    n_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    if is_main:
        attn_backend = (
            "FlashAttention-2 (Native Causal)"
            if use_flash_causal
            else ("FlashAttention (SDPA)" if use_flash_attn else "Handwritten Attention")
        )
        tie_desc = "3-Way Shared" if tie_all else "Decoder 2-Way"
        print(
            f"设备: {device} (world_size={world_size}) | 注意力: {attn_backend} | "
            f"unpad_loss={unpad_loss} | fused_adam={can_fuse_adam} | 词表绑定: {tie_desc} | "
            f"参数量: {n_params:,} ({n_params / 1e6:.2f}M)"
        )

    optimizer = torch.optim.Adam(base_model.parameters(), **adam_kwargs)
    scheduler = NoamScheduler(
        optimizer=optimizer,
        d_model=int(model_cfg["d_model"]),
        warmup_steps=int(train_cfg["warmup_steps"]),
        factor=float(train_cfg.get("lr_factor", 1.0)),
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id,
        label_smoothing=float(train_cfg.get("label_smoothing", 0.1)),
        reduction="sum",
    )

    start_step = 0
    best_bleu = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        base_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = int(ckpt["step"])
        best_bleu = float(ckpt.get("best_bleu", 0.0))
        if is_main:
            print(f"已从 {args.resume} 恢复断点: step={start_step}, best_bleu={best_bleu:.2f}")

    use_amp = bool(train_cfg.get("use_bf16", True)) and device.type == "cuda"
    amp_dtype: torch.dtype | None = None
    scaler = None
    cuda_major = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 0
    if use_amp:
        if cuda_major >= 8 and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            if is_main:
                print("已启用 CUDA bfloat16 混合精度加速 (Ampere/Hopper 原生硬件加速)")
        else:
            amp_dtype = torch.float16
            scaler = (
                torch.amp.GradScaler("cuda", enabled=True)
                if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
                else torch.cuda.amp.GradScaler(enabled=True)
            )
            if is_main:
                print("已启用 CUDA float16 + GradScaler 混合精度加速")

    model: nn.Module = base_model
    use_compile = bool(train_cfg.get("use_compile", False)) and device.type == "cuda" and cuda_major >= 8
    if use_compile:
        if is_main:
            print("正在启用 torch.compile(dynamic=True) 符号形状图编译加速...")
        model = torch.compile(model, dynamic=True)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    save_dir = Path(train_cfg["save_dir"])
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_dir / "zh_spm.model", save_dir / "zh_spm.model")
        shutil.copy2(data_dir / "en_spm.model", save_dir / "en_spm.model")
        with open(save_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)

    default_pad_mult = 8 if (use_compile or device.type == "mps") else 8
    pad_to_multiple = int(train_cfg.get("pad_to_multiple", default_pad_mult))

    train_ds = ParallelDataset(train_data)
    sampler = TokenBucketBatchSampler(
        lengths=train_ds.lengths,
        max_tokens=int(train_cfg["max_tokens_per_batch"]),
        shuffle=not args.overfit32,
        pool_size=bucket_pool_size,
        num_replicas=world_size,
        rank=rank,
    )
    collate_fn = make_bucketed_collate_fn(
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_to_multiple=pad_to_multiple,
    )
    num_workers = 2 if (device.type == "cuda" and not args.overfit32) else 0
    loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    batch_iter = infinite_micro_batches(loader)

    max_steps = int(args.max_steps or train_cfg["max_steps"])
    grad_accum = int(train_cfg.get("grad_accum_steps", 1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    eval_every = int(train_cfg.get("eval_every", 1000))
    save_every = int(train_cfg.get("save_every", 2000))
    keep_last_n = int(train_cfg.get("keep_last_n_checkpoints", 5))
    max_extra_len = int(decode_cfg.get("max_extra_len", 50))

    model.train()
    step_range = range(start_step + 1, max_steps + 1)
    pbar = tqdm(step_range, desc="Training", dynamic_ncols=True) if is_main else step_range
    win_loss_sum = 0.0
    win_tokens = 0
    t0 = time.time()

    try:
        for step in pbar:
            optimizer.zero_grad(set_to_none=True)

            micro_batches = [next(batch_iter) for _ in range(grad_accum)]
            local_ntokens = sum(int(mb["ntokens"].item()) for mb in micro_batches)

            if is_ddp:
                ntok_t = torch.tensor(local_ntokens, dtype=torch.int64, device=device)
                dist.all_reduce(ntok_t, op=dist.ReduceOp.SUM)
                global_ntokens = int(ntok_t.item())
            else:
                global_ntokens = local_ntokens

            local_loss_sum = 0.0
            for mb_idx, mb in enumerate(micro_batches):
                src = mb["src"].to(device, non_blocking=True)
                tgt_in = mb["tgt_in"].to(device, non_blocking=True)
                tgt_out = mb["tgt_out"].to(device, non_blocking=True)
                src_mask = mb["src_mask"].to(device, non_blocking=True)
                tgt_mask = None if use_flash_causal else mb["tgt_mask"].to(device, non_blocking=True)

                sync_ctx = (
                    model.no_sync()  # type: ignore[operator]
                    if (is_ddp and isinstance(model, DDP) and mb_idx < grad_accum - 1)
                    else nullcontext()
                )
                amp_ctx = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None else nullcontext()
                )

                with sync_ctx:
                    with amp_ctx:
                        if unpad_loss:
                            valid_mask = tgt_out != pad_id
                            valid_logits = model(
                                src,
                                tgt_in,
                                src_mask,
                                tgt_mask,
                                tgt_is_causal=use_flash_causal,
                                valid_tgt_mask=valid_mask,
                            )
                            raw_loss = criterion(valid_logits, tgt_out[valid_mask])
                        else:
                            logits = model(
                                src,
                                tgt_in,
                                src_mask,
                                tgt_mask,
                                tgt_is_causal=use_flash_causal,
                            )
                            raw_loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

                        # DDP 会对各卡梯度除以 world_size，乘回 world_size 以严格等价于全局有效词平均梯度
                        norm_loss = (raw_loss * world_size) / max(1, global_ntokens)

                    if scaler is not None:
                        scaler.scale(norm_loss).backward()
                    else:
                        norm_loss.backward()

                local_loss_sum += float(raw_loss.item())

            if scaler is not None:
                old_scale = scaler.get_scale()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                lr = scheduler.step()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < old_scale:
                    scheduler.step_num = max(0, scheduler.step_num - 1)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                lr = scheduler.step()
                optimizer.step()

            if is_ddp:
                stat_t = torch.tensor([local_loss_sum, float(local_ntokens)], dtype=torch.float64, device=device)
                dist.all_reduce(stat_t, op=dist.ReduceOp.SUM)
                win_loss_sum += float(stat_t[0].item())
                win_tokens += int(stat_t[1].item())
            else:
                win_loss_sum += local_loss_sum
                win_tokens += local_ntokens

            if is_main and (step % 20 == 0 or step == 1):
                avg_loss = win_loss_sum / max(1, win_tokens)
                ppl = math.exp(min(avg_loss, 20.0))
                tok_per_s = win_tokens / max(1e-6, time.time() - t0)
                pbar.set_postfix(loss=f"{avg_loss:.3f}", ppl=f"{ppl:.1f}", lr=f"{lr:.2e}", tok_s=f"{tok_per_s:.0f}")  # type: ignore[union-attr]
                win_loss_sum = 0.0
                win_tokens = 0
                t0 = time.time()

            if step % eval_every == 0 or step == max_steps:
                if (is_ddp and dist_eval) or is_main:
                    bleu, samples = evaluate_bleu(
                        model=model,
                        val_pairs=val_pairs,
                        val_raw=val_raw,
                        en_tok=en_tok,
                        device=device,
                        pad_id=pad_id,
                        bos_id=bos_id,
                        eos_id=eos_id,
                        max_extra_len=max_extra_len,
                        batch_size=eval_batch_size,
                        pad_to_multiple=pad_to_multiple,
                        amp_dtype=amp_dtype,
                        dist_eval=is_ddp and dist_eval,
                    )
                else:
                    bleu, samples = 0.0, []

                if is_main:
                    tqdm.write(
                        f"\n[Step {step}/{max_steps}] Val Greedy SacreBLEU = {bleu:.2f} (best = {max(best_bleu, bleu):.2f})"
                    )
                    for zh_s, hyp_s, ref_s in samples:
                        tqdm.write(f"  源(Zh): {zh_s}")
                        tqdm.write(f"  译(Pr): {hyp_s}")
                        tqdm.write(f"  参(En): {ref_s}")

                    if bleu >= best_bleu:
                        best_bleu = bleu
                        save_checkpoint(save_dir / "best.pt", model, optimizer, scheduler, step, best_bleu, cfg)

                if is_ddp:
                    dist.barrier()
                t0 = time.time()

            if (not args.overfit32) and is_main and (step % save_every == 0 or step == max_steps):
                save_checkpoint(save_dir / "last.pt", model, optimizer, scheduler, step, best_bleu, cfg)
                step_ckpt = save_dir / f"step_{step:07d}.pt"
                save_checkpoint(step_ckpt, model, optimizer, scheduler, step, best_bleu, cfg)
                all_step_ckpts = sorted(save_dir.glob("step_*.pt"))
                for old_ckpt in all_step_ckpts[:-keep_last_n]:
                    old_ckpt.unlink(missing_ok=True)

        if is_main:
            print(f"\n训练完成！最高验证集 BLEU = {best_bleu:.2f}，模型保存在: {save_dir}")
    finally:
        if is_ddp and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
