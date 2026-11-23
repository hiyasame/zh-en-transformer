from __future__ import annotations

import argparse
from pathlib import Path
import time
import sacrebleu
import torch
from tqdm import tqdm

from src.data import make_bucketed_collate_fn
from src.decode import beam_search_decode, fast_beam_search_decode, greedy_decode
from src.model import Transformer, make_src_mask
from src.tokenizer import SentencePieceTokenizer
from train import resolve_device


def average_checkpoints(checkpoint_paths: list[str | Path]) -> tuple[dict[str, torch.Tensor], dict]:
    """论文 §6.1: 对最近 K 个 Checkpoint 的模型浮点参数逐张量求算术平均。"""
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths 不能为空")

    avg_state: dict[str, torch.Tensor] = {}
    base_cfg: dict = {}
    k = len(checkpoint_paths)

    for idx, p in enumerate(checkpoint_paths):
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]
        if idx == 0:
            base_cfg = ckpt["config"]
            for key, val in state.items():
                if val.is_floating_point():
                    avg_state[key] = val.float().clone() / k
                else:
                    avg_state[key] = val.clone()
        else:
            for key, val in state.items():
                if val.is_floating_point():
                    avg_state[key] += val.float() / k

    return avg_state, base_cfg


def load_model_and_tokenizers(
    ckpt_paths: list[Path],
    device: torch.device,
) -> tuple[Transformer, SentencePieceTokenizer, SentencePieceTokenizer, dict]:
    """加载单个或多个平均后的 Checkpoint 并构建模型与中英分词器。"""
    if len(ckpt_paths) == 1:
        ckpt = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]
        cfg = ckpt["config"]
        print(f"已加载单模型权重: {ckpt_paths[0]} (step={ckpt.get('step')})")
    else:
        print(f"正在对 {len(ckpt_paths)} 个 Checkpoint 执行权重平均 (Checkpoint Averaging):")
        for p in ckpt_paths:
            print(f"  - {p}")
        state_dict, cfg = average_checkpoints(ckpt_paths)

    ckpt_dir = ckpt_paths[0].parent
    data_dir = Path(cfg["data"]["processed_dir"])
    zh_model_file = ckpt_dir / "zh_spm.model" if (ckpt_dir / "zh_spm.model").exists() else data_dir / "zh_spm.model"
    en_model_file = ckpt_dir / "en_spm.model" if (ckpt_dir / "en_spm.model").exists() else data_dir / "en_spm.model"

    zh_tok = SentencePieceTokenizer(zh_model_file)
    en_tok = SentencePieceTokenizer(en_model_file)

    m_cfg = cfg["model"]
    # 自动兼容旧版 Post-LN 权重与新版 Pre-LN 权重，以及 2-way / 3-way 词表绑定
    norm_first = bool(m_cfg.get("norm_first", "enc_norm.weight" in state_dict))
    tie_all = bool(m_cfg.get("tie_all_embeddings", False))
    model = Transformer(
        src_vocab_size=int(m_cfg.get("src_vocab_size", zh_tok.vocab_size)),
        tgt_vocab_size=int(m_cfg.get("tgt_vocab_size", en_tok.vocab_size)),
        d_model=int(m_cfg["d_model"]),
        num_heads=int(m_cfg["num_heads"]),
        num_encoder_layers=int(m_cfg["num_encoder_layers"]),
        num_decoder_layers=int(m_cfg["num_decoder_layers"]),
        d_ff=int(m_cfg["d_ff"]),
        max_len=int(m_cfg.get("max_len", 512)),
        dropout=0.0,
        pad_id=int(m_cfg.get("pad_id", 0)),
        tie_tgt_weights=bool(m_cfg.get("tie_tgt_weights", True)),
        tie_all_embeddings=tie_all,
        norm_first=norm_first,
        use_flash_attn=bool(m_cfg.get("use_flash_attn", True)),
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, zh_tok, en_tok, cfg


@torch.no_grad()
def translate_sentence(
    text: str,
    model: Transformer,
    zh_tok: SentencePieceTokenizer,
    en_tok: SentencePieceTokenizer,
    device: torch.device,
    beam_size: int = 4,
    alpha: float = 0.6,
    max_extra_len: int = 50,
    fast_beam: bool = True,
) -> str:
    """将单句中文翻译为英文（beam_size=1 走 Greedy，beam_size>1 可配置走批量化或原版手写 Beam Search）。"""
    zh_ids = zh_tok.encode(text)
    if not zh_ids:
        return ""
    src = torch.tensor([zh_ids + [SentencePieceTokenizer.EOS_ID]], dtype=torch.long, device=device)
    src_mask = make_src_mask(src, pad_id=SentencePieceTokenizer.PAD_ID)
    max_len = min(src.size(1) + max_extra_len, 256)

    if beam_size <= 1:
        out_ids = greedy_decode(
            model,
            src,
            src_mask,
            max_len=max_len,
            bos_id=SentencePieceTokenizer.BOS_ID,
            eos_id=SentencePieceTokenizer.EOS_ID,
            pad_id=SentencePieceTokenizer.PAD_ID,
        )[0].cpu().tolist()
    else:
        beam_fn = fast_beam_search_decode if fast_beam else beam_search_decode
        out_ids = beam_fn(
            model,
            src,
            src_mask,
            max_len=max_len,
            beam_size=beam_size,
            alpha=alpha,
            bos_id=SentencePieceTokenizer.BOS_ID,
            eos_id=SentencePieceTokenizer.EOS_ID,
            pad_id=SentencePieceTokenizer.PAD_ID,
        )
    return en_tok.decode(out_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="评测测试集 SacreBLEU 或开启命令行中译英交互体验")
    parser.add_argument("--ckpt", type=str, nargs="*", default=None, help="单个或多个 checkpoint 路径")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="自动查找目录下全部 step_*.pt 做权重平均")
    parser.add_argument("--data-dir", type=str, default=None, help="可选：覆盖评测用的 dataset.pt 所在目录")
    parser.add_argument("--save-avg", type=str, default=None, help="可选：将平均后的权重另存为单个 .pt 文件")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test", help="评测数据集划分")
    parser.add_argument("--beam", type=int, default=4, help="束搜索宽度 (设 1 则使用批量 Greedy 极速评测)")
    parser.add_argument("--max-samples", type=int, default=None, help="限制评测样本条数（默认全量评测）")
    parser.add_argument("--device", type=str, default="mps", help="推理设备 (mps/cuda/cpu)")
    parser.add_argument("--interactive", action="store_true", help="进入交互式输入中文实时翻译模式")
    args = parser.parse_args()

    ckpt_paths: list[Path] = []
    if args.ckpt:
        ckpt_paths = [Path(p) for p in args.ckpt]
    elif args.ckpt_dir:
        cdir = Path(args.ckpt_dir)
        step_files = sorted(cdir.glob("step_*.pt"))
        ckpt_paths = step_files if step_files else ([cdir / "best.pt"] if (cdir / "best.pt").exists() else [])
    if not ckpt_paths:
        raise ValueError("请通过 --ckpt 指定权重文件，或通过 --ckpt-dir 指定包含 step_*.pt / best.pt 的目录")

    device = resolve_device(args.device)
    model, zh_tok, en_tok, cfg = load_model_and_tokenizers(ckpt_paths, device)

    if args.save_avg:
        out_p = Path(args.save_avg)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"step": "averaged", "model_state_dict": model.state_dict(), "config": cfg}, out_p)
        print(f"已将平均权重导出至: {out_p}")

    decode_cfg = cfg.get("decode", {})
    alpha = float(decode_cfg.get("length_penalty_alpha", 0.6))
    max_extra_len = int(decode_cfg.get("max_extra_len", 50))
    use_fast_beam = bool(decode_cfg.get("fast_beam", True))

    if args.interactive:
        print(f"\n=== 进入中译英实时交互体验 (fast_beam={use_fast_beam}, 输入 q 或 exit 退出) ===")
        while True:
            try:
                zh_text = input("\n中文 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n已退出。")
                break
            if not zh_text or zh_text.lower() in {"q", "quit", "exit"}:
                break
            t0 = time.time()
            greedy_en = translate_sentence(zh_text, model, zh_tok, en_tok, device, beam_size=1, max_extra_len=max_extra_len)
            beam_en = translate_sentence(
                zh_text,
                model,
                zh_tok,
                en_tok,
                device,
                beam_size=args.beam,
                alpha=alpha,
                max_extra_len=max_extra_len,
                fast_beam=use_fast_beam,
            )
            dt_ms = (time.time() - t0) * 1000
            print(f"Greedy          > {greedy_en}")
            print(f"Beam (k={args.beam}, α={alpha}) > {beam_en}  ({dt_ms:.0f} ms)")
        return

    # 离线测试集 SacreBLEU 评测
    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg["data"]["processed_dir"])
    if not (data_dir / "dataset.pt").exists():
        # 若是从 Colab 下载回本地的权重，自动回退查找本地同名 data/<mode>/dataset.pt
        local_fallback = Path("data") / ckpt_paths[0].parent.name
        if (local_fallback / "dataset.pt").exists():
            data_dir = local_fallback
    dataset_path = data_dir / "dataset.pt"
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    pairs: list[tuple[list[int], list[int]]] = payload[args.split]
    raw_pairs: list[tuple[str, str]] = payload[f"{args.split}_raw"]

    if args.max_samples:
        pairs = pairs[: args.max_samples]
        raw_pairs = raw_pairs[: args.max_samples]

    refs = [en for _, en in raw_pairs]
    hyps: list[str] = []

    print(f"\n正在评测 {args.split} 集 (共 {len(pairs)} 句, beam_size={args.beam}) ...")
    if args.beam <= 1:
        collate = make_bucketed_collate_fn(pad_to_multiple=8)
        for start in tqdm(range(0, len(pairs), 64), desc="Greedy Eval"):
            batch = collate(pairs[start : start + 64])
            src = batch["src"].to(device)
            src_mask = batch["src_mask"].to(device)
            preds = greedy_decode(model, src, src_mask, max_len=min(src.size(1) + max_extra_len, 256))
            for row in preds.cpu().tolist():
                hyps.append(en_tok.decode(row))
    else:
        for zh_str, _ in tqdm(raw_pairs, desc=f"Beam-{args.beam} Eval"):
            hyps.append(
                translate_sentence(
                    zh_str,
                    model,
                    zh_tok,
                    en_tok,
                    device,
                    beam_size=args.beam,
                    alpha=alpha,
                    max_extra_len=max_extra_len,
                    fast_beam=use_fast_beam,
                )
            )

    bleu_res = sacrebleu.corpus_bleu(hyps, [refs])
    print(f"\n================ 评测结果 ({args.split}) ================")
    print(f"SacreBLEU = {bleu_res.score:.2f}  ({bleu_res.format()})")
    print("---------------------------------------------------------")
    for i in range(min(5, len(hyps))):
        print(f"[{i+1}] 源(Zh): {raw_pairs[i][0]}")
        print(f"    译(Pr): {hyps[i]}")
        print(f"    参(En): {refs[i]}")


if __name__ == "__main__":
    main()
