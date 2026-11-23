from __future__ import annotations

import argparse
from array import array
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import random
import shutil
import sys
import yaml
import sentencepiece as spm
import torch
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import clean_zh_en_pair
from src.tokenizer import SentencePieceTokenizer

# 多进程 Worker 内部缓存的 SentencePiece 实例，避免重复反序列化
_WORKER_ZH_SP: spm.SentencePieceProcessor | None = None
_WORKER_EN_SP: spm.SentencePieceProcessor | None = None


def _init_encode_worker(zh_model_path: str, en_model_path: str) -> None:
    global _WORKER_ZH_SP, _WORKER_EN_SP
    _WORKER_ZH_SP = spm.SentencePieceProcessor(model_file=zh_model_path)
    _WORKER_EN_SP = spm.SentencePieceProcessor(model_file=en_model_path)


def _encode_chunk_worker(
    args: tuple[list[tuple[str, str]], int, bool],
) -> tuple[list[tuple[list[int], list[int]]], list[tuple[str, str]]]:
    chunk, body_limit, drop_long = args
    assert _WORKER_ZH_SP is not None and _WORKER_EN_SP is not None
    encoded: list[tuple[list[int], list[int]]] = []
    kept_raw: list[tuple[str, str]] = []
    for zh, en in chunk:
        zh_ids = list(_WORKER_ZH_SP.encode(zh, out_type=int))
        en_ids = list(_WORKER_EN_SP.encode(en, out_type=int))
        if not zh_ids or not en_ids:
            continue
        if drop_long:
            if len(zh_ids) > body_limit or len(en_ids) > body_limit:
                continue
        else:
            zh_ids = zh_ids[:body_limit]
            en_ids = en_ids[:body_limit]
        encoded.append((zh_ids, en_ids))
        kept_raw.append((zh, en))
    return encoded, kept_raw


def collect_clean_pairs(
    iterable,
    limit: int | None,
    seen: set[int],
    max_chars: int = 350,
    desc: str = "Cleaning",
) -> list[tuple[str, str]]:
    """遍历 HuggingFace 数据集分片，执行中英清洗与基于 64 位哈希的低内存全局去重。"""
    pairs: list[tuple[str, str]] = []
    pbar = tqdm(total=limit, desc=desc, unit="pair") if limit else tqdm(desc=desc, unit="pair")
    for row in iterable:
        trans = row.get("translation", row)
        zh = str(trans.get("zh", "")).strip()
        en = str(trans.get("en", "")).strip()
        if not clean_zh_en_pair(zh, en, min_chars=2, max_chars=max_chars):
            continue
        h = hash((zh, en))
        if h in seen:
            continue
        seen.add(h)
        pairs.append((zh, en))
        pbar.update(1)
        if limit is not None and len(pairs) >= limit:
            break
    pbar.close()
    return pairs


def pack_pairs_to_csr(pairs: list[tuple[list[int], list[int]]]) -> dict[str, torch.Tensor]:
    """将百万级变长 Token ID 序列打包为紧凑连续的 CSR int32 张量（省去 90% Python 对象内存开销，秒级加载）。"""
    src_buf = array("i")
    tgt_buf = array("i")
    src_offsets = array("q", [0])
    tgt_offsets = array("q", [0])

    for s_ids, t_ids in pairs:
        src_buf.extend(s_ids)
        tgt_buf.extend(t_ids)
        src_offsets.append(len(src_buf))
        tgt_offsets.append(len(tgt_buf))

    return {
        "src_flat": torch.frombuffer(memoryview(src_buf), dtype=torch.int32).clone(),
        "src_offsets": torch.frombuffer(memoryview(src_offsets), dtype=torch.int64).clone(),
        "tgt_flat": torch.frombuffer(memoryview(tgt_buf), dtype=torch.int32).clone(),
        "tgt_offsets": torch.frombuffer(memoryview(tgt_offsets), dtype=torch.int64).clone(),
    }


def encode_split_parallel(
    raw_pairs: list[tuple[str, str]],
    zh_model_path: Path,
    en_model_path: Path,
    max_seq_len: int,
    drop_long: bool,
    max_keep: int | None = None,
) -> tuple[list[tuple[list[int], list[int]]], list[tuple[str, str]]]:
    """多核并行编码中英句对（16 核下 450 万句仅需约 30~45 秒）。"""
    body_limit = max_seq_len - 1
    n_workers = min(max(1, os.cpu_count() or 4), 16)

    if len(raw_pairs) < 10_000 or n_workers == 1:
        _init_encode_worker(str(zh_model_path), str(en_model_path))
        enc, raw = _encode_chunk_worker((raw_pairs, body_limit, drop_long))
        if max_keep is not None:
            return enc[:max_keep], raw[:max_keep]
        return enc, raw

    chunk_size = 25_000
    chunks = [
        (raw_pairs[i : i + chunk_size], body_limit, drop_long)
        for i in range(0, len(raw_pairs), chunk_size)
    ]

    all_encoded: list[tuple[list[int], list[int]]] = []
    all_raw: list[tuple[str, str]] = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_encode_worker,
        initargs=(str(zh_model_path), str(en_model_path)),
    ) as pool:
        for enc_part, raw_part in tqdm(
            pool.map(_encode_chunk_worker, chunks),
            total=len(chunks),
            desc=f"Parallel Encoding ({n_workers} cores)",
        ):
            all_encoded.extend(enc_part)
            all_raw.extend(raw_part)
            if max_keep is not None and len(all_encoded) >= max_keep:
                all_encoded = all_encoded[:max_keep]
                all_raw = all_raw[:max_keep]
                break

    return all_encoded, all_raw


def main() -> None:
    parser = argparse.ArgumentParser(description="下载清洗中英平行语料、训练 SentencePiece 并导出预分词包")
    parser.add_argument("--config", type=str, default="configs/local_small.yaml", help="YAML 配置文件路径")
    parser.add_argument("--out-dir", type=str, default=None, help="覆盖预处理数据输出目录")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.out_dir:
        cfg["data"]["processed_dir"] = args.out_dir

    data_cfg = cfg["data"]
    max_train_samples: int = int(data_cfg["max_train_samples"])
    max_seq_len: int = int(data_cfg["max_seq_len"])
    shared_vocab: bool = bool(data_cfg.get("shared_vocab", False))
    out_dir = Path(data_cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_char_cap = min(400, max_seq_len * 5)
    seen_hashes: set[int] = set()
    train_raw: list[tuple[str, str]] = []
    val_raw: list[tuple[str, str]] = []
    test_raw: list[tuple[str, str]] = []

    # 支持多域混合数据源 (sources 列表) 或传统单数据源
    sources = data_cfg.get("sources")
    if sources:
        print(f"[1/4] 正在拉取多域混合语料 (共 {len(sources)} 个数据源，目标总量 {max_train_samples:,} 句) ...")
        for src_spec in sources:
            s_name = src_spec["name"]
            s_cfg = src_spec["config"]
            s_max = int(src_spec["max_samples"] * 1.12)
            s_stream = bool(src_spec.get("streaming", False))
            print(f"  -> 加载数据源: {s_name} ({s_cfg}), streaming={s_stream}, 上限={s_max:,}")
            if s_stream:
                ds_iter = load_dataset(s_name, s_cfg, split="train", streaming=True)
                # 打乱底层 Parquet 分片与缓冲池，确保均匀覆盖新闻/百科/对话而非只读单一子语料分片
                ds_iter = ds_iter.shuffle(seed=42, buffer_size=50_000)
                part = collect_clean_pairs(ds_iter, limit=s_max, seen=seen_hashes, max_chars=raw_char_cap, desc=s_name)
            else:
                ds_full = load_dataset(s_name, s_cfg, split="train")
                part = collect_clean_pairs(ds_full, limit=s_max, seen=seen_hashes, max_chars=raw_char_cap, desc=s_name)
            train_raw.extend(part)

        rng = random.Random(42)
        rng.shuffle(train_raw)

        # 验证与测试集优先采用官方 WMT 新闻人工基准 (wmt19 validation = newstest2018)
        eval_source = data_cfg.get("eval_dataset", {"name": "wmt/wmt19", "config": "zh-en"})
        print(f"  -> 加载官方评测基准: {eval_source['name']} ({eval_source['config']}) ...")
        ds_eval = load_dataset(eval_source["name"], eval_source["config"], split="validation")
        all_eval = collect_clean_pairs(ds_eval, limit=4000, seen=set(), max_chars=450, desc="Official Eval")
        mid = len(all_eval) // 2
        val_raw, test_raw = all_eval[:mid], all_eval[mid:]
    else:
        dataset_name: str = data_cfg["dataset_name"]
        dataset_config: str = data_cfg["dataset_config"]
        fetch_target = int(max_train_samples * 1.20)
        print(f"[1/4] 正在拉取并清洗数据集: {dataset_name} ({dataset_config}) ...")
        use_streaming = "wmt" in dataset_name.lower() and max_train_samples < 6_000_000
        if use_streaming:
            ds_train = load_dataset(dataset_name, dataset_config, split="train", streaming=True).shuffle(
                seed=42, buffer_size=50_000
            )
            train_raw = collect_clean_pairs(ds_train, limit=fetch_target, seen=seen_hashes, max_chars=raw_char_cap, desc="Train")
            ds_val = load_dataset(dataset_name, dataset_config, split="validation")
            all_eval = collect_clean_pairs(ds_val, limit=4000, seen=set(), max_chars=450, desc="Validation")
            mid = len(all_eval) // 2
            val_raw, test_raw = all_eval[:mid], all_eval[mid:]
        else:
            ds = load_dataset(dataset_name, dataset_config)
            train_raw = collect_clean_pairs(ds["train"], limit=fetch_target, seen=seen_hashes, max_chars=raw_char_cap, desc="Train")
            if "validation" in ds and "test" in ds:
                val_raw = collect_clean_pairs(ds["validation"], limit=2000, seen=set(), max_chars=450, desc="Val")
                test_raw = collect_clean_pairs(ds["test"], limit=2000, seen=set(), max_chars=450, desc="Test")
            else:
                val_raw = train_raw[-4000:-2000]
                test_raw = train_raw[-2000:]
                train_raw = train_raw[:-4000]

    del seen_hashes
    print(f"    清洗去重后候选句对: train={len(train_raw):,}, val={len(val_raw):,}, test={len(test_raw):,}")

    # 抽样最多 60 万双语句对（120 万行）训练高质量 BPE 词表
    spm_sample = train_raw[: min(len(train_raw), 600_000)]

    if shared_vocab:
        joint_vocab_size = int(data_cfg.get("vocab_size", 37000))
        joint_txt = out_dir / "train_joint.txt"
        with open(joint_txt, "w", encoding="utf-8") as fj:
            for zh, en in spm_sample:
                fj.write(zh + "\n")
                fj.write(en + "\n")

        print(f"[2&3/4] 训练原论文 §3.4 中英联合共享 SentencePiece 词表 (vocab={joint_vocab_size}, coverage=0.9995) ...")
        joint_tok = SentencePieceTokenizer.train(
            input_txt_path=joint_txt,
            model_prefix=out_dir / "shared_spm",
            vocab_size=joint_vocab_size,
            character_coverage=0.9995,
            model_type="bpe",
        )
        joint_txt.unlink(missing_ok=True)
        # 复制为 zh_spm.model 与 en_spm.model，使下游训练与推理脚本零改动无缝兼容！
        shutil.copy2(out_dir / "shared_spm.model", out_dir / "zh_spm.model")
        shutil.copy2(out_dir / "shared_spm.model", out_dir / "en_spm.model")
        zh_tok = en_tok = joint_tok
    else:
        zh_vocab_size = int(data_cfg["zh_vocab_size"])
        en_vocab_size = int(data_cfg["en_vocab_size"])
        zh_txt = out_dir / "train_zh.txt"
        en_txt = out_dir / "train_en.txt"
        with open(zh_txt, "w", encoding="utf-8") as fz, open(en_txt, "w", encoding="utf-8") as fe:
            for zh, en in spm_sample:
                fz.write(zh + "\n")
                fe.write(en + "\n")

        print(f"[2/4] 训练中文 SentencePiece (vocab={zh_vocab_size}, coverage=0.9995) ...")
        zh_tok = SentencePieceTokenizer.train(
            input_txt_path=zh_txt,
            model_prefix=out_dir / "zh_spm",
            vocab_size=zh_vocab_size,
            character_coverage=0.9995,
            model_type="bpe",
        )
        print(f"[3/4] 训练英文 SentencePiece (vocab={en_vocab_size}, coverage=1.0) ...")
        en_tok = SentencePieceTokenizer.train(
            input_txt_path=en_txt,
            model_prefix=out_dir / "en_spm",
            vocab_size=en_vocab_size,
            character_coverage=1.0,
            model_type="bpe",
        )
        zh_txt.unlink(missing_ok=True)
        en_txt.unlink(missing_ok=True)

    zh_model_file = out_dir / "zh_spm.model"
    en_model_file = out_dir / "en_spm.model"

    print(f"[4/4] 多进程预编码全量语料 (max_seq_len={max_seq_len}) ...")
    train_ids, train_kept_raw = encode_split_parallel(
        train_raw,
        zh_model_file,
        en_model_file,
        max_seq_len=max_seq_len,
        drop_long=True,
        max_keep=max_train_samples,
    )
    sample_raw_64 = train_kept_raw[:64]
    del train_raw, train_kept_raw

    val_ids, val_kept_raw = encode_split_parallel(
        val_raw, zh_model_file, en_model_file, max_seq_len=max_seq_len, drop_long=False
    )
    test_ids, test_kept_raw = encode_split_parallel(
        test_raw, zh_model_file, en_model_file, max_seq_len=max_seq_len, drop_long=False
    )

    # 百万级训练集使用紧凑连续 CSR int32 格式落盘，大幅压缩体积与加载内存
    use_csr = len(train_ids) >= 200_000
    train_storage = pack_pairs_to_csr(train_ids) if use_csr else train_ids
    n_train_final = len(train_ids)
    del train_ids

    payload = {
        "train": train_storage,
        "val": val_ids,
        "test": test_ids,
        "train_sample_raw": sample_raw_64,
        "val_raw": val_kept_raw,
        "test_raw": test_kept_raw,
        "meta": {
            "zh_vocab_size": zh_tok.vocab_size,
            "en_vocab_size": en_tok.vocab_size,
            "shared_vocab": shared_vocab,
            "max_seq_len": max_seq_len,
        },
    }
    save_path = out_dir / "dataset.pt"
    torch.save(payload, save_path)

    size_mb = save_path.stat().st_size / (1024 * 1024)
    print(
        f"\n预处理完成！已保存至 {out_dir}:\n"
        f"  - dataset.pt ({size_mb:.1f} MB, CSR={use_csr}): train={n_train_final:,}, val={len(val_ids):,}, test={len(test_ids):,}\n"
        f"  - 词表规模: src={zh_tok.vocab_size:,} / tgt={en_tok.vocab_size:,} (shared={shared_vocab})"
    )


if __name__ == "__main__":
    main()
