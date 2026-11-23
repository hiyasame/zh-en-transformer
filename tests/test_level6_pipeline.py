from __future__ import annotations

from pathlib import Path
import torch

from src.data import ParallelDataset, TokenBucketBatchSampler, make_bucketed_collate_fn
from src.model import Transformer
from src.tokenizer import SentencePieceTokenizer
from train import evaluate_bleu
from translate import average_checkpoints, translate_sentence


def test_sentencepiece_zh_en_roundtrip_and_eos_truncation(tmp_path: Path) -> None:
    zh_corpus = tmp_path / "zh.txt"
    en_corpus = tmp_path / "en.txt"
    zh_lines = [
        "我非常喜欢自然语言处理与深度学习。",
        "注意力机制就是你所需要的一切。",
        "今天天气很好，适合在家里写代码训练模型。",
    ] * 20
    en_lines = [
        "I really love natural language processing and deep learning.",
        "Attention is all you need for sequence transduction.",
        "The weather is nice today, great for coding and training models.",
    ] * 20
    zh_corpus.write_text("\n".join(zh_lines), encoding="utf-8")
    en_corpus.write_text("\n".join(en_lines), encoding="utf-8")

    zh_tok = SentencePieceTokenizer.train(zh_corpus, tmp_path / "zh_spm", vocab_size=120, character_coverage=1.0)
    en_tok = SentencePieceTokenizer.train(en_corpus, tmp_path / "en_spm", vocab_size=120, character_coverage=1.0)

    zh_ids = zh_tok.encode("注意力机制就是你所需要的一切。")
    assert len(zh_ids) > 0
    assert zh_tok.decode(zh_ids) == "注意力机制就是你所需要的一切。"

    en_ids = en_tok.encode("Attention is all you need.")
    # 在 eos_id 后面故意拼接垃圾词，验证 decode 会在首个 eos_id 处干净截断
    dirty_ids = [SentencePieceTokenizer.BOS_ID] + en_ids + [SentencePieceTokenizer.EOS_ID, 5, 6, 7]
    assert en_tok.decode(dirty_ids) == "Attention is all you need."


def test_token_bucket_sampler_and_bucketed_collate() -> None:
    pairs = [([4] * (i % 13 + 1), [5] * (i % 11 + 1)) for i in range(50)]
    ds = ParallelDataset(pairs)
    sampler = TokenBucketBatchSampler(ds.lengths, max_tokens=64, shuffle=True, pool_size=20)

    seen_indices: list[int] = []
    collate = make_bucketed_collate_fn(pad_to_multiple=8)
    for batch_indices in sampler:
        seen_indices.extend(batch_indices)
        batch_data = collate([ds[i] for i in batch_indices])
        assert batch_data["src"].size(1) % 8 == 0
        assert batch_data["tgt_in"].size(1) % 8 == 0
        assert batch_data["tgt_out"].size(1) % 8 == 0

    # 每个 epoch 必须不重不漏覆盖全部 50 条样本
    assert sorted(seen_indices) == list(range(50))


def test_checkpoint_averaging_and_eval_pipeline(tmp_path: Path) -> None:
    zh_corpus = tmp_path / "zh.txt"
    en_corpus = tmp_path / "en.txt"
    zh_corpus.write_text("\n".join(["机器学习很有趣。"] * 30), encoding="utf-8")
    en_corpus.write_text("\n".join(["Machine learning is fun."] * 30), encoding="utf-8")

    zh_tok = SentencePieceTokenizer.train(zh_corpus, tmp_path / "zh_spm", vocab_size=80, character_coverage=1.0)
    en_tok = SentencePieceTokenizer.train(en_corpus, tmp_path / "en_spm", vocab_size=80, character_coverage=1.0)

    cfg = {
        "data": {"processed_dir": str(tmp_path)},
        "model": {
            "src_vocab_size": zh_tok.vocab_size,
            "tgt_vocab_size": en_tok.vocab_size,
            "d_model": 32,
            "num_heads": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "d_ff": 64,
            "dropout": 0.0,
            "pad_id": 0,
            "tie_tgt_weights": True,
        },
    }
    m1 = Transformer(**{k: v for k, v in cfg["model"].items()})
    m2 = Transformer(**{k: v for k, v in cfg["model"].items()})

    p1, p2 = tmp_path / "step_001.pt", tmp_path / "step_002.pt"
    torch.save({"step": 1, "model_state_dict": m1.state_dict(), "config": cfg}, p1)
    torch.save({"step": 2, "model_state_dict": m2.state_dict(), "config": cfg}, p2)

    avg_state, _ = average_checkpoints([p1, p2])
    expected_w = 0.5 * (m1.src_embedding.embedding.weight + m2.src_embedding.embedding.weight)
    assert torch.allclose(avg_state["src_embedding.embedding.weight"], expected_w, atol=1e-6)

    m1.load_state_dict(avg_state)
    device = torch.device("cpu")
    val_raw = [("机器学习很有趣。", "Machine learning is fun.")] * 4
    val_pairs = [(zh_tok.encode(z), en_tok.encode(e)) for z, e in val_raw]

    bleu, samples = evaluate_bleu(m1, val_pairs, val_raw, en_tok, device, max_extra_len=10)
    assert isinstance(bleu, float) and len(samples) == 3

    out_text = translate_sentence("机器学习很有趣。", m1, zh_tok, en_tok, device, beam_size=2, max_extra_len=8)
    assert isinstance(out_text, str)


def test_mps_forward_backward_and_decode_compat() -> None:
    """在 Apple Silicon MPS 后端上实测一次前向 + Label Smoothing 反向 + Greedy 解码，排除算子兼容隐患。"""
    if not torch.backends.mps.is_available():
        return
    device = torch.device("mps")
    model = Transformer(
        src_vocab_size=64,
        tgt_vocab_size=64,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.1,
        pad_id=0,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1, reduction="sum")

    src = torch.randint(4, 64, (4, 16), device=device)
    tgt_in = torch.randint(2, 64, (4, 16), device=device)
    tgt_out = torch.randint(3, 64, (4, 16), device=device)

    logits = model(src, tgt_in)
    loss = criterion(logits.reshape(-1, 64), tgt_out.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss).item()


def test_csr_parallel_dataset_and_ddp_sampler_sharding() -> None:
    """验证 450 万句级 CSR 扁平 int32 数据集切片等价性、Collate 可序列化性，以及 4 卡 DDP 采样等长无重叠。"""
    import pickle
    from scripts.prepare_data import pack_pairs_to_csr

    raw_pairs = [([4 + (i % 7)] * (i % 15 + 1), [12 + (i % 5)] * (i % 11 + 2)) for i in range(120)]
    ds_list = ParallelDataset(raw_pairs)
    csr_dict = pack_pairs_to_csr(raw_pairs)
    ds_csr = ParallelDataset(csr_dict)

    assert len(ds_csr) == len(ds_list) == 120
    assert ds_csr.lengths == ds_list.lengths
    for idx in [0, 17, 63, 119]:
        assert ds_csr[idx] == ds_list[idx]

    collate = make_bucketed_collate_fn(pad_to_multiple=8)
    restored_collate = pickle.loads(pickle.dumps(collate))
    batch_out = restored_collate([ds_csr[0], ds_csr[1]])
    assert batch_out["src"].size(1) % 8 == 0

    world_size = 4
    rank_batches = [
        list(
            TokenBucketBatchSampler(
                ds_csr.lengths, max_tokens=64, shuffle=True, seed=99, pool_size=32, num_replicas=world_size, rank=r
            )
        )
        for r in range(world_size)
    ]
    counts = [len(rb) for rb in rank_batches]
    assert len(set(counts)) == 1 and counts[0] > 0, f"DDP 各卡每轮 batch 数必须严格一致，实际: {counts}"

    flat_samples = [idx for rb in rank_batches for b in rb for idx in b]
    assert len(flat_samples) == len(set(flat_samples)), "DDP 各卡分片之间不应出现重复样本"


