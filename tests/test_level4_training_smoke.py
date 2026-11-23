from __future__ import annotations

import math
import torch
import torch.nn as nn

from src.model import Transformer, make_src_mask, make_tgt_mask
from src.optim import NoamScheduler
from src.decode import greedy_decode


def test_noam_scheduler_warmup_and_decay_curve() -> None:
    param = nn.Parameter(torch.randn(4, 4))
    optimizer = torch.optim.Adam([param], lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    d_model = 512
    warmup = 4000
    scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup, factor=1.0)

    # 第 1 步
    lr_1 = scheduler.step()
    expected_1 = (d_model**-0.5) * (1 * (warmup**-1.5))
    assert math.isclose(lr_1, expected_1, rel_tol=1e-5)
    assert math.isclose(optimizer.param_groups[0]["lr"], expected_1, rel_tol=1e-5)

    # 推进到第 2000 步（线性升温期中点）
    for _ in range(1999):
        scheduler.step()
    expected_2000 = (d_model**-0.5) * (2000 * (warmup**-1.5))
    assert math.isclose(scheduler.get_last_lr(), expected_2000, rel_tol=1e-5)

    # 推进到第 4000 步（学习率峰值）
    for _ in range(2000):
        scheduler.step()
    peak_lr = (d_model**-0.5) * (warmup**-0.5)
    assert math.isclose(scheduler.get_last_lr(), peak_lr, rel_tol=1e-5)

    # 测试 state_dict 断点保存与恢复，并推进到第 16000 步（应衰减为峰值的恰好 1/2）
    saved = scheduler.state_dict()
    scheduler2 = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup, factor=1.0)
    scheduler2.load_state_dict(saved)
    for _ in range(12000):
        scheduler2.step()
    assert math.isclose(scheduler2.get_last_lr(), peak_lr * 0.5, rel_tol=1e-5)


def test_stage1_copy_task_convergence() -> None:
    """Stage 1 终极冒烟测试：在合成序列复制任务上训练 300 步，验证 Greedy 解码准确率 >= 95%！"""
    torch.manual_seed(42)
    vocab_size = 14  # 0=pad, 1=unk, 2=bos, 3=eos, 4..13=普通数字词
    pad_id, bos_id, eos_id = 0, 2, 3
    seq_len = 6
    batch_size = 32

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.0,
        pad_id=pad_id,
        tie_tgt_weights=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    # 注意：d_model=64 时 d_model^-0.5 = 0.125（比 d_model=512 时大了近 3 倍），且 warmup=80 很短，
    # 设 factor=0.2 使峰值学习率落在约 2.8e-3 的合理区间，避免过大 lr 冲垮小模型参数
    scheduler = NoamScheduler(optimizer, d_model=64, warmup_steps=80, factor=0.2)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)

    model.train()
    for _ in range(300):
        tokens = torch.randint(4, vocab_size, (batch_size, seq_len))
        eos_col = torch.full((batch_size, 1), eos_id, dtype=torch.long)
        bos_col = torch.full((batch_size, 1), bos_id, dtype=torch.long)

        src = torch.cat([tokens, eos_col], dim=1)            # [w1..w6, <eos>]
        tgt_in = torch.cat([bos_col, tokens], dim=1)         # [<bos>, w1..w6]
        tgt_out = torch.cat([tokens, eos_col], dim=1)        # [w1..w6, <eos>]

        src_mask = make_src_mask(src, pad_id=pad_id)
        tgt_mask = make_tgt_mask(tgt_in, pad_id=pad_id)

        logits = model(src, tgt_in, src_mask, tgt_mask)
        loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scheduler.step()
        optimizer.step()

    # 验收：用未见过的新随机批次测试自回归 Greedy 解码
    model.eval()
    eval_tokens = torch.randint(4, vocab_size, (16, seq_len))
    eval_src = torch.cat([eval_tokens, torch.full((16, 1), eos_id, dtype=torch.long)], dim=1)
    eval_src_mask = make_src_mask(eval_src, pad_id=pad_id)

    pred = greedy_decode(
        model,
        eval_src,
        eval_src_mask,
        max_len=seq_len + 2,  # <bos> + 6 tokens + <eos>
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
    )
    assert pred.shape[0] == 16 and pred.shape[1] >= seq_len + 1
    assert (pred[:, 0] == bos_id).all(), "greedy_decode 返回序列第 0 列应为 bos_id"

    pred_body = pred[:, 1 : 1 + seq_len]
    accuracy = (pred_body == eval_tokens).float().mean().item()
    assert accuracy >= 0.95, f"Copy Task 贪心解码准确率仅为 {accuracy:.2%} (< 95%)，请排查模型或解码逻辑！"
