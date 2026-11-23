from __future__ import annotations

import math
import torch
import pytest

from src.model import (
    make_src_mask,
    make_tgt_mask,
    scaled_dot_product_attention,
    MultiHeadAttention,
)


def test_make_src_mask_shape_and_values() -> None:
    src = torch.tensor([[5, 8, 9, 0, 0], [4, 7, 6, 3, 0]])
    mask = make_src_mask(src, pad_id=0)

    assert mask.shape == (2, 1, 1, 5), f"期望形状 (2, 1, 1, 5)，实际得到 {mask.shape}"
    assert mask.dtype == torch.bool, "建议返回 torch.bool 类型掩码 (True=有效, False=<pad>)"
    expected = torch.tensor(
        [
            [[[True, True, True, False, False]]],
            [[[True, True, True, True, False]]],
        ]
    )
    assert torch.equal(mask, expected), "源端 Padding Mask 布尔值与预期不符"


def test_make_tgt_mask_combines_causal_and_padding() -> None:
    # 第 0 条样本末尾含 1 个 <pad>(0)
    tgt = torch.tensor([[2, 11, 12, 0]])
    mask = make_tgt_mask(tgt, pad_id=0)

    assert mask.shape == (1, 1, 4, 4), f"期望形状 (1, 1, 4, 4)，实际得到 {mask.shape}"
    expected = torch.tensor(
        [
            [
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, True, True, False],
                    [True, True, True, False],  # 第 3 列是 <pad>，即使 j<=i 也必须为 False
                ]
            ]
        ]
    )
    assert torch.equal(mask, expected), "目标端 Mask 需同时屏蔽上三角未来位置与 <pad> 列"


def test_scaled_dot_product_attention_exact_math() -> None:
    # 用 d_k = 2 的精确手工算例检验是否漏掉了 1 / sqrt(d_k) 缩放
    q = torch.tensor([[[[1.0, 0.0], [0.0, 2.0]]]])  # (1, 1, 2, 2)
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])  # (1, 1, 2, 2)
    v = torch.tensor([[[[10.0, 0.0], [0.0, 20.0]]]])  # (1, 1, 2, 2)

    out, attn = scaled_dot_product_attention(q, k, v)

    assert out.shape == (1, 1, 2, 2)
    assert attn.shape == (1, 1, 2, 2)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(1, 1, 2), atol=1e-6), "注意力权重最后一维求和必须为 1"

    expected_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(2.0)
    expected_attn = torch.softmax(expected_scores, dim=-1)
    expected_out = torch.matmul(expected_attn, v)

    assert torch.allclose(attn, expected_attn, atol=1e-5), "注意力权重数值不匹配，请检查是否除以了 sqrt(d_k)"
    assert torch.allclose(out, expected_out, atol=1e-5)


def test_mha_self_and_cross_attention_shapes() -> None:
    torch.manual_seed(42)
    mha = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0)
    mha.eval()

    # 自注意力: L_q == L_k == 8
    x = torch.randn(2, 8, 32)
    self_out = mha(x, x, x)
    assert self_out.shape == (2, 8, 32)

    # 交叉注意力: L_q = 6 != L_k = 11 (检验拆头时是否写死了序列长度)
    q = torch.randn(2, 6, 32)
    kv = torch.randn(2, 11, 32)
    src_mask = torch.ones(2, 1, 1, 11, dtype=torch.bool)
    src_mask[:, :, :, -2:] = False
    cross_out = mha(q, kv, kv, mask=src_mask)
    assert cross_out.shape == (2, 6, 32)


def test_mha_causal_no_future_leakage() -> None:
    """核心防坑测试：修改未来时刻的输入，绝不能影响过去时刻的输出（同时也能抓出 view/transpose 顺序写反的 Bug）。"""
    torch.manual_seed(7)
    mha = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.1)
    mha.eval()

    x1 = torch.randn(2, 10, 64)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn(2, 5, 64) * 10.0  # 剧烈篡改第 5~9 个未来位置

    dummy_ids = torch.ones(2, 10, dtype=torch.long)
    causal_mask = make_tgt_mask(dummy_ids, pad_id=0)

    out1 = mha(x1, x1, x1, mask=causal_mask)
    out2 = mha(x2, x2, x2, mask=causal_mask)

    assert torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-5), (
        "检测到未来信息泄露！篡改位置 5~9 的输入后，位置 0~4 的输出发生了变化。"
        "请检查：(1) mask==0 是否在 softmax 前置为极小负数；(2) 拆多头时是否先 view(B, L, h, d_k) 再 transpose(1, 2)。"
    )


def test_mha_padding_invariance() -> None:
    """在序列末尾追加被 Mask 屏蔽的 <pad> 向量，不应改变有效位置的输出。"""
    torch.manual_seed(99)
    mha = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0)
    mha.eval()

    x_valid = torch.randn(1, 5, 32)
    pad_garbage = torch.randn(1, 3, 32) * 50.0
    x_padded = torch.cat([x_valid, pad_garbage], dim=1)

    mask_valid = make_src_mask(torch.tensor([[1, 2, 3, 4, 5]]), pad_id=0)
    mask_padded = make_src_mask(torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]]), pad_id=0)

    out_valid = mha(x_valid, x_valid, x_valid, mask=mask_valid)
    out_padded = mha(x_padded, x_padded, x_padded, mask=mask_padded)

    assert torch.allclose(out_valid, out_padded[:, :5, :], atol=1e-5), (
        "末尾被屏蔽的 <pad> 影响了有效位置的注意力输出，请检查 mask 广播维度是否为最后一维 Key 长度。"
    )


def test_flash_attention_matches_handwritten_mha() -> None:
    """验证 use_flash_attn=True 与 use_flash_attn=False (手写版) 在相同权重与掩码下输出完全一致。"""
    torch.manual_seed(123)
    mha_flash = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.1, use_flash_attn=True).eval()
    mha_manual = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.1, use_flash_attn=False).eval()
    mha_manual.load_state_dict(mha_flash.state_dict())

    q = torch.randn(2, 7, 64)
    kv = torch.randn(2, 11, 64)
    src_mask = make_src_mask(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0], [1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0]]))
    tgt_mask = make_tgt_mask(torch.tensor([[2, 4, 5, 6, 7, 0, 0], [2, 8, 9, 10, 11, 12, 0]]))

    assert torch.allclose(mha_flash(q, kv, kv, mask=src_mask), mha_manual(q, kv, kv, mask=src_mask), atol=1e-5)
    assert torch.allclose(mha_flash(q, q, q, mask=tgt_mask), mha_manual(q, q, q, mask=tgt_mask), atol=1e-5)
