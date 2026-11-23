from __future__ import annotations

import math
import torch

from src.model import (
    TokenEmbedding,
    PositionalEncoding,
    PositionwiseFeedForward,
)


def test_token_embedding_sqrt_d_model_scaling() -> None:
    """检验论文 §3.4：Embedding 输出是否乘以了 sqrt(d_model)。"""
    torch.manual_seed(0)
    d_model = 64
    emb = TokenEmbedding(vocab_size=50, d_model=d_model, pad_id=0)

    params = list(emb.parameters())
    assert len(params) == 1 and params[0].shape == (50, d_model)
    weight = params[0]

    x = torch.tensor([[1, 5, 12], [3, 9, 0]])
    out = emb(x)
    assert out.shape == (2, 3, d_model)

    expected = weight[x] * math.sqrt(d_model)
    assert torch.allclose(out, expected, atol=1e-5), (
        "TokenEmbedding 输出未乘以 sqrt(d_model)！权重绑定后若漏掉该缩放，语义向量会被位置编码淹没。"
    )


def test_positional_encoding_values_and_non_trainable_buffer() -> None:
    """检验论文 §3.5 正余弦公式数值及 register_buffer 行为。"""
    d_model = 16
    pe = PositionalEncoding(d_model=d_model, max_len=100, dropout=0.0)
    pe.eval()

    # 1. 不应包含任何可训练参数，但应包含至少 1 个 buffer 且存在于 state_dict 中
    assert sum(p.numel() for p in pe.parameters()) == 0, "固定正余弦位置编码不应有可训练 Parameter"
    assert len(list(pe.buffers())) >= 1, "请使用 self.register_buffer 注册位置编码矩阵"
    assert len(pe.state_dict()) >= 1, "位置编码 buffer 应随 state_dict 持久化"

    # 2. 传入全零张量，直接提取加上的 PE 数值
    zeros = torch.zeros(2, 10, d_model)
    out = pe(zeros)
    assert out.shape == (2, 10, d_model)

    # pos = 0: sin(0) = 0, cos(0) = 1
    assert torch.allclose(out[0, 0, 0::2], torch.zeros(d_model // 2), atol=1e-6)
    assert torch.allclose(out[0, 0, 1::2], torch.ones(d_model // 2), atol=1e-6)

    # pos = 3, 2i = 0 -> 10000^(0/16) = 1 -> sin(3), cos(3)
    assert math.isclose(out[0, 3, 0].item(), math.sin(3.0), abs_tol=1e-5)
    assert math.isclose(out[0, 3, 1].item(), math.cos(3.0), abs_tol=1e-5)

    # pos = 5, 2i = 8 -> 10000^(8/16) = 100 -> sin(5 / 100), cos(5 / 100)
    assert math.isclose(out[0, 5, 8].item(), math.sin(5.0 / 100.0), abs_tol=1e-5)
    assert math.isclose(out[0, 5, 9].item(), math.cos(5.0 / 100.0), abs_tol=1e-5)


def test_positionwise_ffn_shape_and_backward() -> None:
    torch.manual_seed(1)
    ffn = PositionwiseFeedForward(d_model=32, d_ff=128, dropout=0.1)
    x = torch.randn(3, 9, 32, requires_grad=True)

    out = ffn(x)
    assert out.shape == (3, 9, 32)

    out.sum().backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
