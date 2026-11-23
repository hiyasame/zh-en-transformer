from __future__ import annotations

import torch
import torch.nn as nn

from src.model import (
    make_src_mask,
    make_tgt_mask,
    EncoderLayer,
    DecoderLayer,
    Transformer,
)


def test_encoder_and_decoder_layer_shapes() -> None:
    enc_layer = EncoderLayer(d_model=64, num_heads=4, d_ff=256, dropout=0.1)
    dec_layer = DecoderLayer(d_model=64, num_heads=4, d_ff=256, dropout=0.1)

    src_x = torch.randn(2, 11, 64)
    tgt_x = torch.randn(2, 7, 64)
    src_mask = torch.ones(2, 1, 1, 11, dtype=torch.bool)
    tgt_mask = make_tgt_mask(torch.ones(2, 7, dtype=torch.long), pad_id=0)

    memory = enc_layer(src_x, src_mask)
    assert memory.shape == (2, 11, 64)

    dec_out = dec_layer(tgt_x, memory, src_mask, tgt_mask)
    assert dec_out.shape == (2, 7, 64)


def test_transformer_asymmetric_zh_en_forward() -> None:
    """检验中英不同词表大小 (src_vocab != tgt_vocab) 与不同句长下的前向一致性。"""
    torch.manual_seed(0)
    model = Transformer(
        src_vocab_size=120,
        tgt_vocab_size=200,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.1,
        pad_id=0,
    )
    model.eval()

    src = torch.randint(4, 120, (3, 13))
    src[0, -2:] = 0
    tgt = torch.randint(4, 200, (3, 8))
    tgt[1, -1:] = 0

    src_mask = make_src_mask(src, pad_id=0)
    tgt_mask = make_tgt_mask(tgt, pad_id=0)

    logits_direct = model(src, tgt, src_mask, tgt_mask)
    assert logits_direct.shape == (3, 8, 200)

    memory = model.encode(src, src_mask)
    assert memory.shape == (3, 13, 64)
    dec_out = model.decode(tgt, memory, src_mask, tgt_mask)
    assert dec_out.shape == (3, 8, 64)
    logits_stepwise = model.project(dec_out)

    assert torch.allclose(logits_direct, logits_stepwise, atol=1e-5), (
        "model(src, tgt, ...) 的结果应与 encode -> decode -> project 分步调用完全一致"
    )


def test_transformer_end_to_end_causal_no_leakage() -> None:
    """整网端到端因果测试：确保 Decoder 每一层都没有把 tgt_mask 与 src_mask 传混。"""
    torch.manual_seed(12)
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=64,
        num_heads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=128,
        dropout=0.1,
        pad_id=0,
    )
    model.eval()

    src = torch.randint(4, 100, (2, 9))
    tgt1 = torch.randint(4, 100, (2, 8))
    tgt2 = tgt1.clone()
    tgt2[:, 4:] = torch.randint(4, 100, (2, 4))  # 修改第 4~7 个位置的目标词

    logits1 = model(src, tgt1)
    logits2 = model(src, tgt2)

    assert torch.allclose(logits1[:, :4, :], logits2[:, :4, :], atol=1e-5), (
        "Decoder 端到端检测到未来信息泄露！请检查 DecoderLayer 自注意力是否使用了因果 tgt_mask。"
    )


def test_transformer_two_way_weight_tying() -> None:
    """检验 Decoder 目标词嵌入与输出线性层是否正确绑定了同一份权重参数。"""
    src_vocab, tgt_vocab, d_model = 300, 500, 64
    model_tied = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=d_model,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        tie_tgt_weights=True,
    )
    model_untied = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=d_model,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        tie_tgt_weights=False,
    )

    n_tied = sum(p.numel() for p in model_tied.parameters())
    n_untied = sum(p.numel() for p in model_untied.parameters())
    diff = n_untied - n_tied

    assert diff == tgt_vocab * d_model, (
        f"开启 tie_tgt_weights=True 后应恰好节省 tgt_vocab * d_model = {tgt_vocab * d_model} 个参数，"
        f"但实际差值为 {diff}（并请确认输出投影层设置了 bias=False）。"
    )


def test_all_trainable_parameters_receive_gradients() -> None:
    """确认没有悬空未接入计算图的层或残差支路。"""
    model = Transformer(
        src_vocab_size=80,
        tgt_vocab_size=90,
        d_model=32,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        pad_id=0,
    )
    src = torch.randint(4, 80, (2, 6))
    tgt_in = torch.randint(2, 90, (2, 5))
    tgt_out = torch.randint(3, 90, (2, 5))

    logits = model(src, tgt_in)
    loss = nn.CrossEntropyLoss(ignore_index=0)(logits.reshape(-1, 90), tgt_out.reshape(-1))
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"参数 {name} 未收到任何梯度（可能未参与 forward 计算）"
            assert torch.isfinite(param.grad).all(), f"参数 {name} 的梯度出现了 NaN 或 Inf"


def test_large_vocab_embedding_unit_variance_and_preln_postln() -> None:
    """确保大词表 (V=20000) 下词嵌入乘完 sqrt(d_model) 后标准差接近 1.0（不被 Xavier 压扁），且兼容 Pre-LN / Post-LN。"""
    torch.manual_seed(0)
    model_pre = Transformer(
        src_vocab_size=20000,
        tgt_vocab_size=16000,
        d_model=512,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=1024,
        pad_id=0,
        norm_first=True,
    )
    sample_ids = torch.randint(4, 20000, (16, 64))
    emb_out = model_pre.src_embedding(sample_ids)
    std = emb_out.std().item()
    assert 0.85 <= std <= 1.15, f"大词表 Embedding 缩放后标准差应接近 1.0，实际为 {std:.3f}"
    assert torch.all(model_pre.src_embedding.embedding.weight[0] == 0.0), "<pad> 行初始权重应全为 0"

    model_post = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        norm_first=False,
    )
    out = model_post(torch.randint(4, 100, (2, 6)), torch.randint(4, 100, (2, 5)))
    assert out.shape == (2, 5, 100)


def test_transformer_three_way_shared_vocab_weight_tying() -> None:
    """验证论文 §3.4 三方权重绑定: src_embedding == tgt_embedding == proj 指向同一内存且节省 2*V*d 参数。"""
    vocab, d_model = 400, 64
    model_3way = Transformer(
        src_vocab_size=vocab,
        tgt_vocab_size=vocab,
        d_model=d_model,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        tie_tgt_weights=True,
        tie_all_embeddings=True,
    )
    model_untied = Transformer(
        src_vocab_size=vocab,
        tgt_vocab_size=vocab,
        d_model=d_model,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        tie_tgt_weights=False,
        tie_all_embeddings=False,
    )
    assert model_3way.src_embedding.embedding.weight.data_ptr() == model_3way.tgt_embedding.embedding.weight.data_ptr()
    assert model_3way.src_embedding.embedding.weight.data_ptr() == model_3way.proj.weight.data_ptr()

    n_3way = sum(p.numel() for p in model_3way.parameters())
    n_untied = sum(p.numel() for p in model_untied.parameters())
    assert n_untied - n_3way == 2 * vocab * d_model


def test_flash_causal_and_unpad_loss_exact_gradient_equivalence() -> None:
    """验证原生因果模式 (tgt_is_causal=True) + 有效词紧凑投影 (valid_tgt_mask) 与全矩阵原版前反向梯度严格等价。"""
    from src.data import collate_parallel_batch
    import copy

    torch.manual_seed(77)
    m_orig = Transformer(
        src_vocab_size=120,
        tgt_vocab_size=120,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.0,
        pad_id=0,
        tie_all_embeddings=True,
        use_flash_attn=False,  # 先测手写注意力分支下的因果等价
    )
    m_fast = Transformer(
        src_vocab_size=120,
        tgt_vocab_size=120,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.0,
        pad_id=0,
        tie_all_embeddings=True,
        use_flash_attn=True,   # 同时开启 FlashAttention + tgt_is_causal + valid_tgt_mask
    )
    m_fast.load_state_dict(copy.deepcopy(m_orig.state_dict()))

    batch = collate_parallel_batch(
        [([10, 11, 12, 13], [21, 22]), ([14, 15], [23, 24, 25, 26, 27])],
        pad_id=0,
        bos_id=2,
        eos_id=3,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1, reduction="sum")

    # 1. 原版路径：显式传入 (B, 1, T, T) tgt_mask + 全量 (B, T, V) 投影
    logits_orig = m_orig(batch["src"], batch["tgt_in"], batch["src_mask"], batch["tgt_mask"])
    loss_orig = criterion(logits_orig.reshape(-1, 120), batch["tgt_out"].reshape(-1))
    loss_orig.backward()

    # 2. 极速路径：tgt_mask=None, tgt_is_causal=True + 仅对非 <pad> 词做紧凑投影
    valid_mask = batch["tgt_out"] != 0
    logits_fast = m_fast(
        batch["src"],
        batch["tgt_in"],
        batch["src_mask"],
        tgt_mask=None,
        tgt_is_causal=True,
        valid_tgt_mask=valid_mask,
    )
    assert logits_fast.shape == (int(batch["ntokens"].item()), 120)
    loss_fast = criterion(logits_fast, batch["tgt_out"][valid_mask])
    loss_fast.backward()

    assert torch.allclose(loss_orig, loss_fast, atol=1e-4)
    for (name, p_o), (_, p_f) in zip(m_orig.named_parameters(), m_fast.named_parameters()):
        assert p_o.grad is not None and p_f.grad is not None
        assert torch.allclose(p_o.grad, p_f.grad, atol=1e-4), f"参数 {name} 在加速路径与原版路径下梯度不一致！"

