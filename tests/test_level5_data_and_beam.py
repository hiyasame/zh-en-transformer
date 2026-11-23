from __future__ import annotations

import math
import torch
import torch.nn as nn

from src.data import clean_zh_en_pair, collate_parallel_batch
from src.decode import greedy_decode, beam_search_decode, fast_beam_search_decode
from src.model import Transformer


def test_clean_zh_en_pair_filters_noise() -> None:
    assert clean_zh_en_pair("今天天气非常好，我们去公园散步吧。", "The weather is great today, let's take a walk in the park.")
    assert not clean_zh_en_pair("", "Hello world")
    assert not clean_zh_en_pair("   ", "   ")
    # 双边都是未翻译英文脏样本
    assert not clean_zh_en_pair("This sentence was never translated into Chinese.", "This sentence was never translated into Chinese.")
    # 极端长度比例失衡
    assert not clean_zh_en_pair(
        "好",
        "This is an extremely long English paragraph with dozens of words that obviously does not align with one character.",
    )


def test_collate_parallel_batch_teacher_forcing_shift() -> None:
    batch = [
        ([10, 11, 12], [20, 21]),        # 源 3 词, 目标 2 词
        ([13], [22, 23, 24, 25]),        # 源 1 词, 目标 4 词
    ]
    out = collate_parallel_batch(batch, pad_id=0, bos_id=2, eos_id=3)

    expected_src = torch.tensor([[10, 11, 12, 3], [13, 3, 0, 0]])
    expected_tgt_in = torch.tensor([[2, 20, 21, 0, 0], [2, 22, 23, 24, 25]])
    expected_tgt_out = torch.tensor([[20, 21, 3, 0, 0], [22, 23, 24, 25, 3]])

    assert torch.equal(out["src"], expected_src), "src 应在末尾拼上 eos_id 并用 pad_id 右侧补齐"
    assert torch.equal(out["tgt_in"], expected_tgt_in), "tgt_in 应为 [bos_id] + tgt_ids 并补齐"
    assert torch.equal(out["tgt_out"], expected_tgt_out), "tgt_out 应为 tgt_ids + [eos_id] 并补齐"
    assert out["src_mask"].shape == (2, 1, 1, 4)
    assert out["tgt_mask"].shape == (2, 1, 5, 5)
    assert int(out["ntokens"].item()) == 8, f"有效非 pad 目标 token 数应为 3 + 5 = 8，实际为 {out['ntokens']}"


class _TrapToyModel(nn.Module):
    """构造一个贪心陷阱概率树，检验 Beam Search 是否真正维护了多条束路径：
    - 第 1 步 (<bos>=2):
        选 4 的概率 = 0.55 (贪心局部最优！)
        选 5 的概率 = 0.45
    - 第 2 步:
        若前缀是 [2, 4]: 所有后继概率都很分散，最高仅 0.20 -> 最优联合概率 = 0.55 * 0.20 = 0.11
        若前缀是 [2, 5]: 直接生成 <eos>=3 的概率高达 0.95 -> 联合概率 = 0.45 * 0.95 = 0.4275 (全局最优！)
    """

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        return torch.zeros(src.size(0), src.size(1), 8, device=src.device)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 把当前历史编码进输出特征，方便 project 精确返回对应条件概率
        B, T = tgt.shape
        out = torch.zeros(B, T, 8, device=tgt.device)
        out[:, -1, 0] = float(T)
        if T >= 2:
            out[:, -1, 1] = tgt[:, 1].float()
        return out

    def project(self, dec_out: torch.Tensor) -> torch.Tensor:
        B, T, _ = dec_out.shape
        # vocab_size = 8: 0=pad, 1=unk, 2=bos, 3=eos, 4,5,6,7
        logits = torch.full((B, T, 8), -1e9, device=dec_out.device)
        for b in range(B):
            step = int(round(dec_out[b, -1, 0].item()))
            if step == 1:
                # P(4)=0.55, P(5)=0.45
                logits[b, -1, 4] = math.log(0.55)
                logits[b, -1, 5] = math.log(0.45)
            else:
                first_tok = int(round(dec_out[b, -1, 1].item()))
                if first_tok == 4:
                    # 5 个词平分概率 0.20
                    for tok in (3, 4, 5, 6, 7):
                        logits[b, -1, tok] = math.log(0.20)
                elif first_tok == 5:
                    logits[b, -1, 3] = math.log(0.95)  # eos
                    logits[b, -1, 6] = math.log(0.05)
                else:
                    logits[b, -1, 3] = 0.0
        return logits

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.project(self.decode(tgt, memory, src_mask, tgt_mask))


def test_beam_search_beats_greedy_on_trap_tree() -> None:
    toy = _TrapToyModel()
    src = torch.tensor([[4, 5, 3]])
    src_mask = torch.ones(1, 1, 1, 3, dtype=torch.bool)

    greedy_seq = greedy_decode(toy, src, src_mask, max_len=5, bos_id=2, eos_id=3, pad_id=0)
    assert greedy_seq[0, 1].item() == 4, "贪心解码第 1 步必然掉入局部最优 token 4"

    beam_seq = beam_search_decode(
        toy,
        src,
        src_mask,
        max_len=5,
        beam_size=2,
        alpha=0.6,
        bos_id=2,
        eos_id=3,
        pad_id=0,
    )
    assert beam_seq == [2, 5, 3], (
        f"Beam Search (beam_size=2) 应避开贪心陷阱找到全局最优路径 [2, 5, 3]，实际返回 {beam_seq}"
    )
    fast_seq = fast_beam_search_decode(
        toy, src, src_mask, max_len=5, beam_size=2, alpha=0.6, bos_id=2, eos_id=3, pad_id=0
    )
    assert fast_seq == beam_seq


def test_fast_beam_search_matches_handwritten_beam_search_on_transformer() -> None:
    """验证批量化 fast_beam_search_decode 与手写原版 beam_search_decode 在真实 Transformer 上输出序列完全一致。"""
    torch.manual_seed(2026)
    model = Transformer(
        src_vocab_size=60,
        tgt_vocab_size=60,
        d_model=32,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        dropout=0.0,
        pad_id=0,
    ).eval()

    for length in [4, 9]:
        src = torch.randint(4, 60, (1, length))
        orig_ids = beam_search_decode(model, src, None, max_len=12, beam_size=4, alpha=0.6)
        fast_ids = fast_beam_search_decode(model, src, None, max_len=12, beam_size=4, alpha=0.6)
        assert fast_ids == orig_ids, f"加速版与手写版 Beam Search 结果不一致: {fast_ids} vs {orig_ids}"

