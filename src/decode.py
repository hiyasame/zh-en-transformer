from __future__ import annotations

import torch

from .model import make_src_mask, make_tgt_mask


# ============================================================================
# Level 4 & Level 5: Autoregressive Decoding (Greedy & Beam Search, Paper §6.1)
# ============================================================================


@torch.no_grad()
def greedy_decode(
    model: torch.nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor | None,
    max_len: int,
    bos_id: int = 2,
    eos_id: int = 3,
    pad_id: int = 0,
) -> torch.Tensor:
    """自回归贪心解码（支持批量输入 B >= 1）。

    流程提示：
        1. 切记确保在推理态下执行（或在外部 model.eval()），先调用一次 model.encode(src, src_mask) 得到 memory。
        2. 初始化已生成序列 ys = full((B, 1), bos_id)。
        3. 循环最多 max_len - 1 步（或直到 batch 内所有样本都已生成过 eos_id 提前终止）：
           每次用 make_tgt_mask(ys, pad_id) 构造掩码，调用 model.decode -> model.project 取最后一步 logits[:, -1, :]，
           选 argmax 拼接到 ys 末尾。

    Args:
        model: 具备 encode / decode / project 方法的 Transformer 实例
        src: 源端 Token IDs, shape = (B, S)
        src_mask: 源端掩码, shape = (B, 1, 1, S)（若为 None 可用 make_src_mask 构造）
        max_len: 生成序列的最大总长度（含开头的 bos_id）
        bos_id: 句首符号 ID
        eos_id: 句尾符号 ID
        pad_id: 填充符号 ID

    Returns:
        生成的目标端 Token ID 张量（含开头的 bos_id）, shape = (B, <= max_len)
    """
    if src_mask is None:
        src_mask = make_src_mask(src, pad_id)
    memory = model.encode(src, src_mask)
    B, S = src.shape
    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
    finished = torch.zeros(B, dtype=torch.bool, device=src.device)
    for i in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_id)
        dec_out = model.decode(ys, memory, src_mask, tgt_mask)
        logits = model.project(dec_out[:, -1:, :])[:, 0, :]

        # (B, 1)
        next_tok = logits.argmax(dim=-1, keepdim=True)
        # (B, t + 1)
        ys = torch.cat([ys, next_tok], dim=1)

        finished  |= (next_tok.squeeze(1) == eos_id)
        if finished.all():
            break

    return ys

@torch.no_grad()
def beam_search_decode(
    model: torch.nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor | None,
    max_len: int,
    beam_size: int = 4,
    alpha: float = 0.6,
    bos_id: int = 2,
    eos_id: int = 3,
    pad_id: int = 0,
) -> list[int]:
    """单句 Beam Search 束搜索解码（论文 §6.1: beam_size=4, length_penalty alpha=0.6）。

    长度归一化打分公式（Google NMT, Wu et al. 2016）：
        lp(Y)    = ((5 + len(Y)) / (5 + 1)) ** alpha
        score(Y) = sum_log_prob(Y) / lp(Y)
        （其中 len(Y) 为不含开头的 <bos>、含 <eos> 的生成步数；alpha=0.0 时退化为纯累积对数概率比较）

    Args:
        model: 具备 encode / decode / project 方法的 Transformer 实例
        src: 单条源句 Token IDs, shape = (1, S)
        src_mask: 源端掩码, shape = (1, 1, 1, S)
        max_len: 生成序列最大总长度（含 bos_id）
        beam_size: 束宽 k
        alpha: 长度惩罚系数（论文取 0.6）
        bos_id: 句首 ID
        eos_id: 句尾 ID
        pad_id: 填充 ID

    Returns:
        得分最高的一条候选序列 Token ID 列表（含开头的 bos_id，若生成了 eos_id 则以 eos_id 结尾）
    """
    if src_mask is None:
        src_mask = make_src_mask(src, pad_id)
    memory = model.encode(src, src_mask)

    def calc_score(log_p: float, seq: list[int]) -> float:
        # 扣除开头的 <bos>，按生成步数做 Google NMT 长度惩罚归一化
        length = max(1, len(seq) - 1)
        lp = ((5.0 + length) / 6.0) ** alpha
        return log_p / lp

    # (累积对数概率, token_id 列表)
    beams: list[tuple[float, list[int]]] = [(0.0, [bos_id])]
    for _ in range(max_len - 1):
        candidates: list[tuple[float, list[int]]] = []
        for log_p, seq in beams:
            # 已经生成过 <eos> 的分支不再继续扩展，直接保留参与全局排名
            if seq[-1] == eos_id:
                candidates.append((log_p, seq))
                continue
            ys = torch.tensor([seq], dtype=torch.long, device=src.device)
            tgt_mask = make_tgt_mask(ys, pad_id)
            dec_out = model.decode(ys, memory, src_mask, tgt_mask)
            logits = model.project(dec_out[:, -1:, :])[0, 0, :]  # (V,)
            log_probs = torch.log_softmax(logits, dim=-1)
            k = min(beam_size, log_probs.size(-1))
            top_log_p, top_ids = torch.topk(log_probs, k)
            for tok_log_p, tok_id in zip(top_log_p.tolist(), top_ids.tolist()):
                candidates.append((log_p + tok_log_p, seq + [tok_id]))
        # 按长度归一化后的得分从高到低保留前 beam_size 条路径
        candidates.sort(key=lambda item: calc_score(item[0], item[1]), reverse=True)
        beams = candidates[:beam_size]
        # 所有保留的束都已遇到 <eos> 时提前结束
        if all(seq[-1] == eos_id for _, seq in beams):
            break
    return beams[0][1]


@torch.no_grad()
def fast_beam_search_decode(
    model: torch.nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor | None,
    max_len: int,
    beam_size: int = 4,
    alpha: float = 0.6,
    bos_id: int = 2,
    eos_id: int = 3,
    pad_id: int = 0,
) -> list[int]:
    """批量化束搜索加速版（与手写版 beam_search_decode 数学输出 100% 严格一致，将 k 条活跃束合并为单次 GPU 矩阵前向）。"""
    if src_mask is None:
        src_mask = make_src_mask(src, pad_id)
    memory = model.encode(src, src_mask)

    def calc_score(log_p: float, seq: list[int]) -> float:
        length = max(1, len(seq) - 1)
        lp = ((5.0 + length) / 6.0) ** alpha
        return log_p / lp

    beams: list[tuple[float, list[int]]] = [(0.0, [bos_id])]
    for _ in range(max_len - 1):
        candidates: list[tuple[float, list[int]]] = []
        active_beams: list[tuple[float, list[int]]] = []
        for log_p, seq in beams:
            if seq[-1] == eos_id:
                candidates.append((log_p, seq))
            else:
                active_beams.append((log_p, seq))

        if not active_beams:
            break

        # 所有未结束的活跃束在当前步的长度必然完全相等，直接打包为 (M, cur_len) 一次性解码
        m = len(active_beams)
        ys = torch.tensor([seq for _, seq in active_beams], dtype=torch.long, device=src.device)
        mem_b = memory.expand(m, -1, -1)
        mask_b = src_mask.expand(m, -1, -1, -1)
        tgt_mask = make_tgt_mask(ys, pad_id)

        dec_out = model.decode(ys, mem_b, mask_b, tgt_mask)
        logits = model.project(dec_out[:, -1:, :])[:, 0, :]  # (M, V)
        log_probs = torch.log_softmax(logits, dim=-1)
        k = min(beam_size, log_probs.size(-1))
        top_log_p, top_ids = torch.topk(log_probs, k, dim=-1)

        top_log_p_list = top_log_p.cpu().tolist()
        top_ids_list = top_ids.cpu().tolist()
        for idx, (base_log_p, seq) in enumerate(active_beams):
            for tok_log_p, tok_id in zip(top_log_p_list[idx], top_ids_list[idx]):
                candidates.append((base_log_p + tok_log_p, seq + [tok_id]))

        candidates.sort(key=lambda item: calc_score(item[0], item[1]), reverse=True)
        beams = candidates[:beam_size]
        if all(seq[-1] == eos_id for _, seq in beams):
            break

    return beams[0][1]

