from __future__ import annotations

from typing import Callable, Iterator
import random
import torch
from torch.utils.data import Dataset, Sampler

from .model import make_src_mask, make_tgt_mask


# ============================================================================
# Level 5: Data Cleaning, Teacher-Forcing Collation & Token Batching (§5.1)
# ============================================================================


def clean_zh_en_pair(
    zh: str,
    en: str,
    min_chars: int = 2,
    max_chars: int = 300,
    min_zh_ratio: float = 0.3,
) -> bool:
    """判断一条中英平行句对是否干净可用。

    过滤建议（全部满足才返回 True）：
        1. strip() 后两边均非空，且字符数均在 [min_chars, max_chars] 范围内；
        2. 中文端包含足够比例的中文字符（例如 Unicode 范围 '\\u4e00' <= ch <= '\\u9fff'
           在非空白字符中的占比 >= min_zh_ratio），剔除双边都是纯英文的脏样本；
        3. 中英文长度没有极端失衡（例如中文字符数与英文单词数的比值在 [0.2, 5.0] 之间）。
    """
    zh_clean = zh.strip()
    en_clean = en.strip()
    # 1. 检查长度
    if len(zh_clean) < min_chars or len(zh_clean) > max_chars:
        return False
    if len(en_clean) < min_chars or len(en_clean) > max_chars:
        return False
    # 2. 检查汉字比例
    zh_len = len(zh_clean)
    kanji_num = 0
    for ch in zh_clean:
        if '\u4e00' <= ch <= '\u9fff':
            kanji_num += 1
    if kanji_num / zh_len < min_zh_ratio:
        return False
    # 3. 保证中文字符数和英文单词数的比例没有极端失衡
    en_words_len = len(en.split())
    ratio = zh_len / en_words_len
    if ratio < 0.2 or ratio > 5.0:
        return False
    return True


def collate_parallel_batch(
    batch: list[tuple[list[int], list[int]]],
    pad_id: int = 0,
    bos_id: int = 2,
    eos_id: int = 3,
) -> dict[str, torch.Tensor]:
    """将一个 Batch 的 (zh_token_ids, en_token_ids) 组装成 Teacher Forcing 训练张量。

    输入约定：
        batch 中每项为 (src_ids, tgt_ids)，其中 src_ids / tgt_ids 为【纯正文子词 ID 列表】（尚未加 bos/eos）：
        - 源端序列组装为: src_ids + [eos_id]，右侧用 pad_id 补齐到当前 batch 最大源长度 S；
        - 目标完整序列为: [bos_id] + tgt_ids + [eos_id]：
            * Decoder 输入 tgt_in  取前 T 个 token（即 [bos_id] + tgt_ids，不含末尾 eos_id）
            * Loss 监督标签 tgt_out 取后 T 个 token（即 tgt_ids + [eos_id]，不含开头 bos_id）
          两者右侧均用 pad_id 补齐到同一长度 T。

    Returns:
        包含以下键的字典：
            "src":      LongTensor, shape = (B, S)
            "tgt_in":   LongTensor, shape = (B, T)
            "tgt_out":  LongTensor, shape = (B, T)
            "src_mask": BoolTensor, shape = (B, 1, 1, S)
            "tgt_mask": BoolTensor, shape = (B, 1, T, T)
            "ntokens":  标量 LongTensor, 当前 batch 中 tgt_out != pad_id 的有效目标 token 总数
    """
    src_seqs = []
    tgt_in_seqs = []
    tgt_out_seqs = []
    for src_ids, tgt_ids in batch:
        src_seqs.append(src_ids + [eos_id])
        tgt_in_seqs.append([bos_id] + tgt_ids)
        tgt_out_seqs.append(tgt_ids + [eos_id])
    # 1. 求出当前 batch 内源端最大长度 S、目标端最大长度 T
    max_src_len = max(len(s) for s in src_seqs)
    max_tgt_len = max(len(t) for t in tgt_in_seqs)
    # 2. 右侧补齐 pad_id 并转为 LongTensor
    src = torch.tensor(
        [s + [pad_id] * (max_src_len - len(s)) for s in src_seqs],
        dtype=torch.long,
    )
    tgt_in = torch.tensor(
        [t + [pad_id] * (max_tgt_len - len(t)) for t in tgt_in_seqs],
        dtype=torch.long,
    )
    tgt_out = torch.tensor(
        [t + [pad_id] * (max_tgt_len - len(t)) for t in tgt_out_seqs],
        dtype=torch.long,
    )
    # 3. 构造掩码并统计有效非 pad 目标词数
    src_mask = make_src_mask(src, pad_id=pad_id)
    tgt_mask = make_tgt_mask(tgt_in, pad_id=pad_id)
    ntokens = (tgt_out != pad_id).sum()
    return {
        "src": src,
        "tgt_in": tgt_in,
        "tgt_out": tgt_out,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
        "ntokens": ntokens,
    }



class ParallelDataset(Dataset[tuple[list[int], list[int]]]):
    """封装预编码双语数据集（兼容普通列表与 450 万句级零内存膨胀的 CSR 扁平 int32 张量格式）。"""

    def __init__(self, pairs: list[tuple[list[int], list[int]]] | dict[str, torch.Tensor]) -> None:
        if isinstance(pairs, dict):
            self._is_csr = True
            self.src_flat = pairs["src_flat"]
            self.src_offsets = pairs["src_offsets"]
            self.tgt_flat = pairs["tgt_flat"]
            self.tgt_offsets = pairs["tgt_offsets"]
            self.pairs = None
            src_lens = (self.src_offsets[1:] - self.src_offsets[:-1] + 1).tolist()
            tgt_lens = (self.tgt_offsets[1:] - self.tgt_offsets[:-1] + 1).tolist()
            self.lengths: list[tuple[int, int]] = list(zip(src_lens, tgt_lens))
        else:
            self._is_csr = False
            self.pairs = pairs
            # +1 对应 src 末尾的 <eos> 以及 tgt_in 开头的 <bos>
            self.lengths = [(len(s) + 1, len(t) + 1) for s, t in pairs]

    def __len__(self) -> int:
        if self._is_csr:
            return int(self.src_offsets.numel() - 1)
        return len(self.pairs)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        if self._is_csr:
            s0 = int(self.src_offsets[idx].item())
            s1 = int(self.src_offsets[idx + 1].item())
            t0 = int(self.tgt_offsets[idx].item())
            t1 = int(self.tgt_offsets[idx + 1].item())
            return self.src_flat[s0:s1].tolist(), self.tgt_flat[t0:t1].tolist()
        return self.pairs[idx]  # type: ignore[index]


class BucketedCollate:
    """顶层可序列化 Collate 类（支持 Linux/macOS 多进程 DataLoader num_workers > 0）：
    调用 collate_parallel_batch 并将序列长度向上对齐到 pad_to_multiple (如 8) 的倍数，
    避免 MPS 与 CUDA torch.compile 频繁重编译算子。
    """

    def __init__(
        self,
        pad_id: int = 0,
        bos_id: int = 2,
        eos_id: int = 3,
        pad_to_multiple: int = 8,
    ) -> None:
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_to_multiple = pad_to_multiple

    def _pad_right_2d(self, x: torch.Tensor) -> torch.Tensor:
        cur_len = x.size(1)
        rem = cur_len % self.pad_to_multiple
        if rem == 0:
            return x
        extra = self.pad_to_multiple - rem
        pad_block = torch.full((x.size(0), extra), self.pad_id, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad_block], dim=1)

    def __call__(self, batch: list[tuple[list[int], list[int]]]) -> dict[str, torch.Tensor]:
        out = collate_parallel_batch(batch, pad_id=self.pad_id, bos_id=self.bos_id, eos_id=self.eos_id)
        if self.pad_to_multiple > 1:
            out["src"] = self._pad_right_2d(out["src"])
            out["tgt_in"] = self._pad_right_2d(out["tgt_in"])
            out["tgt_out"] = self._pad_right_2d(out["tgt_out"])
            out["src_mask"] = make_src_mask(out["src"], pad_id=self.pad_id)
            out["tgt_mask"] = make_tgt_mask(out["tgt_in"], pad_id=self.pad_id)
        return out


def make_bucketed_collate_fn(
    pad_id: int = 0,
    bos_id: int = 2,
    eos_id: int = 3,
    pad_to_multiple: int = 8,
) -> Callable[[list[tuple[list[int], list[int]]]], dict[str, torch.Tensor]]:
    return BucketedCollate(pad_id=pad_id, bos_id=bos_id, eos_id=eos_id, pad_to_multiple=pad_to_multiple)


class TokenBucketBatchSampler(Sampler[list[int]]):
    """论文 §5.1: 按近似长度分桶、并按最大 Token 总数动态组 Batch 的采样器。

    采用缓冲池分桶 (Megabatch Pool Bucketing)：
        1. 每个 epoch 先整体随机打乱索引（若 shuffle=True），切分为若干个大小为 pool_size 的缓冲池；
        2. 在每个缓冲池内部按 max(src_len, tgt_len) 排序，使同 Batch 内句长高度接近，最小化 <pad> 浪费；
        3. 动态累加样本直到 `batch_size * max(max_src, max_tgt) > max_tokens` 时切分新 Batch；
        4. 最后将所有组好的 Batch 顺序再次随机打乱，兼顾随机梯度无偏性与高吞吐。
    """

    def __init__(
        self,
        lengths: list[tuple[int, int]],
        max_tokens: int = 4096,
        shuffle: bool = True,
        seed: int = 42,
        pool_size: int = 2048,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.pool_size = pool_size
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.epoch = 0
        self._batches = self._build_batches(epoch=0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._batches = self._build_batches(epoch=epoch)

    def _build_batches(self, epoch: int) -> list[list[int]]:
        n = len(self.lengths)
        if n == 0:
            return []

        indices = list(range(n))
        rng = random.Random(self.seed + epoch)
        if self.shuffle:
            rng.shuffle(indices)

        batches: list[list[int]] = []
        for start in range(0, n, self.pool_size):
            pool = indices[start : start + self.pool_size]
            pool.sort(key=lambda idx: (max(self.lengths[idx]), self.lengths[idx][0]))

            cur_batch: list[int] = []
            max_src = 0
            max_tgt = 0
            for idx in pool:
                s_len, t_len = self.lengths[idx]
                cand_src = max(max_src, s_len)
                cand_tgt = max(max_tgt, t_len)
                cand_tokens = (len(cur_batch) + 1) * max(cand_src, cand_tgt)

                if cur_batch and cand_tokens > self.max_tokens:
                    batches.append(cur_batch)
                    cur_batch = [idx]
                    max_src, max_tgt = s_len, t_len
                else:
                    cur_batch.append(idx)
                    max_src, max_tgt = cand_src, cand_tgt

            if cur_batch:
                batches.append(cur_batch)

        if self.shuffle:
            rng.shuffle(batches)

        if self.num_replicas > 1 and len(batches) >= self.num_replicas:
            even_total = (len(batches) // self.num_replicas) * self.num_replicas
            batches = batches[self.rank : even_total : self.num_replicas]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        self._batches = self._build_batches(self.epoch)
        self.epoch += 1
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)
