from __future__ import annotations

from pathlib import Path
import os
from typing import Iterable
import sentencepiece as spm


class SentencePieceTokenizer:
    """基于 SentencePiece 的单语分词器封装（分别用于中文源端与英文目标端）。"""

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到分词器模型文件: {self.model_path}")
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_path))

    @classmethod
    def train(
        cls,
        input_txt_path: str | Path,
        model_prefix: str | Path,
        vocab_size: int = 16000,
        character_coverage: float = 0.9995,
        model_type: str = "bpe",
    ) -> SentencePieceTokenizer:
        """从纯文本语料训练一个 SentencePiece 模型并返回加载好的实例。"""
        model_prefix = Path(model_prefix)
        model_prefix.parent.mkdir(parents=True, exist_ok=True)

        # 词表 >= 512 时开启 byte_fallback，遇到罕见生僻字/符号自动退化为 UTF-8 字节而不丢失为 <unk>
        use_byte_fallback = vocab_size >= 512

        spm.SentencePieceTrainer.train(
            input=str(input_txt_path),
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            pad_id=cls.PAD_ID,
            unk_id=cls.UNK_ID,
            bos_id=cls.BOS_ID,
            eos_id=cls.EOS_ID,
            byte_fallback=use_byte_fallback,
            hard_vocab_limit=False,
            normalization_rule_name="nmt_nfkc",
            input_sentence_size=1_200_000,
            shuffle_input_sentence=True,
            num_threads=max(1, os.cpu_count() or 4),
        )
        return cls(f"{model_prefix}.model")

    def encode(self, text: str) -> list[int]:
        """将字符串编码为纯正文子词 ID 列表（不自动加 bos/eos，交由 collate 统一控制）。"""
        return list(self.sp.encode(text.strip(), out_type=int))

    def decode(self, ids: Iterable[int]) -> str:
        """将子词 ID 序列还原为可读自然文本（自动过滤 pad/bos，遇到首个 eos 立即截断）。"""
        clean_ids: list[int] = []
        for tok_id in ids:
            val = int(tok_id)
            if val == self.EOS_ID:
                break
            if val in (self.PAD_ID, self.BOS_ID):
                continue
            clean_ids.append(val)
        return self.sp.decode(clean_ids)

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size())
