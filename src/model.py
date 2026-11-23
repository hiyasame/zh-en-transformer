from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Level 1: Attention & Masking (Paper §3.2)
# ============================================================================


def make_src_mask(src: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """构造源端 Padding Mask。

    约定（全项目统一）：
        Mask 中 True (或 1) 表示【有效位置，允许被 Attention 看到】；
        False (或 0) 表示【被屏蔽位置（如 <pad> 或未来时刻），Attention 分数需置为极小负数】。

    Args:
        src: 源语言 Token ID 张量, shape = (B, S)
        pad_id: <pad> 的 token ID

    Returns:
        布尔张量, shape = (B, 1, 1, S), 位于 src.device 上
    """
    return (src != pad_id).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """构造目标端联合 Mask（下三角因果 Causal Mask & 目标端 Padding Mask）。

    位置 (b, 0, i, j) 为 True 当且仅当：
        1. j <= i （第 i 个位置不能偷看 j > i 的未来位置）
        2. tgt[b, j] != pad_id （Key 位置不是 <pad>）

    Args:
        tgt: 目标语言输入 Token ID 张量, shape = (B, T)
        pad_id: <pad> 的 token ID

    Returns:
        布尔张量, shape = (B, 1, T, T), 位于 tgt.device 上
    """
    T = tgt.size(1)
    # (B, 1, 1, T)
    pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)
    # torch.tril 保留矩阵下三角，右上角未来位置 j > i 全部置为 false
    # (T, T)
    causal_mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=tgt.device))
    return pad_mask & causal_mask


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """论文公式 (1): Scaled Dot-Product Attention。

        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    注意：
        - 若传入 mask，将 mask == 0 (即 False) 的位置在 softmax 之前填充为 -1e9
          （或 torch.finfo(scores.dtype).min，避免 fp16/mps 下 -inf 产生 NaN）。
        - 若传入 dropout 模块，将其作用于 softmax 之后的注意力权重矩阵上。

    Args:
        q: Query 张量, shape = (..., L_q, d_k)
        k: Key 张量,   shape = (..., L_k, d_k)
        v: Value 张量, shape = (..., L_k, d_v)
        mask: 可广播至 (..., L_q, L_k) 的掩码张量 (True=保留, False=屏蔽)
        dropout: 可选的 nn.Dropout 实例

    Returns:
        (output, attn_weights):
            output: shape = (..., L_q, d_v)
            attn_weights: shape = (..., L_q, L_k)
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
    attn_weights = scores.softmax(dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    return attn_weights @ v, attn_weights

class MultiHeadAttention(nn.Module):
    """论文 §3.2.2: Multi-Head Attention。

        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_p = dropout
        self.use_flash_attn = use_flash_attn
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """前向计算（需同时兼容 L_q == L_k 的自注意力与 L_q != L_k 的交叉注意力）。

        Args:
            q: shape = (B, L_q, d_model)
            k: shape = (B, L_k, d_model)
            v: shape = (B, L_k, d_model)
            mask: 可广播至 (B, num_heads, L_q, L_k) 的布尔掩码
            is_causal: 当为 True 且 mask 为 None 时，直接启用原生 FlashAttention-2 下三角因果内核

        Returns:
            输出张量, shape = (B, L_q, d_model)
        """
        B, L_q, d_model = q.shape
        B, L_k, d_model = k.shape
        B, L_v, d_model = v.shape
        # (B, num_heads, L_q, d_k)
        Q = self.w_q(q).view(B, L_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(B, L_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(B, L_v, self.num_heads, self.d_k).transpose(1, 2)
        # (B, num_heads, L_q, d_k)
        if self.use_flash_attn and hasattr(F, "scaled_dot_product_attention"):
            if is_causal and mask is None:
                res = F.scaled_dot_product_attention(
                    Q,
                    K,
                    V,
                    attn_mask=None,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=True,
                )
            else:
                attn_mask = mask.bool() if (mask is not None and mask.dtype != torch.bool) else mask
                res = F.scaled_dot_product_attention(
                    Q,
                    K,
                    V,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout_p if self.training else 0.0,
                )
        else:
            eff_mask = mask
            if is_causal and eff_mask is None:
                eff_mask = torch.tril(torch.ones((L_q, L_k), dtype=torch.bool, device=q.device))
            res, _ = scaled_dot_product_attention(Q, K, V, mask=eff_mask, dropout=self.dropout)
        res = res.transpose(1, 2).contiguous().view(B, L_q, d_model)
        return self.w_o(res)




# ============================================================================
# Level 2: Embeddings, Sinusoidal PE & Feed-Forward (Paper §3.3 - §3.5)
# ============================================================================


class TokenEmbedding(nn.Module):
    """论文 §3.4: 带 sqrt(d_model) 缩放的词嵌入层。"""

    def __init__(self, vocab_size: int, d_model: int, pad_id: int = 0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将输入 ID 映射为词向量并乘以 sqrt(d_model)。

        Args:
            x: shape = (B, L)

        Returns:
            shape = (B, L, d_model)
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """论文 §3.5: 正余弦固定位置编码 (Sinusoidal Positional Encoding)。

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    提示：
        - 使用 self.register_buffer('pe', pe) 注册位置矩阵，使其不参与梯度更新，
          但能随 model.to(device) 自动迁移设备并保存在 state_dict 中。
        - 相加后对结果应用一次 Dropout (论文 §5.4)。
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        # (max_len, 1)
        self.pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        self.div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(self.pos * self.div_term)
        pe[:, 1::2] = torch.cos(self.pos * self.div_term)
        pe = pe.view(1, max_len, d_model)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """为输入词向量加上前 L 个位置的位置编码并执行 Dropout。

        Args:
            x: shape = (B, L, d_model)

        Returns:
            shape = (B, L, d_model)
        """
        L = x.size(1)
        return self.dropout(x + self.pe[:, :L, :])



class PositionwiseFeedForward(nn.Module):
    """论文 §3.3 公式 (2): Position-wise Feed-Forward Networks。

        FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_ = nn.Linear(in_features=d_model, out_features=d_ff)
        self.out_ = nn.Linear(in_features=d_ff, out_features=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x shape = (B, L, d_model) -> Returns: shape = (B, L, d_model)"""
        return self.out_(self.dropout(torch.relu(self.in_(x))))


# ============================================================================
# Level 3: Encoder, Decoder & Full Transformer (Paper §3.1, §3.4)
# ============================================================================


class EncoderLayer(nn.Module):
    """单层 Encoder（默认采用梯度更稳的 Pre-LN 结构，亦可通过 norm_first=False 切回原论文 Post-LN）。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = True,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.self_att = MultiHeadAttention(d_model, num_heads, dropout=dropout, use_flash_attn=use_flash_attn)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Args: x (B, S, d_model), src_mask (B, 1, 1, S) -> Returns: (B, S, d_model)"""
        if self.norm_first:
            nx = self.norm1(x)
            x = x + self.dropout(self.self_att(nx, nx, nx, mask=src_mask))
            nx2 = self.norm2(x)
            x = x + self.dropout(self.ffn(nx2))
            return x
        x = self.norm1(x + self.dropout(self.self_att(x, x, x, mask=src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    """单层 Decoder（含 Masked Self-Attention、Cross-Attention、FFN 三个子层，支持 Pre-LN / Post-LN）。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = True,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, use_flash_attn=use_flash_attn)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, use_flash_attn=use_flash_attn)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        tgt_is_causal: bool = False,
    ) -> torch.Tensor:
        """注意区分：自注意力用 tgt_mask（或原生 FlashAttention 因果模式 tgt_is_causal=True），交叉注意力用 src_mask！"""
        self_mask = None if tgt_is_causal else tgt_mask
        if self.norm_first:
            nx = self.norm1(x)
            x = x + self.dropout(self.self_attn(nx, nx, nx, mask=self_mask, is_causal=tgt_is_causal))
            nx2 = self.norm2(x)
            x = x + self.dropout(self.cross_attn(nx2, memory, memory, mask=src_mask))
            nx3 = self.norm3(x)
            x = x + self.dropout(self.ffn(nx3))
            return x
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask=self_mask, is_causal=tgt_is_causal)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, mask=src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x

class Transformer(nn.Module):
    """完整 Encoder-Decoder Transformer 模型（适配 Zh -> En 独立词表 + Decoder 双向权重绑定）。"""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        pad_id: int = 0,
        tie_tgt_weights: bool = True,
        tie_all_embeddings: bool = False,
        norm_first: bool = True,
        use_flash_attn: bool = True,
    ) -> None:
        """初始化完整模型。

        关键要求：
            1. 默认源端 (中文) 与目标端 (英文) 使用各自独立的 TokenEmbedding；
               当 tie_all_embeddings=True (原论文 §3.4 共享 BPE 词表模式) 时，源端词嵌入、目标端词嵌入与
               输出线性投影层三方共享同一个 nn.Parameter 权重矩阵！
            2. 当 tie_tgt_weights=True 时，目标端 Embedding 与输出投影层共享权重。
            3. 线性层采用 Xavier Uniform 初始化；词嵌入矩阵按 N(0, d_model^-0.5) 初始化，
               确保乘以 sqrt(d_model) 后词向量具备单位方差 (std=1.0)，不被位置编码淹没。
        """
        super().__init__()
        if tie_all_embeddings and src_vocab_size != tgt_vocab_size:
            raise ValueError(f"开启 tie_all_embeddings 时源词表 ({src_vocab_size}) 与目标词表 ({tgt_vocab_size}) 大小必须一致")
        self.pad_id = pad_id
        self.norm_first = norm_first
        self.use_flash_attn = use_flash_attn
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model, pad_id)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model, pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.encoder_modules = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout, norm_first=norm_first, use_flash_attn=use_flash_attn)
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_modules = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, d_ff, dropout, norm_first=norm_first, use_flash_attn=use_flash_attn)
                for _ in range(num_decoder_layers)
            ]
        )
        self.enc_norm: nn.Module = nn.LayerNorm(d_model) if norm_first else nn.Identity()
        self.dec_norm: nn.Module = nn.LayerNorm(d_model) if norm_first else nn.Identity()
        self.proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # 1. 先对所有二维及以上的线性投影矩阵做 Xavier 初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # 2. 词嵌入矩阵与输出投影按 N(0, d_model^-0.5) 初始化（乘完 sqrt(d_model) 后方差恰好为 1.0）
        emb_std = d_model**-0.5
        nn.init.normal_(self.src_embedding.embedding.weight, mean=0.0, std=emb_std)
        nn.init.normal_(self.tgt_embedding.embedding.weight, mean=0.0, std=emb_std)
        if not (tie_tgt_weights or tie_all_embeddings):
            nn.init.normal_(self.proj.weight, mean=0.0, std=emb_std)
        with torch.no_grad():
            self.src_embedding.embedding.weight[pad_id].zero_()
            self.tgt_embedding.embedding.weight[pad_id].zero_()

        # 3. 权重绑定（支持原论文 §3.4 三方全绑定 或 独立词表下的 Decoder 双向绑定）
        if tie_all_embeddings:
            self.tgt_embedding.embedding.weight = self.src_embedding.embedding.weight
            self.proj.weight = self.src_embedding.embedding.weight
        elif tie_tgt_weights:
            self.proj.weight = self.tgt_embedding.embedding.weight

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """编码源语言序列: src (B, S) -> memory (B, S, d_model)"""
        src_embedded = self.src_embedding(src)
        x = self.pos_encoding(src_embedded)
        for module in self.encoder_modules:
            x = module(x, src_mask)
        return self.enc_norm(x)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        tgt_is_causal: bool = False,
    ) -> torch.Tensor:
        """解码目标序列隐状态: tgt (B, T), memory (B, S, d_model) -> dec_out (B, T, d_model)"""
        tgt_embedded = self.tgt_embedding(tgt)
        x = self.pos_encoding(tgt_embedded)
        for module in self.decoder_modules:
            x = module(x, memory, src_mask, tgt_mask, tgt_is_causal=tgt_is_causal)
        return self.dec_norm(x)

    def project(self, dec_out: torch.Tensor) -> torch.Tensor:
        """映射到目标词表未归一化 Logits（不要在这里做 Softmax）: (..., d_model) -> (..., tgt_vocab_size)"""
        return self.proj(dec_out)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        tgt_is_causal: bool = False,
        valid_tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """完整前向传播。
        - 默认返回目标词表全量 Logits: shape = (B, T, tgt_vocab_size)；
        - 当传入 valid_tgt_mask (shape = (B, T) 的非 <pad> 布尔掩码) 时，仅对真实有效词执行大词表输出投影，
          返回紧凑形状 (N_valid, tgt_vocab_size)，跳过全部 <pad> 冗余矩阵乘且梯度 100% 严格等价。
        """
        if src_mask is None:
            src_mask = make_src_mask(src, pad_id=self.pad_id)
        if tgt_mask is None and not tgt_is_causal:
            tgt_mask = make_tgt_mask(tgt, pad_id=self.pad_id)
        dec_out = self.decode(
            tgt,
            self.encode(src, src_mask),
            src_mask,
            tgt_mask,
            tgt_is_causal=tgt_is_causal,
        )
        if valid_tgt_mask is not None:
            return self.project(dec_out[valid_tgt_mask])
        return self.project(dec_out)
