# Zh → En Transformer (From-Scratch Reproduction)

从零手写复现《Attention Is All You Need》(Vaswani et al., 2017) 完整架构，训练并开源高质量**中文 → 英文**神经机器翻译模型（最高 **2.14 亿参数 `Transformer Big`**，支持多卡 H100 DDP + 原生 FlashAttention-2 + 本地 Apple Silicon 毫秒级推理）。

---

## ✨ 项目特性

- **纯手写核心架构**（`src/`）：缩放点积注意力、多头自注意力/交叉注意力、正余弦位置编码、Pre-LN / Post-LN 双模式残差层、Noam 学习率调度器、中英双语清洗分桶以及批量化 Beam Search 束搜索解码。
- **原论文 §3.4 三方共享词表绑定**：支持 37,000 规模中英联合 BPE 子词词表，`src_embedding == tgt_embedding == output_proj` 三方共享同一权重矩阵，英文技术专有名词（如 `ZooKeeper`、`Transformer`、`MySQL`）零碎片跨语种对齐。
- **全套可配置无损工程加速**（所有优化均可通过 YAML 一键切回纯手写原版实现）：
  - `use_flash_causal`: 解锁原生 FlashAttention-2 下三角因果硬件内核；
  - `unpad_loss`: 剔除 `<pad>` 位置的 37K 大词表线性投影与交叉熵冗余计算；
  - `fused_adam`: CUDA 单内核融合 `Adam(fused=True)`；
  - `fast_beam`: 活跃束矩阵批量化 Beam Search（Apple M2 Max 本地推理提速约 **10×**）；
  - `torchrun` 多卡 NVLink `DDP` 全局有效 Token 精确加权梯度同步与多卡分片并行验证。
- **6 级渐进式单元测试保障**：29 项 `pytest` 测试全覆盖（因果防泄露、Padding 不变性、权重绑定、梯度等价性、束搜索陷阱树等）。

---

## 📦 预训练模型下载与开箱即用推理（GitHub Release）

已在 [**Releases (`v1.0.0`)**](https://github.com/hiyasame/zh-en-transformer/releases/tag/v1.0.0) 中提供 3 个不同规模的训练完成模型包（每个压缩包解压后均自带 `avg5_final.pt` 五权重平均模型、`zh_spm.model`、`en_spm.model` 与 `config.yaml`）：

| 模型包 | 架构规模 | 参数量 | 训练语料与词表 | 评测表现 | 适用场景 |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **`modal_wmt_big.tar.gz`** | Transformer Big<br>(`N=6, d=1024, H=16, d_ff=4096`) | **214.25M** | 450 万多域句对 (`WMT19` + `News Commentary` + `OPUS`)<br>37,000 三方共享 BPE | **WMT `newstest2018`**<br>Val Greedy BLEU = **24.84** | **主力推荐**：技术文档、新闻时政、长难句翻译 |
| **`modal_base.tar.gz`** | Transformer Base<br>(`N=6, d=512, H=8, d_ff=2048`) | **62.57M** | 100 万句 (`OPUS-100`)<br>独立词表 (20K Zh / 16K En) | `OPUS-100` Test<br>Beam-4 BLEU = **35.26** | 轻量级快速推理与基线对比 |
| **`local_small.tar.gz`** | Transformer Small<br>(`N=4, d=256, H=4, d_ff=1024`) | **11.50M** | 6 万句 (`OPUS-100`)<br>独立词表 (8K / 8K) | 轻量级验证通过 | 本地笔记本 CPU/MPS 秒级调试体验 |

### 30 秒快速体验预训练旗舰模型

```bash
git clone https://github.com/hiyasame/zh-en-transformer.git
cd zh-en-transformer
pip install -r requirements.txt

# 下载并解压 2.14 亿参数旗舰模型包到 checkpoints/modal_wmt_big/
curl -L https://github.com/hiyasame/zh-en-transformer/releases/download/v1.0.0/modal_wmt_big.tar.gz | tar -xz

# 启动交互式中译英终端（自动识别 CUDA / Apple MPS / CPU）
python translate.py --ckpt checkpoints/modal_wmt_big/avg5_final.pt --interactive
```

### 翻译实测样例（`modal_wmt_big` 本地直出）

- **源句**：`ZooKeeper 提供了一系列简单，高性能的原语用于协调不同的分布式系统`
  - **译文**：`ZooKeeper provides a series of simple, high-performance primitives for coordinating different distributed systems`
- **源句**：`Transformer 模型完全抛弃了循环与卷积结构，仅依靠多头注意力机制捕捉全局上下文依赖。`
  - **译文**：`Transformer models completely discarded cyclic and convolution structures, relying only on multi-head attention mechanisms to capture global context dependency.`
- **源句**：`美联储周三宣布维持联邦基金利率目标区间不变，但暗示通胀压力正在逐步缓解。`
  - **译文**：`The Fed announced on Wednesday that the target range for the federal funds rate would remain unchanged, but hinted that inflationary pressures were gradually easing.`

---

## 🛠️ 从零训练指南

### 1. 运行全套单元测试

```bash
pytest -v
```

### 2. 本地 Apple Silicon (MPS) / 单卡快速训练（约 20~30 分钟）

```bash
# Step 1: 下载清洗语料、训练 SentencePiece 分词器并导出预编码包
python scripts/prepare_data.py --config configs/local_small.yaml

# Step 2 (可选): 30 秒过拟合 32 条真实中英句子，验证全链路无误
python train.py --config configs/local_small.yaml --overfit32

# Step 3: 正式训练 6 万句小规模模型
python train.py --config configs/local_small.yaml

# Step 4: 最近 5 个 Checkpoint 权重平均并评测测试集 SacreBLEU
python translate.py --ckpt-dir checkpoints/local_small --beam 4
```

### 3. Modal 云端 `4× H100` 论文级复现（推荐）

数据预处理自动在 16 核纯 CPU 容器执行，多卡训练完成后即刻释放 4 卡集群并交由单卡容器完成权重平均与评测：

```bash
pip install modal
modal setup

# 方案 A (旗舰级): 4x H100 训练 450 万多域句对 + 2.14 亿参数 Transformer Big (~40-48 分钟)
modal run --detach modal_train.py --scale big --gpu H100:4

# 方案 B (标准满步数): 4x H100 训练 450 万多域句对 + 标准 Transformer Base (~25-32 分钟，共享缓存数据)
modal run --detach modal_train.py --scale base --gpu H100:4

# 云端跑完后流式拉回本地 checkpoints/
modal run modal_train.py --scale big --download-only
```

---

## 📂 目录结构

```text
├── configs/
│   ├── local_small.yaml      # 本地 Apple Silicon 6万句轻量配置
│   ├── modal_base.yaml       # 单卡 H100/A100 100万句 OPUS Base 配置
│   ├── modal_wmt_base.yaml   # 4x H100 450万多域句对 + 37K 共享词表 Base 配置
│   └── modal_wmt_big.yaml    # 4x H100 450万多域句对 + 37K 共享词表 2.14亿参数 Big 配置
├── scripts/
│   ├── prepare_data.py       # 多源流式清洗去重、联合 BPE 训练、16核并行编码与 CSR 打包
│   └── pack_for_colab.py     # Colab 源码打包辅助工具
├── src/
│   ├── model.py              # 手写 Transformer 核心架构 (兼容 Pre-LN/Post-LN 与原生 FlashAttention-2)
│   ├── data.py               # 双语清洗、CSR 零膨胀数据集与动态 Token 分桶采样器
│   ├── decode.py             # Greedy 解码、手写 Beam Search 与矩阵批量化 Fast Beam Search
│   ├── optim.py              # 论文 §5.3 Noam 学习率调度器
│   └── tokenizer.py          # SentencePiece BPE 训练与编解码封装
├── tests/                    # Level 1 ~ Level 6 共 29 项渐进式单元测试
├── train.py                  # 单卡/多卡 DDP 训练主引擎
├── translate.py              # 多权重算术平均、SacreBLEU 评测与交互式翻译 CLI
└── modal_train.py            # Modal 云端 CPU 预处理 + 多卡 H100 训练 + 单卡评测编排
```
