from __future__ import annotations

import base64
import json
from pathlib import Path
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ZIP = PROJECT_ROOT / "translator_colab_code.zip"
OUT_IPYNB = PROJECT_ROOT / "colab_train.ipynb"

INCLUDE_PATHS = [
    "src",
    "configs",
    "scripts",
    "tests",
    "train.py",
    "translate.py",
    "modal_train.py",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
]


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if "__pycache__" in parts or ".pytest_cache" in parts:
        return True
    if path.suffix in {".pyc", ".pyo", ".DS_Store"} or path.name == ".DS_Store":
        return True
    return False


def build_notebook(b64_zip: str) -> dict:
    # 将 base64 折成 100 字符一行，保持 JSON 干净易读
    b64_lines = [b64_zip[i : i + 100] for i in range(0, len(b64_zip), 100)]
    cell2_lines = [
        "# 2. 自动释放内置源码包（无需手动上传 zip！若手动上传了新 zip 则优先使用新 zip）并安装依赖\n",
        "import base64, io, zipfile\n",
        "from pathlib import Path\n",
        "\n",
        "EMBEDDED_CODE_B64 = (\n",
    ]
    for line in b64_lines:
        cell2_lines.append(f"    '{line}'\n")
    cell2_lines.extend(
        [
            ")\n",
            "\n",
            "work_dir = Path('/content/translator')\n",
            "work_dir.mkdir(parents=True, exist_ok=True)\n",
            "manual_zip = Path('/content/translator_colab_code.zip')\n",
            "if manual_zip.exists():\n",
            "    with zipfile.ZipFile(manual_zip, 'r') as zf:\n",
            "        zf.extractall(work_dir)\n",
            "    print(f'已从手动上传的 {manual_zip} 解压代码')\n",
            "else:\n",
            "    raw_bytes = base64.b64decode(EMBEDDED_CODE_B64)\n",
            "    with zipfile.ZipFile(io.BytesIO(raw_bytes), 'r') as zf:\n",
            "        zf.extractall(work_dir)\n",
            "    print(f'已自动释放笔记本内置源码至 {work_dir}（零额外上传！）')\n",
            "\n",
            "%cd /content/translator\n",
            "!pip install -q sentencepiece datasets sacrebleu pyyaml tqdm pytest\n",
            "!pytest -q",
        ]
    )

    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 《Attention Is All You Need》中译英 (Zh → En) Colab 一键训练台\n",
                    "\n",
                    "**本笔记本已内置你的全部源码，无需再额外上传任何 `.zip` 文件！**\n",
                    "\n",
                    "1. 顶部菜单 **代码执行程序 (Runtime) → 更改运行时类型 (Change runtime type)** → 选 **L4 GPU**（省额度首选）或 **A100 GPU**（最快）。\n",
                    "2. 直接从上到下依次运行下方 5 个单元格即可（数据和模型权重会自动存进你的 Google Drive，断线零损失）。",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. 检查 GPU 型号并挂载 Google Drive\n",
                    "!nvidia-smi\n",
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell2_lines,
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. 选择训练档位并准备数据 (自动缓存在你的 Google Drive，只跑一次)\n",
                    "#   - 'colab_base' : 100 万句, 论文标准 6 层 512 维 Transformer Base (25000 步: T4 约 5.5~6 小时, L4 约 1.5 小时, A100 约 45 分钟)\n",
                    "#   - 'local_small': 6 万句短句, 4 层 256 维 (T4 约 15 分钟热身)\n",
                    "MODE = 'colab_base'\n",
                    "\n",
                    "CONFIG_PATH = f'configs/{MODE}.yaml'\n",
                    "DRIVE_ROOT = f'/content/drive/MyDrive/translator_runs/{MODE}'\n",
                    "DATA_DIR = f'{DRIVE_ROOT}/data'\n",
                    "CKPT_DIR = f'{DRIVE_ROOT}/checkpoints'\n",
                    "\n",
                    "from pathlib import Path\n",
                    "if not (Path(DATA_DIR) / 'dataset.pt').exists():\n",
                    "    !python scripts/prepare_data.py --config {CONFIG_PATH} --out-dir {DATA_DIR}\n",
                    "else:\n",
                    "    print(f'检测到 Google Drive 已有预处理数据缓存: {DATA_DIR}/dataset.pt，直接秒读！')",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 4. 启动训练（若 Drive 中已有 last.pt 则自动无损断点续训；T4 自动 fp16、L4/A100 自动 bf16+compile）\n",
                    "from pathlib import Path\n",
                    "\n",
                    "last_ckpt = Path(CKPT_DIR) / 'last.pt'\n",
                    "if last_ckpt.exists():\n",
                    "    print(f'发现已有进度 {last_ckpt}，自动断点续训...')\n",
                    "    !python train.py --config {CONFIG_PATH} --data-dir {DATA_DIR} --save-dir {CKPT_DIR} --device cuda --resume {last_ckpt}\n",
                    "else:\n",
                    "    !python train.py --config {CONFIG_PATH} --data-dir {DATA_DIR} --save-dir {CKPT_DIR} --device cuda",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 5. 训完收官：平均最近 5 个权重 -> 评测测试集 Beam Search SacreBLEU -> 永久存入 Google Drive -> 自动关机停计费！\n",
                    "avg_ckpt = f'{CKPT_DIR}/avg5_final.pt'\n",
                    "!python translate.py --ckpt-dir {CKPT_DIR} --save-avg {avg_ckpt} --device cuda --beam 4 --max-samples 500\n",
                    "\n",
                    "print(f'\\n全部完成！最终模型已永久保存在 Google Drive: {CKPT_DIR}/avg5_final.pt')\n",
                    "print('正在自动释放 GPU 运行时并停止计费...')\n",
                    "from google.colab import runtime\n",
                    "runtime.unassign()",
                ],
            },
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "L4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    count = 0
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in INCLUDE_PATHS:
            target = PROJECT_ROOT / rel
            if not target.exists():
                continue
            if target.is_file():
                zf.write(target, arcname=rel)
                count += 1
            else:
                for sub in sorted(target.rglob("*")):
                    if sub.is_file() and not should_skip(sub):
                        zf.write(sub, arcname=str(sub.relative_to(PROJECT_ROOT)))
                        count += 1

    b64_str = base64.b64encode(OUT_ZIP.read_bytes()).decode("ascii")
    nb_dict = build_notebook(b64_str)
    nb_json = json.dumps(nb_dict, ensure_ascii=False, indent=1)
    # 校验生成的 JSON 100% 合法
    json.loads(nb_json)
    OUT_IPYNB.write_text(nb_json, encoding="utf-8")

    print(
        f"已打包 {count} 个源码文件:\n"
        f"  - 单文件自包含笔记本 (只需上传这一个!): {OUT_IPYNB} ({OUT_IPYNB.stat().st_size / 1024:.1f} KB)\n"
        f"  - 备用源码压缩包: {OUT_ZIP} ({OUT_ZIP.stat().st_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    main()
