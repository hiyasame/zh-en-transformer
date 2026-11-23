from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import threading
import yaml
import modal

APP_NAME = "zh-en-transformer"
VOLUME_NAME = "zh-en-transformer-vol"
REMOTE_CODE_DIR = "/root/translator"

SCALE_PRESETS: dict[str, tuple[str, str]] = {
    "big": ("configs/modal_wmt_big.yaml", "modal_wmt_big"),
    "base": ("configs/modal_wmt_base.yaml", "modal_wmt_base"),
    "opus_base": ("configs/modal_base.yaml", "modal_base"),
}

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "sentencepiece>=0.2.0",
        "datasets>=2.19.0,<4.0.0",
        "sacrebleu>=2.4.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    )
    .add_local_dir(
        ".",
        remote_path=REMOTE_CODE_DIR,
        ignore=[
            "data",
            "checkpoints",
            "runs",
            ".git",
            ".idea",
            "__pycache__",
            ".pytest_cache",
            "*.pt",
            "*.zip",
            "*.ipynb",
        ],
    )
)


def _resolve_data_name(config_rel_path: str, fallback_name: str, root_dir: Path | None = None) -> str:
    """从 YAML 配置的 data.processed_dir 提取数据集目录名，使 big 与 base 自动共享同一份 450 万句缓存。"""
    cfg_path = (root_dir / config_rel_path) if root_dir is not None else Path(config_rel_path)
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        proc_dir = cfg.get("data", {}).get("processed_dir")
        if proc_dir:
            return Path(proc_dir).name
    return fallback_name


@app.function(
    image=image,
    cpu=16.0,
    memory=32768,
    volumes={"/vol": vol},
    timeout=7200,
)
def prepare_data_remote(
    config: str = "configs/modal_wmt_big.yaml",
    data_name: str = "modal_wmt_45m",
    force: bool = False,
) -> str:
    """在廉价 16 核纯 CPU 容器上拉取多域语料、联合训练 37K BPE 并多核并行编码（不消耗一秒钟 H100 GPU 额度！）。"""
    vol.reload()
    data_dir = Path(f"/vol/data/{data_name}")
    dataset_file = data_dir / "dataset.pt"

    if dataset_file.exists() and not force:
        size_mb = dataset_file.stat().st_size / (1024 * 1024)
        msg = f"[缓存命中] 已存在预处理数据包 {dataset_file} ({size_mb:.1f} MB)，直接跳过下载与分词！"
        print(msg)
        return str(data_dir)

    print(f"[16核 CPU 极速预处理] 正在生成多域平行语料包 ({config}) -> {data_dir} ...")
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--config",
            config,
            "--out-dir",
            str(data_dir),
        ],
        cwd=REMOTE_CODE_DIR,
        check=True,
    )
    vol.commit()
    return str(data_dir)


@app.function(
    image=image,
    gpu="H100:4",
    cpu=16.0,
    memory=65536,
    volumes={"/vol": vol},
    timeout=21600,
)
def train_remote(
    config: str = "configs/modal_wmt_big.yaml",
    run_name: str = "modal_wmt_big",
    data_name: str = "modal_wmt_45m",
    max_steps: int = 0,
    fresh: bool = False,
) -> str:
    """支持多卡 (H100:4 NVLink DDP) 原生 FlashAttention-2 训练；训练一结束立即释放 4 卡集群，绝不为评测空烧多卡额度！"""
    import torch

    vol.reload()
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_name = torch.cuda.get_device_name(0) if num_gpus > 0 else "CPU"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if num_gpus > 0 else 0.0
    print(
        f"=== Modal GPU 集群已就绪: {num_gpus}x {gpu_name} "
        f"(单卡 {vram_gb:.1f} GB, 总显存 {vram_gb * max(1, num_gpus):.1f} GB VRAM) ==="
    )

    data_dir = Path(f"/vol/data/{data_name}")
    ckpt_dir = Path(f"/vol/checkpoints/{run_name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stop_commit = threading.Event()

    def _periodic_commit() -> None:
        while not stop_commit.wait(180.0):
            try:
                vol.commit()
            except Exception as exc:  # noqa: BLE001
                print(f"[Volume Commit 提示] {exc}")

    committer = threading.Thread(target=_periodic_commit, daemon=True)
    committer.start()

    try:
        if num_gpus > 1:
            launcher = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={num_gpus}",
                "train.py",
            ]
        else:
            launcher = [sys.executable, "train.py"]

        train_cmd = [
            *launcher,
            "--config",
            config,
            "--data-dir",
            str(data_dir),
            "--save-dir",
            str(ckpt_dir),
            "--device",
            "cuda",
        ]
        if max_steps > 0:
            train_cmd.extend(["--max-steps", str(max_steps)])

        last_ckpt = ckpt_dir / "last.pt"
        if last_ckpt.exists() and not fresh:
            print(f"[断点续训] 检测到已有进度 {last_ckpt}，自动接续训练...")
            train_cmd.extend(["--resume", str(last_ckpt)])

        subprocess.run(train_cmd, cwd=REMOTE_CODE_DIR, check=True)
        vol.commit()
    finally:
        stop_commit.set()
        vol.commit()

    return str(ckpt_dir)


@app.function(
    image=image,
    gpu=["A10G", "L40S", "A100", "H100", "L4"],
    cpu=4.0,
    memory=16384,
    volumes={"/vol": vol},
    timeout=3600,
)
def eval_and_average_remote(
    run_name: str = "modal_wmt_big",
    data_name: str = "modal_wmt_45m",
    eval_beam: int = 4,
    eval_samples: int = 500,
) -> dict[str, str]:
    """自动抢占任意零排队的空闲单卡 (A10G/L40S/A100/H100/L4)，秒起容器完成 5 权重平均与批量化 Beam-4 评测。"""
    import torch

    vol.reload()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    data_dir = Path(f"/vol/data/{data_name}")
    ckpt_dir = Path(f"/vol/checkpoints/{run_name}")
    avg_ckpt = ckpt_dir / "avg5_final.pt"

    print(f"[单卡 {gpu_name} 权重平均与评测] 正在生成 {avg_ckpt} 并评测 Beam-{eval_beam} SacreBLEU ...")
    eval_cmd = [
        sys.executable,
        "translate.py",
        "--ckpt-dir",
        str(ckpt_dir),
        "--data-dir",
        str(data_dir),
        "--save-avg",
        str(avg_ckpt),
        "--device",
        "cuda",
        "--beam",
        str(eval_beam),
        "--max-samples",
        str(eval_samples),
    ]
    subprocess.run(eval_cmd, cwd=REMOTE_CODE_DIR, check=True)
    vol.commit()
    return {
        "ckpt_dir": str(ckpt_dir),
        "avg_ckpt": str(avg_ckpt),
    }


def download_checkpoints_to_local(run_name: str, include_best: bool = False) -> None:
    local_dir = Path("checkpoints") / run_name
    local_dir.mkdir(parents=True, exist_ok=True)
    files_to_pull = ["avg5_final.pt", "zh_spm.model", "en_spm.model", "config.yaml"]
    if include_best:
        files_to_pull.append("best.pt")
    print(f"\n正在通过流式通道将云端模型拉取回本地目录: {local_dir} ...")
    for fname in files_to_pull:
        rel = f"checkpoints/{run_name}/{fname}"
        out_file = local_dir / fname
        try:
            total_bytes = 0
            with open(out_file, "wb") as f_out:
                for chunk in vol.read_file(rel):
                    f_out.write(chunk)
                    total_bytes += len(chunk)
            print(f"  - 已下载 {out_file} ({total_bytes / (1024 * 1024):.1f} MB)")
        except Exception as exc:  # noqa: BLE001
            out_file.unlink(missing_ok=True)
            print(f"  - [跳过] {rel}: {exc}")
    print(
        f"\n本地模型已就绪！现在可直接离线交互翻译：\n"
        f"  python translate.py --ckpt {local_dir / 'avg5_final.pt'} --interactive"
    )


@app.local_entrypoint()
def main(
    scale: str = "big",
    gpu: str = "H100:4",
    config: str = "",
    run_name: str = "",
    max_steps: int = 0,
    fresh: bool = False,
    force_data: bool = False,
    eval_only: bool = False,
    download: bool = True,
    download_only: bool = False,
) -> None:
    """Modal 本地命令行总入口（默认启动 4x H100 跑 450 万句旗舰 Transformer Big）。

    常用示例：
        # 1. 旗舰方案：4x H100 跑 450万多域语料 + 37K 共享词表 + 2.14亿参数 Transformer Big (~40-48分钟)
        modal run modal_train.py --scale big --gpu H100:4

        # 2. 若训练已完成，仅在云端秒开空闲单卡做 5 权重平均 + Beam-4 评测并下载回本地
        modal run modal_train.py --scale big --eval-only

        # 3. 单独把云端已生成的 avg5_final.pt 拉取回本地 Mac
        modal run modal_train.py --scale big --download-only
    """
    if not config or not run_name:
        if scale not in SCALE_PRESETS:
            raise ValueError(f"未知 --scale '{scale}'，可选值: {list(SCALE_PRESETS.keys())}")
        preset_cfg, preset_run = SCALE_PRESETS[scale]
        config = config or preset_cfg
        run_name = run_name or preset_run

    data_name = _resolve_data_name(config, fallback_name=run_name)

    if download_only:
        download_checkpoints_to_local(run_name)
        return

    if not eval_only:
        print(
            f"==> 启动任务: scale={scale} | gpu={gpu} | config={config} | "
            f"data=/vol/data/{data_name} | ckpt=/vol/checkpoints/{run_name}"
        )

        # 第 1 阶段：在 16 核纯 CPU 容器准备 450 万句数据与 37K 联合词表（不烧一分钱 GPU 额度）
        prepare_data_remote.remote(config=config, data_name=data_name, force=force_data)

        # 第 2 阶段：启动多卡 DDP (H100:4) 极速训练，训完即刻释放 4 卡集群
        runner = train_remote.with_options(gpu=gpu) if gpu != "H100:4" else train_remote
        runner.remote(
            config=config,
            run_name=run_name,
            data_name=data_name,
            max_steps=max_steps,
            fresh=fresh,
        )

    # 第 3 阶段：自动挑选零排队单卡做 5 权重平均与批量化 Beam-4 评测
    res = eval_and_average_remote.remote(run_name=run_name, data_name=data_name)
    print(f"\n云端权重平均与评测完成: {res}")

    # 第 4 阶段：流式拉取模型回本地 checkpoints/<run_name>/
    if download:
        download_checkpoints_to_local(run_name)
