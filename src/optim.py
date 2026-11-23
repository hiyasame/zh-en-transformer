from __future__ import annotations

import torch.optim as optim


# ============================================================================
# Level 4: Noam Learning Rate Scheduler (Paper §5.3 Eq. 3)
# ============================================================================


class NoamScheduler:
    """论文 §5.3 公式 (3) 学习率调度器：

        lr = factor * (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    约定：
        - 初始 step_num = 0；每次调用 scheduler.step() 时，step_num 自增 1（因此第 1 次 step() 对应 step=1），
          并将计算出的最新 lr 写入 optimizer.param_groups 中每个 group 的 'lr' 字段。
        - 提供 get_last_lr() -> float 返回最近一次更新后的学习率，以及 state_dict() / load_state_dict() 以便断点续训。
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
    ) -> None:
        self.step_num = 0
        self._last_lr = 0.0
        self.factor = factor
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.optimizer = optimizer

    def step(self) -> float:
        """推进 1 个真实参数更新步，更新 optimizer 学习率并返回当前 lr。"""
        self.step_num += 1
        lr = self.factor * (self.d_model ** (-0.5)) * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))
        self._last_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_last_lr(self) -> float:
        """返回最近一次设定的学习率。"""
        return self._last_lr

    def state_dict(self) -> dict:
        return { "step_num": self.step_num, "_last_lr": self._last_lr }

    def load_state_dict(self, state_dict: dict) -> None:
        self.step_num = state_dict["step_num"]
        self._last_lr = state_dict["_last_lr"]
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self._last_lr