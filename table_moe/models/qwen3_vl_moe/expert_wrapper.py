import typing as tp
import torch
from torch import nn

from ...ops import maybe_run_fp16_fused_expert
from ...offload.linear_wrapper import OffloadLinearWrapper


def _maybe_dynamo_disable(fn):
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "disable"):
        return torch._dynamo.disable(fn)
    return fn


class QwenExpertWrapper(nn.Module):
    """
    把 Qwen expert 中的 (gate_proj, up_proj, down_proj) 映射到
    LinearCache 期望的 w1, w2, w3 + SwiGLU 逻辑。
    """
    def __init__(self, expert_module: nn.Module, device: torch.device):
        super().__init__()
        self.w1 = OffloadLinearWrapper(expert_module.gate_proj, device)
        self.w3 = OffloadLinearWrapper(expert_module.up_proj, device)
        self.w2 = OffloadLinearWrapper(expert_module.down_proj, device)
        self.act_fn = expert_module.act_fn

        # 加载参数的同步事件 (Load/Swap 完成后触发，Compute 等待)
        self.w1_event = torch.cuda.Event()
        self.w2_event = torch.cuda.Event()
        self.w3_event = torch.cuda.Event()
        
        # [新增] 计算完成事件 (Compute 完成后触发，Load/Prefetch 等待)
        # 用于防止 Write-After-Read (WAR) 竞争，即防止新参数覆盖正在计算的 Buffer
        self.compute_event = torch.cuda.Event()
        
        self.load = False
        self.free = True

    @_maybe_dynamo_disable
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.load:            
            # 只有在刚刚 Load/Swap 进来时才需要等待传输完成
            self.w1_event.wait()
            current_hidden_states = self.act_fn(self.w1(hidden_states))
            self.w3_event.wait()  
            current_hidden_states = current_hidden_states * self.w3(hidden_states)
            self.w2_event.wait()
            current_hidden_states = self.w2(current_hidden_states)

            self.load = False

        else:            
            current_hidden_states = maybe_run_fp16_fused_expert(self, hidden_states)
            if current_hidden_states is None:
                current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
                current_hidden_states = self.w2(current_hidden_states)
            
        # [新增] 记录计算结束事件
        # 表示当前 Stream 对此 Buffer 的使用已结束，后续的 Copy Stream 可以安全覆盖
        self.compute_event.record(torch.cuda.current_stream())
            
        return current_hidden_states
