import copy

import torch
import torch.nn as nn

def safe_clone(module: nn.Module) -> nn.Module:
    """Clone a module safely by detaching from computation graph."""
    device = next(module.parameters()).device

    module_cpu = module.to("cpu")

    cloned = copy.deepcopy(module_cpu)
    cloned.load_state_dict(module_cpu.state_dict())

    cloned = cloned.to(device)
    module = module.to(device)

    return cloned