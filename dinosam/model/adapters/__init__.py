import torch.nn as nn


def get_adapter(name: str, in_dim: int, factor: int = 8) -> nn.Module:
    if name == "mona":
        from ..mona import Mona
        return Mona(in_dim, factor)
    elif name == "fc":
        from .fc_adapter import FCAdapter
        return FCAdapter(in_dim, factor)
    elif name == "dual_attn":
        from .dual_attn_adapter import DualAttnAdapter
        return DualAttnAdapter(in_dim, factor)
    else:
        raise ValueError(f"Unknown adapter type: {name}")
