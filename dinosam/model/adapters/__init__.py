import torch.nn as nn


def _remap_legacy_state_dict(state_dict):
    """Remap old mona1/mona2 keys to adapter1/adapter2 for backward compatibility.

    Only touches model_state_dict keys starting with 'mona1.' or 'mona2.'.
    Returns the state_dict with renamed keys. Idempotent for new-format checkpoints.
    """
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("mona1."):
            remapped["adapter1." + key[len("mona1."):]] = value
        elif key.startswith("mona2."):
            remapped["adapter2." + key[len("mona2."):]] = value
        else:
            remapped[key] = value
    return remapped


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
