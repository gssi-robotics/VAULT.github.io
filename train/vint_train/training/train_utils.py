"""Runtime helpers used by the deployment nodes.

Only ``get_action`` is imported at deployment time; the original training
utilities (loss, optimizer, eval loops, wandb logging) have been removed
along with the rest of the training pipeline.
"""

import numpy as np
import torch


# Action min/max from the original NoMaD training data_config.yaml.
# Kept inline so the deployment runtime does not depend on the YAML.
ACTION_STATS = {
    "min": np.array([-2.5, -4.0]),
    "max": np.array([5.0, 4.0]),
}


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    return ndata * (stats["max"] - stats["min"]) + stats["min"]


def get_action(diffusion_output, action_stats=ACTION_STATS):
    device = diffusion_output.device
    ndeltas = diffusion_output.reshape(diffusion_output.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)
