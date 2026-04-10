from typing import Dict, Tuple

import torch
import torch.nn as nn


class NeuronCoverage:
    """
    Simple neuron coverage tracker.
    Coverage = fraction of neuron units activated above threshold at least once.

    For Conv2d outputs, each channel is treated as one neuron by spatial averaging.
    For Linear outputs, each dimension is treated as one neuron.
    """
    def __init__(self, model: nn.Module, threshold: float = 0.0):
        self.model = model
        self.threshold = threshold
        self.handles = []
        self.covered: Dict[str, torch.Tensor] = {}
        self.total_neurons = 0
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, layer_name: str):
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return

            with torch.no_grad():
                out = output.detach()

                if out.dim() == 4:
                    reduced = out.mean(dim=(2, 3))
                elif out.dim() == 2:
                    reduced = out
                else:
                    return

                num_units = reduced.shape[1]
                key = f"{layer_name}_{num_units}"

                if key not in self.covered:
                    self.covered[key] = torch.zeros(
                        num_units, dtype=torch.bool, device=reduced.device
                    )

                activated = (reduced > self.threshold).any(dim=0)
                self.covered[key] |= activated
        return hook

    def reset(self):
        for layer_name in self.covered:
            self.covered[layer_name].zero_()

    def coverage(self) -> Tuple[int, int, float]:
        covered_units = 0
        total_units = 0
        for _, mask in self.covered.items():
            covered_units += mask.sum().item()
            total_units += mask.numel()

        ratio = covered_units / total_units if total_units > 0 else 0.0
        return covered_units, total_units, ratio

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()