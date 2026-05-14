"""Implementations of general metric functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.text import Perplexity
from ..utils.registry import metric_registry


@metric_registry.register("acc")
class Accuracy(MulticlassAccuracy):
    """Subclass of `torchmetrics.classification.MulticlassAccuracy` that flattens inputs."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds_flat = preds.view(-1, preds.size(-1))
        preds_flat = preds_flat.argmax(axis=-1)
        target = target.view(-1)
        super().update(preds=preds_flat, target=target)


metric_registry.register("ppl", Perplexity) # ignore_index
metric_registry.register("cross_entropy", nn.CrossEntropyLoss)