import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardTripletLoss(nn.Module):
    """
    Triplet loss with Euclidean (L2) distance and margin m (default 1.0).
    Expects L2-normalized embeddings but works generally.
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        # d(a,p), d(a,n)
        d_ap = torch.norm(anchor - positive, dim=1)
        d_an = torch.norm(anchor - negative, dim=1)
        loss = F.relu(self.margin + d_ap - d_an)
        return loss.mean()
