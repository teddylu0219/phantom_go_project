import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from siamese_network import PhantomGoSiamese
from triplet_losses import StandardTripletLoss

torch.backends.cudnn.benchmark = True

def dummy_data_generator(batch_size=8, N=19, H=12, K=32, device='cpu'):
    """
    Generate random dummy tensors with correct shapes:
      - anchor: (B, H*6, N, N)
      - positive: (B, 2, N, N)
      - negatives: (B, K, 2, N, N)
    Values are {0,1} Bernoulli for simplicity.
    """
    anchor = torch.randint(0, 2, (batch_size, H*6, N, N), dtype=torch.float32, device=device)
    positive = torch.randint(0, 2, (batch_size, 2, N, N), dtype=torch.float32, device=device)
    negatives = torch.randint(0, 2, (batch_size, K, 2, N, N), dtype=torch.float32, device=device)
    return anchor, positive, negatives

def select_hard_negatives(model, anchor, negatives):
    """
    From K negatives pick the closest one for each sample.
    negatives: (B, K, 2, N, N)
    return: (B, 2, N, N)
    """
    with torch.no_grad():
        dists = model.compute_distances(anchor, negatives)  # (B, K)
        idx = torch.argmin(dists, dim=1)  # (B,)
    B = anchor.size(0)
    # Gather selected negatives
    gather_idx = idx.view(B, 1, 1, 1, 1).expand(B, 1, *negatives.shape[2:])  # (B,1,2,N,N)
    hard_negs = torch.gather(negatives, 1, gather_idx).squeeze(1)  # (B,2,N,N)
    return hard_negs

def train_step(model, optimizer, loss_fn, batch, device):
    model.train()
    anchor, positive, negatives = batch
    anchor, positive, negatives = anchor.to(device), positive.to(device), negatives.to(device)

    # Encode
    anc_emb = model.encode_anchor(anchor)         # (B, D)
    pos_emb = model.encode_board(positive)        # (B, D)

    # Semi-hard: pick closest negative per sample
    hard_negs = select_hard_negatives(model, anchor, negatives)
    neg_emb = model.encode_board(hard_negs)       # (B, D)

    loss = loss_fn(anc_emb, pos_emb, neg_emb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate_retrieval(model, device='cpu', B=16, N=19, H=12, K=64):
    """
    Mix the true positive into the K negatives and check argmin rank hit-rate.
    """
    model.eval()
    anchor, positive, negatives = dummy_data_generator(B, N, H, K, device)
    # Insert the positive randomly among negatives for each sample
    dists_neg = model.compute_distances(anchor, negatives)  # (B, K)
    # Now compute distance to true positive
    pos_d = torch.norm(model.encode_anchor(anchor) - model.encode_board(positive), dim=1)  # (B,)
    # Compare: for each i, is pos_d[i] smaller than min neg distance?
    min_neg_d, _ = dists_neg.min(dim=1)
    hits = (pos_d < min_neg_d).float().mean().item()
    return hits

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # Hyperparams
    N, H, K = 9, 12, 32
    obs_channels = H * 6
    embed_dim = 512
    batch_size = 16
    iters = 20

    model = PhantomGoSiamese(obs_in_channels=obs_channels, board_in_channels=2, embed_dim=embed_dim).to(device)
    loss_fn = StandardTripletLoss(margin=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Quick smoke training with dummy data
    for step in range(1, iters + 1):
        batch = dummy_data_generator(batch_size, N, H, K, device)
        loss = train_step(model, optimizer, loss_fn, batch, device)
        if step % 5 == 0:
            hit = evaluate_retrieval(model, device=device, B=16, N=N, H=H, K=K)
            print(f"[Iter {step:03d}] loss={loss:.4f}  retrieval_hit={hit:.3f}")

    # Show softmin example for one batch
    anchor, positive, negatives = dummy_data_generator(batch_size=2, N=N, H=H, K=8, device=device)
    d = model.compute_distances(anchor, negatives)  # (2, 8)
    w = model.softmin_weights(d, temperature=10.0)
    print("Distances[0]:", d[0].cpu().numpy())
    print("Weights[0]:  ", w[0].cpu().numpy())

if __name__ == "__main__":
    main()
