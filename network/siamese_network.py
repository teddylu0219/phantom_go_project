
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ELU(inplace=True),
    )

class PhantomGoSiamese(nn.Module):
    """
    Siamese network with two encoders:
    - anchor_encoder: encodes observation history tensor (C_obs x N x N)
    - board_encoder: encodes board tensor (2 x N x N)
    Both produce L2-normalized embeddings in a shared space.
    """
    def __init__(self, obs_in_channels: int, board_in_channels: int = 2, embed_dim: int = 512):
        super().__init__()
        self.obs_in_channels = obs_in_channels
        self.board_in_channels = board_in_channels
        self.embed_dim = embed_dim

        # Anchor encoder (a bit deeper; histories are richer)
        self.anchor_feat = nn.Sequential(
            conv_block(obs_in_channels, 64),
            conv_block(64, 64),
            conv_block(64, 128, s=2),     # downsample to expand RF
            conv_block(128, 128),
            conv_block(128, 256, s=2),    # downsample
            conv_block(256, 256),
            conv_block(256, 256),
        )
        self.anchor_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # GAP -> (B,256,1,1)
            nn.Flatten(),
            nn.Linear(256, embed_dim, bias=False),
        )

        # Board encoder (light yet capable)
        self.board_feat = nn.Sequential(
            conv_block(board_in_channels, 64),
            conv_block(64, 64),
            conv_block(64, 128, s=2),
            conv_block(128, 128),
            conv_block(128, 256, s=2),
            conv_block(256, 256),
        )
        self.board_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, embed_dim, bias=False),
        )

    def encode_anchor(self, anchor: torch.Tensor) -> torch.Tensor:
        """
        anchor: (B, C_obs, N, N)
        returns (B, D) L2-normalized
        """
        x = self.anchor_feat(anchor)
        x = self.anchor_head(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def encode_board(self, board: torch.Tensor) -> torch.Tensor:
        """
        board: (B, 2, N, N)
        returns (B, D) L2-normalized
        """
        x = self.board_feat(board)
        x = self.board_head(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    @torch.no_grad()
    def compute_distances(self, anchor: torch.Tensor, boards: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise L2 distances between one batch of anchors and a set of boards.
        anchor:  (B, C_obs, N, N)
        boards:  (B, K, 2, N, N)
        returns: (B, K) distances
        """
        B, K = boards.shape[0], boards.shape[1]
        anc_emb = self.encode_anchor(anchor)  # (B, D)
        boards_flat = boards.view(B * K, *boards.shape[2:])
        brd_emb = self.encode_board(boards_flat).view(B, K, -1)  # (B, K, D)
        # L2 distance: ||a - b||_2
        # expand anc to (B, K, D)
        anc_exp = anc_emb.unsqueeze(1).expand_as(brd_emb)
        d = torch.norm(anc_exp - brd_emb, dim=2)
        return d

    @staticmethod
    def softmin_weights(distances: torch.Tensor, temperature: float = 10.0) -> torch.Tensor:
        """
        distances: (B, K), returns softmin weights over K.
        """
        return torch.softmax(-distances / temperature, dim=1)
