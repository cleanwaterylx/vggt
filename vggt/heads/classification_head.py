import torch.nn as nn
import torch
from functools import partial
from torch.utils.checkpoint import checkpoint
from vggt.layers.block import Block
from vggt.layers.mlp import Mlp
from vggt.layers.attention import FlashAttentionRope
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        need_project=True,
        use_checkpoint=False,
        num_token_groups=9,   # 新增
    ):
        super().__init__()

        self.num_token_groups = num_token_groups

        # 如果 token 已经是 C 维，可以不需要 projection
        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        self.use_checkpoint = use_checkpoint
        
        self.rope = RotaryPositionEmbedding2D(frequency=100)
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.blocks = nn.ModuleList([
            Block(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                attn_class=FlashAttentionRope,
                rope=rope
            ) for _ in range(depth)
        ])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, pos=None):
        """
        hidden: [B, S, P, 25*C]
        """
        B, S, P, C_all = hidden.shape
        C = C_all // self.num_token_groups

        # 把 concat 的 feature 还原成 token
        hidden = hidden.view(B, S, P, self.num_token_groups, C)

        # 合并 patch 和 group
        hidden = hidden.reshape(B, S, P * self.num_token_groups, C)

        # transformer batch
        hidden = hidden.view(B * S, P * self.num_token_groups, C)

        hidden = self.projects(hidden)

        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, pos=pos, use_reentrant=False)
            else:
                hidden = blk(hidden, pos=pos)

        out = self.linear_out(hidden)

        return out

class ClassificationHead(nn.Module):
    """ 
    Classification head for image-level classification
    """

    def __init__(self, dec_embed_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dec_embed_dim, dec_embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dec_embed_dim // 2, 1)
        )


    def forward(self, decout, N, patch_start_idx, temp=32.0):
        # classification decoder output: (B*N, num_register_tokens, dec_embed_dim)
        BN, n, c = decout.shape
        B = BN // N
        
        # ---- 1. 分离 register 和 patch ----
        reg = decout[:, 1:patch_start_idx, :]       # (B*N, K, C)
        patch = decout[:, patch_start_idx:, :]     # (B*N, P, C)
        
        # ---- 2. 对 patch 做 attention guided by reg ----
        # 先得到 reg_mean 做 query
        reg_mean = reg.mean(dim=1, keepdim=True)   # (B*N, 1, C)

        # attention score: patch · reg_mean^T / temp
        attn_score = torch.matmul(patch, reg_mean.transpose(1, 2)) / temp   # (B*N, P, 1)
        attn_weight = torch.softmax(attn_score, dim=1)                      # (B*N, P, 1)

        patch_pool = (attn_weight * patch).sum(dim=1)                        # (B*N, C)

        # ---- 3. 合并 reg + patch_pool ----
        feat = reg_mean.squeeze(1) + patch_pool   # (B*N, C)
        feat = feat.view(B, N, c)                # (B, N, C)

        # ---- 4. 分类 ----
        logits = self.classifier(feat).squeeze(-1)  # (B, N)

        # return logits, feat
        return logits
