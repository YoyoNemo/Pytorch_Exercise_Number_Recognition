import torch
from torch import nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=4, outputDim=32, in_channels=1):
        super().__init__()
        self.patch_size = patch_size
        self.inputDim = in_channels * patch_size**2
        self.outputDim = outputDim
        self.proj = nn.Linear(self.inputDim, self.outputDim)

    def forward(self, images):
        patches = F.unfold(
            images, kernel_size=self.patch_size, stride=self.patch_size
        ).transpose(1, 2)
        return self.proj(patches)


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = Q @ K.transpose(-2, -1) / (Q.shape[-1] ** 0.5)
        weights = scores.softmax(dim=-1)
        mixed = weights @ V
        return mixed, weights


class EncoderBlock(nn.Module):
    def __init__(self, dim=32, heads=4, mlp_dim=64, use_manual_attention=False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.use_manual_attention = use_manual_attention

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        if use_manual_attention:
            self.attn = SingleHeadSelfAttention(dim=dim)
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=heads, batch_first=True
            )
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        normalized1 = self.norm1(x)
        if self.use_manual_attention:
            attn_out, _ = self.attn(normalized1)
        else:
            attn_out, _ = self.attn(
                normalized1, normalized1, normalized1, need_weights=False
            )
        add_res = attn_out + x
        normalized2 = self.norm2(add_res)
        mlp_out = self.mlp(normalized2)
        return mlp_out + add_res


class MiniViT(nn.Module):
    def __init__(
        self,
        encode_layers=1,
        dim=32,
        heads=4,
        mlp_dim=64,
        image_size=28,
        patch_size=4,
        in_channels=1,
        num_classes=10,
        use_manual_attention=False,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size, in_channels=in_channels, outputDim=dim
        )
        self.encoder = nn.Sequential(
            *[
                EncoderBlock(
                    dim=dim,
                    heads=heads,
                    mlp_dim=mlp_dim,
                    use_manual_attention=use_manual_attention,
                )
                for _ in range(encode_layers)
            ]
        )
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(0.02 * torch.randn(1, num_patches + 1, dim))
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, images):
        patches = self.patch_embed(images)
        cls = self.cls_token.expand(patches.shape[0], -1, -1)
        cls_patches = torch.cat((cls, patches), dim=1)
        pos_embed_cls_patches = cls_patches + self.pos_embed
        encoded = self.encoder(pos_embed_cls_patches)
        final_cls = encoded[:, 0, :]
        normalized = self.norm(final_cls)
        return self.classifier(normalized)
