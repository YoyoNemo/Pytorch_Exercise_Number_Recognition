import torch
from torch import nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, kernal_size=4, outputDim=32):
        super().__init__()
        self.kernal_size = kernal_size
        self.inputDim = kernal_size**2
        self.outputDim = outputDim
        self.proj = nn.Linear(self.inputDim, self.outputDim)

    def forward(self, images):
        patches = F.unfold(
            images, kernel_size=self.kernal_size, stride=self.kernal_size
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
    def __init__(self, layers=1):
        super().__init__()
        self.patch_embed = PatchEmbedding()
        self.encoder = nn.Sequential(
            *[EncoderBlock(use_manual_attention=False) for _ in range(layers)]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 32))
        self.pos_embed = nn.Parameter(0.02 * torch.randn(1, 50, 32))
        self.norm = nn.LayerNorm(32)
        self.classifier = nn.Linear(32, 10)

    def forward(self, images):
        patches = self.patch_embed(images)
        cls = self.cls_token.expand(patches.shape[0], -1, -1)
        cls_patches = torch.cat((cls, patches), dim=1)
        pos_embed_cls_patches = cls_patches + self.pos_embed
        encoded = self.encoder(pos_embed_cls_patches)
        final_cls = encoded[:, 0, :]
        normalized = self.norm(final_cls)
        return self.classifier(normalized)


# Used for debuging
if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    dataset = datasets.MNIST(
        root="/home/luming/project/pytorch exercise/data",
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )
    images, labels = next(iter(DataLoader(dataset, batch_size=2)))
    model = MiniViT()
    logits = model(images)
    print("result: ", logits.shape)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    print("pos_embed 收到梯度:", model.pos_embed.grad is not None)
    print("loss: ", loss)
