from pathlib import Path
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def create_model():
    return nn.Sequential(
        nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(start_dim=1),
        nn.Linear(8 * 14 * 14, 10),
    )


project_dir = Path(__file__).resolve().parent

# 用和训练时相同的随机种子，取出同一组验证图片。
full_train_data = datasets.MNIST(
    root=str(project_dir / "data"),
    train=True,
    download=False,
    transform=transforms.ToTensor(),
)

_, val_data = random_split(
    full_train_data,
    lengths=[50000, 10000],
    generator=torch.Generator().manual_seed(42),
)

val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

model = create_model()
weights = torch.load(
    project_dir / "mnist_cnn_training.pt",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(weights)
model.eval()

# 整个验证集共用一张矩阵，行是真实类别，列是预测类别。
matrix = torch.zeros((10, 10), dtype=torch.long)
wrong_sevens = []
right_sevens = []

with torch.no_grad():
    for images, labels in val_loader:
        scores = model(images)
        predictions = scores.argmax(dim=1)
        wrong_mask = (labels == 7) & (predictions == 9)
        right_mask = (labels == 7) & (predictions == 7)
        wrong_sevens.extend(images[wrong_mask])
        right_sevens.extend(images[right_mask])

        for true_label, predicted_label in zip(labels.tolist(), predictions.tolist()):
            matrix[true_label, predicted_label] += 1

total = matrix.sum().item()
correct = matrix.diag().sum().item()

print("验证图片总数", total)
print("识别正确数", correct)
print(f"验证准确率 {correct / total:.2%}")
print("\n混淆矩阵(行是真实类别,列是预测类别)")
print(matrix)

print("筛出的图片数", len(wrong_sevens))  # 预期是 28

fig, axes = plt.subplots(2, 4, figsize=(8, 4))
for ax, image in zip(axes.flat, wrong_sevens[:8]):
    ax.imshow(image[0], cmap="gray")
    ax.axis("off")

fig.tight_layout()
fig.savefig(project_dir / "mnist_7_as_9.png", dpi=150)

for ax, image in zip(axes.flat, right_sevens[:8]):
    ax.imshow(image[0], cmap="gray")
    ax.axis("off")

fig.tight_layout()
fig.savefig(project_dir / "mnist_7_as_7.png", dpi=150)

image = wrong_sevens[7].reshape(1, 1, 28, 28)

with torch.no_grad():
    scores = model(image)

score_7 = scores[0, 7].item()
score_9 = scores[0, 9].item()

print(f"7 的分数 = {score_7:.3f}")
print(f"9 的分数 = {score_9:.3f}")
print(f"分数差 = {score_9 - score_7:.3f}")

masked_image = image.clone()
masked_image[0, 0, 14:17, 10:14] = 0  # 中横左端
masked_image[0, 0, 13:16, 18:20] = 0  # 中横右端

with torch.no_grad():
    masked_scores = model(masked_image)

gap_after = (masked_scores[0, 9] - masked_scores[0, 7]).item()

print(f"原图的分数差(9 - 7)= {score_9 - score_7:+.3f}")
print(f"遮挡后的分数差(9 - 7)= {gap_after:+.3f}")
print("遮挡后的预测类别 =", masked_scores.argmax(dim=1).item())
