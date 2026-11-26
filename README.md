# MNIST Digit Recognition with PyTorch

This repository compares a linear classifier, two MLP variants, a CNN, and a small Vision Transformer on MNIST. Training scripts save model weights and logs under matching experiment names.

## Repository layout

| Path | Contents |
| --- | --- |
| [`src/mlp/`](src/mlp/) | Training scripts for the linear classifier and both MLP variants, plus a linear classifier walkthrough with Chinese comments |
| [`src/cnn/`](src/cnn/) | CNN training and validation-set confusion matrix analysis |
| [`src/miniViT/`](src/miniViT/) | MiniViT implementation and training script |
| [`par/`](par/) | Model weights under `mlp/`, `cnn/`, and `MiniViT/` |
| [`result/`](result/) | Training logs under the same three model directories |
| `data/MNIST/` | Local MNIST dataset downloaded through torchvision |

## Model architectures

| Model | Architecture |
| --- | --- |
| Linear classifier | Flatten the `1×28×28` image and apply `Linear(784, 10)`. |
| MLP | Flatten the image, then apply `Linear(784, 128) → ReLU → Linear(128, 10)`. |
| MLP without ReLU | Apply `Linear(784, 128) → Linear(128, 10)` to the flattened image. This variant shows the effect of removing the activation. |
| CNN | A `3×3` convolution maps one input channel to eight channels with padding of one. ReLU and `2×2` max pooling follow. The classifier flattens the resulting `8×14×14` features and outputs 10 class scores. |
| MiniViT | Split the image into 49 non-overlapping `4×4` patches, project them to tokens, prepend a learned CLS token, and add learned position embeddings. After the encoder, apply LayerNorm and a linear classifier to the CLS representation. |

The [MiniViT training configuration](src/miniViT/mnist_miniViT_train.py) uses **4 encoder blocks, 4 attention heads, a token width of 32, and an MLP hidden width of 64**. Each block applies self-attention and a GELU MLP in sequence, with LayerNorm before each sublayer and a residual connection around it. The [model implementation](src/miniViT/vit_encoder.py) also includes a manual single-head attention variant available through `use_manual_attention=True`.

## Setup

Activate a Python environment with `torch` and `torchvision` installed. The CNN analysis script also needs `matplotlib`. Run the examples from the **repository root** so the training scripts can find `./data`.

Download the official training and test sets once per machine, or make an existing copy available under `data/MNIST/`.

```bash
python - <<'PY'
from torchvision.datasets import MNIST

MNIST(root="data", train=True, download=True)
MNIST(root="data", train=False, download=True)
PY
```

## Training

Choose one of the following scripts. Each accepts `--help`.

```bash
python src/mlp/mnist_train.py --checkpoint linear_baseline
python src/mlp/mnist_mlp_train.py --checkpoint mlp_relu
python src/mlp/mnist_mlp_no_relu_train.py --checkpoint mlp_no_relu
python src/mlp/mnist_train_GPT_annotated.py --checkpoint linear_one_epoch
python src/cnn/mnist_cnn_train.py --checkpoint cnn_baseline
python src/miniViT/mnist_miniViT_train.py --checkpoint vit_4l_4h
```

`--checkpoint` takes a filename. The scripts append `.pt` when the extension is omitted and use their own default filename when the flag is omitted. The MiniViT command above saves weights to `par/MiniViT/vit_4l_4h.pt` and writes a matching log to `result/MiniViT/vit_4l_4h.log`. Output also appears in the terminal.

The linear classifier and MLPs use the `mlp/` subdirectory, and the CNN uses `cnn/`. The scripts create the output directories as needed. Reusing a name appends to the existing log and writes to the same checkpoint file. Use distinct names to keep separate experiments.

The main training scripts use SGD with a learning rate of `0.01`, cross-entropy loss, and a batch size of `64`. They currently run `200` epochs, configurable through the `epochs` variable in each script. They use a fixed split of 50,000 training images and 10,000 validation images, save the checkpoint with the lowest validation loss, and evaluate that checkpoint on the official 10,000-image test set. The annotated linear classifier runs one epoch on the full training set, evaluates it, and saves its final weights.

The main training scripts use CUDA when available and otherwise run on CPU. In a Slurm batch job that has requested a GPU, run the same training script from the repository root.

```bash
srun python -u src/miniViT/mnist_miniViT_train.py --checkpoint vit_4l_4h
```

To inspect CNN predictions on the validation set, run `python src/cnn/mnist_confusion.py`. It loads `par/cnn/mnist_cnn_training.pt`, prints a confusion matrix, and saves examples of correctly classified `7`s and `7`s misclassified as `9`.

See [LICENSE](LICENSE) for licensing information.
