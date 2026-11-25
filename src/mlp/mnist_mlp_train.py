import argparse
import sys
from pathlib import Path


checkpoint_name = "mnist_mlp_training.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP on MNIST.")
    parser.add_argument(
        "--checkpoint",
        metavar="NAME",
        default=checkpoint_name,
        help=(
            "Weights filename in par/mlp. "
            "The .pt suffix is optional (default: %(default)s)."
        ),
    )
    checkpoint_name = parser.parse_args().checkpoint
    if (
        checkpoint_name in {"", ".", ".."}
        or Path(checkpoint_name).name != checkpoint_name
        or "\\" in checkpoint_name
    ):
        parser.error("--checkpoint expects a filename without a directory")


if not checkpoint_name.lower().endswith(".pt"):
    checkpoint_name += ".pt"

script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent.parent if script_dir.parent.name == "src" else script_dir
checkpoint_path = project_dir / "par" / "mlp" / checkpoint_name
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    class _Tee:
        def __init__(self, stream, log_file):
            self._stream = stream
            self._log_file = log_file

        def write(self, text):
            written = self._stream.write(text)
            self._log_file.write(text)
            return written

        def flush(self):
            self._stream.flush()
            self._log_file.flush()

        def __getattr__(self, name):
            return getattr(self._stream, name)

    log_dir = project_dir / "result" / checkpoint_path.parent.name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / checkpoint_path.with_suffix(".log").name
    _log_file = log_path.open(
        "a", encoding="utf-8", errors="backslashreplace", buffering=1
    )
    sys.stdout = _Tee(sys.stdout, _log_file)
    sys.stderr = _Tee(sys.stderr, _log_file)


from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch


def evaluate(model, data_loader, loss_fn, device):
    ## An evaluator used to monitor model's performence on a selected dataset with a given loss function.
    model.eval()
    correct_ans = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            flat_images = flatten(images)
            scores = model(flat_images)
            predictions = scores.argmax(dim=1)
            loss = loss_fn(scores, labels)
            loss_sum += loss.item() * labels.shape[0]
            correct_ans += (predictions == labels).sum().item()
            total += labels.shape[0]

    avg_loss = loss_sum / total
    accuracy = correct_ans / total
    return correct_ans, total, avg_loss, accuracy


def create_model():
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

to_tensor = transforms.ToTensor()

flatten = nn.Flatten(start_dim=1)

model = create_model().to(device)

##linear = nn.Linear(in_features=28**2, out_features=10).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

full_train_data = datasets.MNIST(
    root="./data", train=True, download=False, transform=to_tensor
)

train_data, val_data = random_split(
    full_train_data, lengths=[50000, 10000], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

val_loader = DataLoader(dataset=val_data, batch_size=64, shuffle=False)

epochs = 200
best_val_loss = float("inf")
best_epoch = 0

for epoch in range(epochs):
    print("\nEpoch:", epoch + 1, "/", epochs)

    model.train()
    training_step = 0
    train_loss_sum = 0.0
    train_total = 0

    print("----------Training starts----------\n")

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        flat_images = flatten(images)
        scores = model(flat_images)
        predictions = scores.argmax(dim=1)
        loss = loss_fn(scores, labels)
        train_loss_sum += loss.item() * labels.shape[0]
        train_total += labels.shape[0]
        loss.backward()
        optimizer.step()

        training_step += 1
        # if training_step % 100 == 0:
        #    print("batch:", training_step, "loss:", loss.item())

    print("Training completed with " + str(training_step) + " steps.")
    print("Average training loss: " + str(train_loss_sum / train_total))

    print("----------Validation starts----------\n")

    (val_correct_ans, val_total, avg_val_loss, val_accuracy) = evaluate(
        model, val_loader, loss_fn, device
    )
    print("Cases tested: ", val_total)
    print("Cases passed: ", val_correct_ans)
    print("Accuracy: ", val_accuracy)
    print("Average loss: ", avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        state = model.state_dict()
        torch.save(state, checkpoint_path)
        print(
            "Parameters in epoch",
            best_epoch,
            f"is saved to {checkpoint_path}",
        )

print("Training and validation done in", epochs, "epochs.\n")
print("Best epoch: ", best_epoch)
print("Best validation loss: ", best_val_loss)
##state = linear.state_dict()
##torch.save(state, checkpoint_path)
##print(f"Parameters saved to {checkpoint_path}")

print("\n----------Testing starts----------\n")

best_model = create_model().to(device)

saved_state = torch.load(
    checkpoint_path, weights_only=True, map_location=device
)
best_model.load_state_dict(saved_state)
best_model.eval()

test_data = datasets.MNIST(
    root="./data", train=False, download=False, transform=to_tensor
)

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

(test_correct_ans, test_total, avg_test_loss, test_accuracy) = evaluate(
    best_model, test_loader, loss_fn, device
)
print("Cases tested: ", test_total)
print("Cases passed: ", test_correct_ans)
print("Accuracy: ", test_accuracy)
print("Average loss: ", avg_test_loss)
