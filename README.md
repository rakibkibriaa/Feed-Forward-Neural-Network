# Feed-Forward Neural Network from Scratch

A fully connected neural network written from scratch in NumPy and trained to classify clothing images from [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist). There's no PyTorch or TensorFlow in the model itself. Every layer has its own forward and backward pass, and backpropagation is written out by hand. torchvision is used only to download the dataset.

The best model reaches **89.25% test accuracy** and **0.892 macro-F1** on the 10,000-image test set.

## What's implemented

Each building block is its own class with a `forward` and a `backward` method. Layers are stacked in a `NeuralNetwork` container, so changing the architecture only means changing a few `add_layer` calls.

- **Dense layer.** A fully connected layer with Xavier (Glorot) uniform initialization. The backward pass computes the weight and bias gradients, updates the parameters, and passes the gradient on to the previous layer.
- **Batch normalization.** Normalizes each mini-batch, then applies a learnable scale (γ) and shift (β). The backward pass derives the gradients through the batch mean and variance in full.
- **ReLU activation.**
- **Dropout.** Inverted dropout: activations are scaled by `1/keep_prob` during training so their expected value stays the same.
- **Adam optimizer.** Each Dense layer has its own Adam state, with first and second moment estimates and bias correction (β₁ = 0.9, β₂ = 0.999).
- **Softmax + cross-entropy.** Softmax is numerically stabilized by subtracting the max logit before exponentiating. The combined gradient `ŷ − y` feeds straight into the backward pass.

Everything is vectorized. A mini-batch moves through the network as one `(features × batch)` matrix, with no per-sample loops.

## Experiments

The 60,000 training images were split 80/20 into training and validation sets. I compared three architectures, each trained with four learning rates (15 epochs, batch size 256, Adam), and picked the best model by validation macro-F1.

| Model | Architecture |
|---|---|
| 1 | 784 → 256 → 64 → 10 |
| 2 | 784 → 1024 → 256 → 64 → 10 |
| 3 | 784 → 512 → 128 → 32 → 10, with dropout 0.3 after the first two hidden layers |

Every hidden layer uses Dense → BatchNorm → ReLU.

**Validation macro-F1 (%)**

| Learning rate | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| 0.005  | 88.87 | 89.71 | 88.15 |
| 0.003  | –     | 89.75 | –     |
| 0.001  | 88.86 | –     | 88.28 |
| 0.0005 | 88.78 | **89.99** | 88.08 |
| 0.0001 | 88.94 | 89.81 | 86.81 |

**Best model: Model 2 with learning rate 0.0005**

| | Accuracy | Macro-F1 | Loss |
|---|---|---|---|
| Validation | 89.98% | 89.99% | 0.334 |
| **Test** | **89.25%** | **89.25%** | 0.373 |

Some observations:

- **The wider network helped.** Model 2 was the best at every learning rate I tried. The gain over Model 1 is about one point, which suggests the smaller network is capacity-limited on this dataset.
- **Learning rate mattered less than architecture.** Within each model, validation F1 stayed within about half a point across learning rates. The one exception is Model 3 at 0.0001, which dropped noticeably.
- **Dropout didn't pay off here.** Model 3 did worst overall. Part of the reason is probably 15 epochs being too short for a regularized network. The other part is that my dropout layer has no separate evaluation mode (see Limitations), so it also drops units when evaluating.

The full report, with per-epoch loss, accuracy, and F1 curves and a confusion matrix for every run, is in `code/report_1905098.pdf`.

## Repository layout

```
├── README.md
├── Spec.pdf                 # problem specification: required components, dataset, and evaluation criteria
└── code/
    ├── 1905098.ipynb        # implementation, training, and evaluation
    ├── model_1905098.pkl    # trained weights of the best model (Model 2, lr = 0.0005)
    ├── report_1905098.pdf   # full results with plots and confusion matrices
    └── requirements.txt
```

## Running it

```bash
git clone https://github.com/rakibkibriaa/Feed-Forward-Neural-Network.git
cd Feed-Forward-Neural-Network

python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
jupyter notebook code/1905098.ipynb
```

`requirements.txt` includes `pywin32`, which only installs on Windows. On Linux or macOS, delete that line first. The packages actually needed are `numpy`, `torch`, `torchvision`, `scikit-learn`, `matplotlib`, and `tqdm`.

**To train from scratch,** run the cells from top to bottom. The first cell downloads Fashion-MNIST into `data/`, and the second defines the layers, builds the network, and trains it. To try a different architecture, comment out the current block of `add_layer` calls and uncomment another one.

**To evaluate the saved model without retraining,** set `num_epochs = 0` in the training cell and run it. That builds Model 2 without training it. Then uncomment the `load_model_parameters(...)` line in the loading cell, point it at `model_1905098.pkl`, and run the test cell.

## Limitations

- **Batch norm has no running statistics.** At inference, each batch is normalized with its own mean and variance instead of averages stored during training. This works here because validation and test sets are evaluated in a single pass, but predictions on a single image, or a very small batch, would be unreliable.
- **Dropout has no train/eval switch,** so it also drops units during validation and testing. Adding a `training` flag to the network would fix both this and the batch norm issue.
- **An MLP ignores spatial structure.** Each image is flattened to a 784-dimensional vector, so the network can't use the 2D layout of pixels. A small CNN would typically do several points better on Fashion-MNIST. The point of this project was to understand backpropagation from the ground up, not to beat a convolutional model.

## Author

**Rakib Kibria** · [GitHub](https://github.com/rakibkibriaa)
