"""
Module 6: Neural Networks - Numerical Examples

This script demonstrates key concepts from Module 6 with concrete calculations.
Run with: pixi run python course-companion-computations/module6_examples.py

References in Course Companion:
- demo_parameter_counting()         -> Module 6, Section 6.1 (Parameter Counting)
- demo_relu_vs_sigmoid_gradients()  -> Module 6, Section 6.1 (Activation Functions)
- demo_xor_hidden_layer()           -> Module 6, Section 6.1 (XOR Problem)
- demo_cross_entropy_vs_mse()       -> Module 6, Section 6.2 (Loss Functions)
- demo_backprop_by_hand()           -> Module 6, Section 6.2 (Backpropagation)
- demo_learning_rate_effects_nn()   -> Module 6, Section 6.2 (Learning Rate)
- demo_dropout_effect()             -> Module 6, Section 6.2 (Dropout)
- demo_early_stopping()             -> Module 6, Section 6.2 (Early Stopping)

Last updated: 2026-01-02
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def demo_parameter_counting():
    """
    Walk through parameter counting for different network architectures.
    Shows how depth vs width affects parameter count.
    """
    print("=" * 60)
    print("PARAMETER COUNTING WALKTHROUGH")
    print("=" * 60)

    def count_params(architecture):
        """Count parameters for a fully connected architecture."""
        total = 0
        details = []
        for i in range(len(architecture) - 1):
            input_size = architecture[i]
            output_size = architecture[i + 1]
            weights = input_size * output_size
            biases = output_size
            layer_params = weights + biases
            total += layer_params
            details.append({
                'layer': i + 1,
                'shape': f'{input_size}x{output_size}',
                'weights': weights,
                'biases': biases,
                'total': layer_params,
            })
        return total, details

    # Example 1: The architecture from the module
    arch1 = [784, 256, 128, 10]
    total1, details1 = count_params(arch1)

    print(f"\nArchitecture 1: {arch1}")
    print(f"{'Layer':>6} {'Shape':>12} {'Weights':>10} {'Biases':>8} {'Total':>10}")
    print("-" * 50)
    for d in details1:
        print(f"{d['layer']:>6} {d['shape']:>12} {d['weights']:>10,} {d['biases']:>8} {d['total']:>10,}")
    print("-" * 50)
    print(f"{'Total':>6} {'':<12} {'':<10} {'':<8} {total1:>10,}")

    # Example 2: Deeper but narrower
    arch2 = [784, 128, 64, 32, 16, 10]
    total2, details2 = count_params(arch2)

    print(f"\nArchitecture 2 (deeper, narrower): {arch2}")
    print(f"{'Layer':>6} {'Shape':>12} {'Weights':>10} {'Biases':>8} {'Total':>10}")
    print("-" * 50)
    for d in details2:
        print(f"{d['layer']:>6} {d['shape']:>12} {d['weights']:>10,} {d['biases']:>8} {d['total']:>10,}")
    print("-" * 50)
    print(f"{'Total':>6} {'':<12} {'':<10} {'':<8} {total2:>10,}")

    # Example 3: Wide but shallow
    arch3 = [784, 512, 10]
    total3, details3 = count_params(arch3)

    print(f"\nArchitecture 3 (wide, shallow): {arch3}")
    print(f"{'Layer':>6} {'Shape':>12} {'Weights':>10} {'Biases':>8} {'Total':>10}")
    print("-" * 50)
    for d in details3:
        print(f"{d['layer']:>6} {d['shape']:>12} {d['weights']:>10,} {d['biases']:>8} {d['total']:>10,}")
    print("-" * 50)
    print(f"{'Total':>6} {'':<12} {'':<10} {'':<8} {total3:>10,}")

    print(f"\nComparison:")
    print(f"  Architecture 1 (3 layers): {total1:>10,} parameters")
    print(f"  Architecture 2 (5 layers): {total2:>10,} parameters")
    print(f"  Architecture 3 (2 layers): {total3:>10,} parameters")
    print(f"\nThe first layer (784→hidden) dominates the parameter count.")
    print(f"Deeper networks can have FEWER parameters than wide shallow ones.")


def demo_relu_vs_sigmoid_gradients():
    """
    Show the vanishing gradient problem numerically.
    Demonstrates why ReLU enables training deep networks.
    """
    print("\n" + "=" * 60)
    print("RELU VS SIGMOID GRADIENTS (VANISHING GRADIENT PROBLEM)")
    print("=" * 60)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)

    def relu(x):
        return np.maximum(0, x)

    def relu_derivative(x):
        return 1.0 if x > 0 else 0.0

    # Show gradients at different input values
    x_values = np.array([-5, -2, -1, 0, 1, 2, 5])

    print(f"\nGradients at different input values:")
    print(f"{'x':>6} {'sigmoid(x)':>12} {'sig_grad':>12} {'relu(x)':>10} {'relu_grad':>12}")
    print("-" * 55)
    for x in x_values:
        print(f"{x:>6} {sigmoid(x):>12.4f} {sigmoid_derivative(x):>12.4f} "
              f"{relu(x):>10.1f} {relu_derivative(x):>12.1f}")

    # Simulate gradient flow through multiple layers
    print(f"\nGradient flowing backward through 10 layers:")
    print(f"(Starting gradient = 1.0, each layer multiplies by activation gradient)")
    print()

    # Sigmoid: gradient through saturated neurons (x=2 for all)
    sigmoid_grad = 1.0
    relu_grad = 1.0
    x_saturated = 2.0  # Saturated region for sigmoid

    print(f"{'Layer':>6} {'Sigmoid grad':>15} {'ReLU grad':>15}")
    print("-" * 40)
    for layer in range(10, 0, -1):
        print(f"{layer:>6} {sigmoid_grad:>15.6f} {relu_grad:>15.1f}")
        sigmoid_grad *= sigmoid_derivative(x_saturated)
        relu_grad *= relu_derivative(x_saturated)

    print(f"\nAfter 10 layers:")
    print(f"  Sigmoid gradient: {sigmoid_grad:.2e} (effectively zero!)")
    print(f"  ReLU gradient: {relu_grad:.1f} (unchanged)")
    print(f"\nThis is the vanishing gradient problem. Sigmoid gradients")
    print(f"shrink exponentially, making early layers nearly impossible to train.")


def demo_xor_hidden_layer():
    """
    Train a tiny neural network to solve XOR.
    Shows how the hidden layer transforms the space.
    """
    print("\n" + "=" * 60)
    print("XOR WITH A HIDDEN LAYER")
    print("=" * 60)

    torch.manual_seed(42)

    # XOR data
    X = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    print(f"\nXOR truth table:")
    print(f"{'x1':>4} {'x2':>4} {'XOR':>6}")
    print("-" * 16)
    for i in range(4):
        print(f"{X[i, 0].item():>4.0f} {X[i, 1].item():>4.0f} {y[i, 0].item():>6.0f}")

    # Simple network: 2 inputs -> 4 hidden -> 1 output
    class XORNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(2, 4)
            self.output = nn.Linear(4, 1)

        def forward(self, x):
            h = torch.tanh(self.hidden(x))  # tanh works better for XOR
            return torch.sigmoid(self.output(h))

        def get_hidden_representation(self, x):
            return torch.tanh(self.hidden(x))

    model = XORNet()
    optimizer = optim.Adam(model.parameters(), lr=0.5)
    criterion = nn.BCELoss()

    # Train
    for epoch in range(2000):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    # Show results
    print(f"\nAfter training:")
    print(f"{'Input':>10} {'Output':>10} {'Prediction':>12}")
    print("-" * 35)

    model.eval()
    with torch.no_grad():
        for i in range(4):
            output = model(X[i:i+1])
            pred = 1 if output.item() > 0.5 else 0
            print(f"({X[i, 0]:.0f}, {X[i, 1]:.0f}){'':<4} {output.item():>10.3f} {pred:>12}")

    print(f"\nThe hidden layer transforms the inputs so that:")
    print(f"  (0,0) and (1,1) map to similar hidden representations")
    print(f"  (0,1) and (1,0) map to different hidden representations")
    print(f"  The output layer can then linearly separate them!")


def demo_cross_entropy_vs_mse():
    """
    Show why cross-entropy works better than MSE for classification.
    Demonstrates gradient behavior for confident-wrong predictions.
    """
    print("\n" + "=" * 60)
    print("CROSS-ENTROPY VS MSE FOR CLASSIFICATION")
    print("=" * 60)

    def mse_loss(y_true, y_pred):
        return (y_true - y_pred) ** 2

    def mse_gradient(y_true, y_pred):
        return -2 * (y_true - y_pred)

    def cross_entropy_loss(y_true, y_pred):
        eps = 1e-10  # Prevent log(0)
        return -(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

    def cross_entropy_gradient(y_true, y_pred):
        eps = 1e-10
        return -(y_true / (y_pred + eps)) + (1 - y_true) / (1 - y_pred + eps)

    # Scenario: True label is 1, but model predicts various probabilities
    y_true = 1.0
    predictions = [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01]

    print(f"\nTrue label: {y_true} (positive class)")
    print(f"Comparing loss and gradients at different predictions:")
    print()
    print(f"{'Prediction':>12} {'MSE Loss':>12} {'MSE Grad':>12} {'CE Loss':>12} {'CE Grad':>12}")
    print("-" * 65)

    for p in predictions:
        mse_l = mse_loss(y_true, p)
        mse_g = mse_gradient(y_true, p)
        ce_l = cross_entropy_loss(y_true, p)
        ce_g = cross_entropy_gradient(y_true, p)
        print(f"{p:>12.2f} {mse_l:>12.4f} {mse_g:>12.4f} {ce_l:>12.4f} {ce_g:>12.2f}")

    print(f"\nKey insight:")
    print(f"  When prediction is 0.01 (confident AND wrong):")
    print(f"    MSE gradient: {mse_gradient(1.0, 0.01):.2f}")
    print(f"    CE gradient:  {cross_entropy_gradient(1.0, 0.01):.2f}")
    print(f"\n  Cross-entropy provides MUCH stronger gradients for")
    print(f"  confident wrong predictions, enabling faster correction.")


def demo_backprop_by_hand():
    """
    Walk through backpropagation on a minimal network.
    Shows forward pass, loss, and gradient computation step by step.
    """
    print("\n" + "=" * 60)
    print("BACKPROPAGATION BY HAND")
    print("=" * 60)

    print(f"\nNetwork: 1 input -> 1 hidden (ReLU) -> 1 output (sigmoid)")
    print(f"Single training example: x=0.5, y=1")

    # Initialize weights (simple values for clarity)
    w1 = 0.5   # input to hidden weight
    b1 = 0.1   # hidden bias
    w2 = 0.8   # hidden to output weight
    b2 = -0.2  # output bias

    x = 0.5
    y_true = 1.0

    print(f"\nInitial weights: w1={w1}, b1={b1}, w2={w2}, b2={b2}")

    # Forward pass
    print(f"\n--- FORWARD PASS ---")
    z1 = w1 * x + b1
    print(f"z1 = w1*x + b1 = {w1}*{x} + {b1} = {z1}")

    h1 = max(0, z1)  # ReLU
    print(f"h1 = ReLU(z1) = max(0, {z1}) = {h1}")

    z2 = w2 * h1 + b2
    print(f"z2 = w2*h1 + b2 = {w2}*{h1} + {b2} = {z2}")

    y_pred = 1 / (1 + np.exp(-z2))  # Sigmoid
    print(f"y_pred = sigmoid(z2) = 1/(1+e^(-{z2:.3f})) = {y_pred:.4f}")

    # Loss (binary cross-entropy)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    print(f"\nLoss = -[y*log(y_pred) + (1-y)*log(1-y_pred)] = {loss:.4f}")

    # Backward pass
    print(f"\n--- BACKWARD PASS ---")

    # Gradient of loss w.r.t. y_pred
    dL_dy = -y_true / y_pred + (1 - y_true) / (1 - y_pred)
    print(f"dL/dy_pred = {dL_dy:.4f}")

    # Gradient of sigmoid: sigmoid(z) * (1 - sigmoid(z))
    dy_dz2 = y_pred * (1 - y_pred)
    print(f"dy_pred/dz2 = y_pred*(1-y_pred) = {dy_dz2:.4f}")

    # Chain rule: dL/dz2
    dL_dz2 = dL_dy * dy_dz2
    print(f"dL/dz2 = dL/dy_pred * dy_pred/dz2 = {dL_dz2:.4f}")

    # Gradients for w2 and b2
    dL_dw2 = dL_dz2 * h1
    dL_db2 = dL_dz2 * 1
    print(f"\ndL/dw2 = dL/dz2 * h1 = {dL_dz2:.4f} * {h1} = {dL_dw2:.4f}")
    print(f"dL/db2 = dL/dz2 * 1 = {dL_db2:.4f}")

    # Propagate to hidden layer
    dL_dh1 = dL_dz2 * w2
    print(f"\ndL/dh1 = dL/dz2 * w2 = {dL_dz2:.4f} * {w2} = {dL_dh1:.4f}")

    # ReLU gradient: 1 if z1 > 0, else 0
    dh1_dz1 = 1.0 if z1 > 0 else 0.0
    dL_dz1 = dL_dh1 * dh1_dz1
    print(f"dh1/dz1 = ReLU'(z1) = {dh1_dz1:.1f} (z1={z1} > 0)")
    print(f"dL/dz1 = dL/dh1 * dh1/dz1 = {dL_dz1:.4f}")

    # Gradients for w1 and b1
    dL_dw1 = dL_dz1 * x
    dL_db1 = dL_dz1 * 1
    print(f"\ndL/dw1 = dL/dz1 * x = {dL_dz1:.4f} * {x} = {dL_dw1:.4f}")
    print(f"dL/db1 = dL/dz1 * 1 = {dL_db1:.4f}")

    # Update step
    lr = 0.1
    print(f"\n--- WEIGHT UPDATE (lr={lr}) ---")
    print(f"w1_new = w1 - lr*dL/dw1 = {w1} - {lr}*{dL_dw1:.4f} = {w1 - lr*dL_dw1:.4f}")
    print(f"b1_new = b1 - lr*dL/db1 = {b1} - {lr}*{dL_db1:.4f} = {b1 - lr*dL_db1:.4f}")
    print(f"w2_new = w2 - lr*dL/dw2 = {w2} - {lr}*{dL_dw2:.4f} = {w2 - lr*dL_dw2:.4f}")
    print(f"b2_new = b2 - lr*dL/db2 = {b2} - {lr}*{dL_db2:.4f} = {b2 - lr*dL_db2:.4f}")

    print(f"\nPyTorch automates all of this with loss.backward()!")


def demo_learning_rate_effects_nn():
    """
    Show how learning rate affects neural network training.
    Uses a simple regression problem for clarity.
    """
    print("\n" + "=" * 60)
    print("LEARNING RATE EFFECTS ON NEURAL NETWORKS")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # Simple regression data
    X = torch.linspace(-2, 2, 100).reshape(-1, 1)
    y = torch.sin(X * np.pi) + torch.randn_like(X) * 0.1

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1, 32)
            self.fc2 = nn.Linear(32, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    learning_rates = [0.0001, 0.01, 1.0]
    labels = ["Too small (0.0001)", "Good (0.01)", "Too large (1.0)"]

    print(f"\nTraining same network with different learning rates (50 epochs):")
    print(f"{'Learning Rate':>20} {'Final Loss':>15} {'Status':>20}")
    print("-" * 60)

    for lr, label in zip(learning_rates, labels):
        torch.manual_seed(42)  # Same initialization
        model = SimpleNet()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(50):
            epoch_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)

            # Check for explosion
            if np.isnan(avg_loss) or avg_loss > 1e6:
                break

        final_loss = losses[-1]
        if np.isnan(final_loss) or final_loss > 100:
            status = "DIVERGED"
            final_str = "inf"
        elif final_loss > 0.5:
            status = "Barely learning"
            final_str = f"{final_loss:.4f}"
        else:
            status = "Converged"
            final_str = f"{final_loss:.4f}"

        print(f"{label:>20} {final_str:>15} {status:>20}")

    print(f"\nInterpretation:")
    print(f"  - lr=0.0001: Too slow, hasn't converged in 50 epochs")
    print(f"  - lr=0.01:   Good balance of speed and stability")
    print(f"  - lr=1.0:    Overshoots minima, loss explodes")


def demo_dropout_effect():
    """
    Show how dropout reduces overfitting.
    Trains on small dataset where overfitting is easy.
    """
    print("\n" + "=" * 60)
    print("DROPOUT EFFECT ON OVERFITTING")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # Small dataset (easy to overfit)
    n_train = 50
    n_test = 200
    n_features = 20

    # Generate data with some noise
    X_train = torch.randn(n_train, n_features)
    true_weights = torch.randn(n_features, 1)
    y_train = X_train @ true_weights + torch.randn(n_train, 1) * 0.5

    X_test = torch.randn(n_test, n_features)
    y_test = X_test @ true_weights + torch.randn(n_test, 1) * 0.5

    class NetWithDropout(nn.Module):
        def __init__(self, dropout_rate):
            super().__init__()
            self.fc1 = nn.Linear(n_features, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            return self.fc3(x)

    dropout_rates = [0.0, 0.3, 0.5]

    print(f"\nTraining with {n_train} samples, {n_features} features (overfit-prone):")
    print(f"{'Dropout':>10} {'Train MSE':>12} {'Test MSE':>12} {'Gap':>10}")
    print("-" * 48)

    for dropout in dropout_rates:
        torch.manual_seed(42)
        model = NetWithDropout(dropout)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Train
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train)
            train_mse = criterion(train_pred, y_train).item()
            test_pred = model(X_test)
            test_mse = criterion(test_pred, y_test).item()

        gap = test_mse - train_mse
        print(f"{dropout:>10.1f} {train_mse:>12.4f} {test_mse:>12.4f} {gap:>10.4f}")

    print(f"\nInterpretation:")
    print(f"  - Dropout=0: Big gap between train/test (overfitting)")
    print(f"  - Dropout=0.3-0.5: Smaller gap, better generalization")
    print(f"  - Higher dropout forces distributed representations")


def demo_early_stopping():
    """
    Show train vs validation loss curves and optimal stopping point.
    """
    print("\n" + "=" * 60)
    print("EARLY STOPPING IN ACTION")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # Data that's easy to overfit
    n_train = 100
    n_val = 100
    n_features = 10

    X_train = torch.randn(n_train, n_features)
    true_weights = torch.randn(n_features, 1)
    y_train = X_train @ true_weights + torch.randn(n_train, 1) * 0.3

    X_val = torch.randn(n_val, n_features)
    y_val = X_val @ true_weights + torch.randn(n_val, 1) * 0.3

    class OverfitNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(n_features, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    model = OverfitNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    epochs = 150

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Validate
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            val_losses.append(val_loss.item())

    # Find optimal stopping point
    best_epoch = np.argmin(val_losses)
    best_val_loss = val_losses[best_epoch]

    print(f"\nLoss over {epochs} epochs:")
    print(f"{'Epoch':>8} {'Train Loss':>12} {'Val Loss':>12}")
    print("-" * 35)

    # Show key epochs
    key_epochs = [0, 10, 25, 50, best_epoch, 100, 149]
    key_epochs = sorted(set([e for e in key_epochs if e < epochs]))

    for e in key_epochs:
        marker = " <-- best" if e == best_epoch else ""
        print(f"{e:>8} {train_losses[e]:>12.4f} {val_losses[e]:>12.4f}{marker}")

    print(f"\nOptimal stopping point: epoch {best_epoch}")
    print(f"  Train loss at stop: {train_losses[best_epoch]:.4f}")
    print(f"  Val loss at stop:   {val_losses[best_epoch]:.4f}")
    print(f"\nIf we kept training to epoch {epochs-1}:")
    print(f"  Train loss: {train_losses[-1]:.4f} (lower!)")
    print(f"  Val loss:   {val_losses[-1]:.4f} (higher - overfitting)")
    print(f"\nEarly stopping saves the model at epoch {best_epoch},")
    print(f"preventing {epochs - best_epoch - 1} epochs of wasted overfitting.")


if __name__ == "__main__":
    demo_parameter_counting()
    demo_relu_vs_sigmoid_gradients()
    demo_xor_hidden_layer()
    demo_cross_entropy_vs_mse()
    demo_backprop_by_hand()
    demo_learning_rate_effects_nn()
    demo_dropout_effect()
    demo_early_stopping()
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
