# ===========================================================
# Simple Kolmogorov–Arnold Network (KAN) — 1D Example
# Inner and Outer Function Visualization
# ===========================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ===========================================================
# B-spline Activation (Inner Function ψ)
# ===========================================================
class BSplineActivation(nn.Module):
    def __init__(self, num_knots=11, degree=3):
        super().__init__()
        # Define knots between -π and π (since sin(x) is periodic in this range)
        knots = np.linspace(-np.pi, np.pi, num_knots)
        self.knots = torch.tensor(knots, dtype=torch.float32)
        self.degree = degree

        # Learnable coefficients for B-spline basis functions
        self.weights = nn.Parameter(torch.randn(num_knots - degree - 1))

    def bspline_basis(self, j, k, x):
        # Recursive Cox–de Boor formula
        if k == 0:
            return ((self.knots[j] <= x) & (x < self.knots[j + 1])).float()

        denom1 = self.knots[j + k] - self.knots[j]
        denom2 = self.knots[j + k + 1] - self.knots[j + 1]

        term1 = 0
        if denom1 > 0:
            term1 = ((x - self.knots[j]) / denom1) * \
                self.bspline_basis(j, k - 1, x)

        term2 = 0
        if denom2 > 0:
            term2 = ((self.knots[j + k + 1] - x) / denom2) * \
                self.bspline_basis(j + 1, k - 1, x)

        return term1 + term2

    def b_spline(self, x):
        # Ensure x is 1D
        x = x.squeeze(-1)
        n = len(self.knots) - 1
        B = torch.zeros((x.shape[0], n - self.degree), dtype=torch.float32)
        for j in range(n - self.degree):
            B[:, j] = self.bspline_basis(j, self.degree, x)
        return B

    def forward(self, x):
        # Ensure x is 1D
        x = x.squeeze(-1)
        B = self.b_spline(x)
        return torch.matmul(B, self.weights).unsqueeze(1)


# ===========================================================
# Simple KAN Model: 1 Input → 3 Hidden Nodes (Outer Φ) → 1 Output
# ===========================================================
class SimpleKAN(nn.Module):
    def __init__(self, input_dim=1, hidden_nodes=3, num_knots=11):
        super().__init__()
        self.hidden_nodes = nn.ModuleList(
            [BSplineActivation(num_knots) for _ in range(hidden_nodes)])
        self.output_layer = nn.Linear(hidden_nodes, 1)

    def forward(self, x):
        # Each hidden node is an outer function Φ_q(ψ_q(x))
        h = torch.cat([layer(x) for layer in self.hidden_nodes], dim=1)
        return self.output_layer(h)


# ===========================================================
# 1. Generate Training Data (y = sin(x))
# ===========================================================
x = torch.linspace(-np.pi, np.pi, 100).unsqueeze(1)
y = torch.sin(x)


# ===========================================================
# 2. Initialize Model, Optimizer, Loss
# ===========================================================
model = SimpleKAN(input_dim=1, hidden_nodes=3, num_knots=11)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()


# ===========================================================
# 3. Train Model
# ===========================================================
for epoch in range(2000):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    if epoch % 400 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


# ===========================================================
# 4. Plot the Inner Function (B-spline Basis)
# ===========================================================
inner_layer = model.hidden_nodes[0]  # Just take one (since all get same x)
B = inner_layer.b_spline(x).detach().numpy()

plt.figure(figsize=(10, 5))
plt.title("Inner Function: B-spline Basis Expansion (ψ)")
for i in range(B.shape[1]):
    plt.plot(x.numpy(), B[:, i], label=f"B[{i}]")
plt.xlabel("x")
plt.ylabel("Basis Value")
plt.legend()
plt.grid()
plt.show()


# ===========================================================
# 5. Plot the Outer Functions (Hidden Nodes Φ)
# ===========================================================
plt.figure(figsize=(10, 5))
plt.title("Outer Functions (Hidden Layer Outputs Φ)")
for i, hidden in enumerate(model.hidden_nodes):
    y_hidden = hidden(x).detach().numpy()
    plt.plot(x.numpy(), y_hidden, label=f"Φ{i+1}(ψ{i+1}(x))")
plt.xlabel("x")
plt.ylabel("Hidden Node Output")
plt.legend()
plt.grid()
plt.show()


# ===========================================================
# 6. Plot Learned Function vs True sin(x)
# ===========================================================
y_hat = model(x).detach().numpy()

plt.figure(figsize=(10, 5))
plt.title("Learned Function vs True sin(x)")
plt.plot(x.numpy(), y.numpy(), label="True sin(x)", linewidth=2)
plt.plot(x.numpy(), y_hat, label="KAN Approximation",
         linestyle='--', linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
