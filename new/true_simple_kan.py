# =============================================
# Kolmogorov-Arnold Network (True 1D version)
# =============================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ---------- B-spline Basis Function Layer ----------
class BSplineBasis(nn.Module):
    def __init__(self, num_knots=15, degree=3):
        super().__init__()
        self.degree = degree
        self.knots = torch.linspace(-np.pi, np.pi, num_knots)
        self.num_basis = num_knots - degree - 1

    def bspline_basis(self, j, k, x, knots):
        """Cox-de Boor recursive formula"""
        if k == 0:
            return ((knots[j] <= x) & (x < knots[j + 1])).float()

        denom1 = knots[j + k] - knots[j]
        denom2 = knots[j + k + 1] - knots[j + 1]

        term1 = 0
        term2 = 0
        if denom1 > 0:
            term1 = (x - knots[j]) / denom1 * \
                self.bspline_basis(j, k - 1, x, knots)
        if denom2 > 0:
            term2 = (knots[j + k + 1] - x) / denom2 * \
                self.bspline_basis(j + 1, k - 1, x, knots)

        return term1 + term2

    def forward(self, x):
        """Compute full basis matrix B for all inputs"""
        B = torch.zeros(x.shape[0], self.num_basis)
        for j in range(self.num_basis):
            B[:, j] = self.bspline_basis(
                j, self.degree, x.squeeze(), self.knots)
        return B


# ---------- Kolmogorov–Arnold Node ----------
class KANNode(nn.Module):
    def __init__(self, num_knots=15, degree=3):
        super().__init__()
        self.basis = BSplineBasis(num_knots, degree)
        self.control_points = nn.Parameter(
            torch.randn(self.basis.num_basis, 1))

    def forward(self, x):
        B = self.basis(x)
        y = B @ self.control_points  # shape: [batch, 1]
        return y


# ---------- Complete 1D KAN ----------
class SimpleKAN1D(nn.Module):
    def __init__(self, num_nodes=3, num_knots=15, degree=3):
        super().__init__()
        self.nodes = nn.ModuleList(
            [KANNode(num_knots, degree) for _ in range(num_nodes)])
        # linear sum of node outputs
        self.output_layer = nn.Linear(num_nodes, 1, bias=False)

    def forward(self, x):
        hidden_outputs = [node(x) for node in self.nodes]  # list of [N, 1]
        h = torch.cat(hidden_outputs, dim=1)  # shape: [N, num_nodes]
        # return final output + intermediate hidden values
        return self.output_layer(h), h


# ---------- Create Training Data ----------
x = torch.linspace(-np.pi, np.pi, 200).unsqueeze(1)
y_true = torch.sin(x)

# ---------- Model, Optimizer, Loss ----------
model = SimpleKAN1D(num_nodes=3, num_knots=15, degree=3)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# ---------- Training Loop ----------
for epoch in range(2000):
    optimizer.zero_grad()
    y_pred, _ = model(x)
    loss = criterion(y_pred, y_true)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

# ---------- Visualization ----------
with torch.no_grad():
    y_pred, hidden_vals = model(x)

# Plot: Learned function vs True sin(x)
plt.figure(figsize=(10, 5))
plt.plot(x.numpy(), y_true.numpy(), label='True sin(x)', color='blue')
plt.plot(x.numpy(), y_pred.numpy(), label='KAN Approximation', color='red')
plt.title("Kolmogorov–Arnold Network Approximation (1D, 3 Hidden Nodes)")
plt.legend()
plt.grid(True)
plt.show()

# Plot: Each hidden node's learned function
plt.figure(figsize=(10, 6))
for i in range(hidden_vals.shape[1]):
    plt.plot(x.numpy(), hidden_vals[:, i].numpy(), label=f"Hidden Node {i+1}")
plt.title("Outputs of the 3 Hidden Nodes (Each has its own spline basis)")
plt.legend()
plt.grid(True)
plt.show()
