import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# TODO: degree and no of knots fine tuning

# ----------------------------
# Simplified B-Spline Activation
# ----------------------------


class SimpleBSplineActivation(nn.Module):
    def __init__(self, num_knots=5, degree=2):
        super(SimpleBSplineActivation, self).__init__()
        knots = np.linspace(-2, 2, num_knots)
        self.knots = torch.tensor(knots, dtype=torch.float32)
        self.degree = degree
        self.weights = nn.Parameter(torch.randn(num_knots - degree - 1))

    def forward(self, x):
        batch_size = x.shape[0]
        outputs = torch.zeros((batch_size, 1), device=x.device)

        basis_values = self.b_spline(x[:, 0])   # (batch_size, num_basis)
        # outputs[:, 0] = torch.matmul(basis_values, self.weights)
        outputs[:, 0] = basis_values @ self.weights

        return outputs

    def b_spline(self, x):
        n = len(self.knots) - 1
        B = torch.zeros((x.shape[0], n - self.degree),
                        dtype=torch.float32, device=x.device)

        for j in range(n - self.degree):
            B[:, j] = self.bspline_basis(j, self.degree, x, self.knots)

        return B

    def bspline_basis(self, j, k, x, knots):
        if k == 0:
            return ((knots[j] <= x) & (x < knots[j+1])).float()

        denom1 = knots[j+k] - knots[j]
        denom2 = knots[j+k+1] - knots[j+1]

        term1 = ((x - knots[j]) / denom1) * self.bspline_basis(j,
                                                               k-1, x, knots) if denom1 != 0 else 0
        term2 = ((knots[j+k+1] - x) / denom2) * \
            self.bspline_basis(j+1, k-1, x, knots) if denom2 != 0 else 0

        return term1 + term2


# ----------------------------
# Minimal KAN
# ----------------------------
class SimpleKAN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=2, output_dim=1, num_knots=5):
        super(SimpleKAN, self).__init__()
        # 2 B-spline activations (like hidden units)
        self.hidden_layers = nn.ModuleList(
            [SimpleBSplineActivation(num_knots=num_knots, degree=2)
             for _ in range(hidden_dim)]
        )
        # Linear layer combines them
        self.output_layer = nn.Linear(hidden_dim * input_dim, output_dim)

    def forward(self, x):
        h = torch.cat([layer(x) for layer in self.hidden_layers],
                      dim=1)  # concat outputs
        return self.output_layer(h)


# ----------------------------
# Training Loop
# ----------------------------
def train_simple_kan():
    # Dataset: y = sin(x)
    x_vals = torch.linspace(-2, 2, 100).unsqueeze(1)  # (100,1)
    y_vals = torch.sin(x_vals)

    model = SimpleKAN(input_dim=1, hidden_dim=2, output_dim=1, num_knots=5)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    losses = []

    for epoch in range(500):
        optimizer.zero_grad()
        y_pred = model(x_vals)
        loss = loss_fn(y_pred, y_vals)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Plot loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.show()

    # Plot prediction vs true sin(x)
    with torch.no_grad():
        y_pred = model(x_vals)

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals.numpy(), y_vals.numpy(), label="True sin(x)")
    plt.plot(x_vals.numpy(), y_pred.numpy(),
             label="KAN approx", linestyle="--")
    plt.legend()
    plt.title("Minimal KAN Approximation")
    plt.show()


# Run training
train_simple_kan()
