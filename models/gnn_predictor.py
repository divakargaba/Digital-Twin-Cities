import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import mean_squared_error, r2_score

class GNNCityPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNCityPredictor, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def evaluate_gnn(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index).squeeze()
        y_true = data.y.cpu().numpy()
        y_pred = out.cpu().numpy()

        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_true, y_pred)

        print(f"GNN Evaluation:\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}")

def train_gnn(model, data, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index).squeeze()
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Final evaluation
    evaluate_gnn(model, data)
