from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from ..metrics import classification_metrics


class TSFAM(nn.Module):
    """Time–Sensor Feature Attention Module.

    This implementation follows the manuscript architecture: depthwise 3×1 temporal
    and 1×3 sensor convolutions, average/max pooling descriptors, and adaptive
    fusion of temporal and sensor branches.
    """
    def __init__(self, channels):
        super().__init__()
        self.time_conv = nn.Conv2d(channels, channels, kernel_size=(3,1),
                                   padding=(1,0), groups=channels, bias=False)
        self.sensor_conv = nn.Conv2d(channels, channels, kernel_size=(1,3),
                                     padding=(0,1), groups=channels, bias=False)
        self.branch_gate = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _avg_max_gate(z):
        gap = z.mean(dim=(2,3), keepdim=True)
        gmp = z.amax(dim=(2,3), keepdim=True)
        return torch.sigmoid(gap + gmp)

    def forward(self, x):
        xt = self.time_conv(x)
        xs = self.sensor_conv(x)
        xt = xt * self._avg_max_gate(xt)
        xs = xs * self._avg_max_gate(xs)
        pt = xt.mean(dim=(2,3), keepdim=True).expand_as(xt)
        ps = xs.mean(dim=(2,3), keepdim=True).expand_as(xs)
        a = self.branch_gate(torch.cat([pt, ps], dim=1))
        return a*xt + (1.0-a)*xs


class TSFCNet(nn.Module):
    def __init__(self, window=60, n_sensors=16, n_classes=3, deep_channels=20,
                 dropout=0.20):
        super().__init__()
        self.window = window
        self.n_sensors = n_sensors
        self.expand = nn.Conv2d(1, deep_channels, kernel_size=1)
        self.tsfam = TSFAM(deep_channels)
        self.compress = nn.Conv2d(deep_channels, 1, kernel_size=1)
        self.pool = nn.AvgPool2d(2, 2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear((window//2)*(n_sensors//2), 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.expand(x.unsqueeze(1))
        x = self.tsfam(x)
        x = self.compress(x)
        x = self.pool(x)
        return self.head(x)


class TSFCAblationNet(nn.Module):
    """Minimal temporal/sensor branch ablation model."""
    def __init__(self, window=60, n_sensors=16, n_classes=3, deep_channels=20,
                 mode="both", pool_mode="avgmax"):
        super().__init__()
        if mode not in {"baseline", "time", "sensor", "both"}:
            raise ValueError(mode)
        if pool_mode not in {"none", "avg", "max", "avgmax"}:
            raise ValueError(pool_mode)
        self.mode, self.pool_mode = mode, pool_mode
        self.expand = nn.Conv2d(1, deep_channels, 1)
        self.time_conv = nn.Conv2d(deep_channels, deep_channels, (3,1), padding=(1,0), groups=deep_channels)
        self.sensor_conv = nn.Conv2d(deep_channels, deep_channels, (1,3), padding=(0,1), groups=deep_channels)
        self.compress = nn.Conv2d(deep_channels, 1, 1)
        self.pool = nn.AvgPool2d(2,2)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear((window//2)*(n_sensors//2),64),
                                nn.ReLU(), nn.Linear(64,n_classes))

    def _attend(self, z):
        if self.pool_mode == "none": return z
        terms=[]
        if self.pool_mode in {"avg","avgmax"}: terms.append(z.mean((2,3),keepdim=True))
        if self.pool_mode in {"max","avgmax"}: terms.append(z.amax((2,3),keepdim=True))
        gate = torch.sigmoid(sum(terms))
        return z*gate

    def forward(self, x):
        z = self.expand(x.unsqueeze(1))
        if self.mode == "baseline":
            h = z
        elif self.mode == "time":
            h = self._attend(self.time_conv(z))
        elif self.mode == "sensor":
            h = self._attend(self.sensor_conv(z))
        else:
            h = 0.5*(self._attend(self.time_conv(z)) + self._attend(self.sensor_conv(z)))
        return self.fc(self.pool(self.compress(h)))


def train_torch_classifier(model, X, y, test_size=0.30, seed=20260913,
                           epochs=200, batch_size=32, learning_rate=1e-3,
                           optimizer_name="adam"):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    mu = X[tr].mean(axis=(0,1), keepdims=True)
    sd = X[tr].std(axis=(0,1), keepdims=True) + 1e-8
    Xn = (X-mu)/sd
    train_ds = TensorDataset(torch.tensor(Xn[tr], dtype=torch.float32), torch.tensor(y[tr], dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(Xn[te], dtype=torch.float32), torch.tensor(y[te], dtype=torch.long))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if optimizer_name.lower() == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("optimizer_name must be 'adam' or 'adamw'")
    loss_fn = nn.CrossEntropyLoss()
    history=[]
    for _ in range(int(epochs)):
        model.train(); losses=[]
        for xb,yb in train_dl:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        history.append(float(np.mean(losses)))
    model.eval(); yp=[]; yt=[]
    with torch.no_grad():
        for xb,yb in test_dl:
            yp.extend(model(xb.to(device)).argmax(1).cpu().numpy())
            yt.extend(yb.numpy())
    return model, history, classification_metrics(yt,yp), (np.asarray(yt),np.asarray(yp)), (mu,sd), (tr,te)
