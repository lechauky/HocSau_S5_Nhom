# models.py — định nghĩa mô hình dùng chung (bản BASELINE)
# Thành viên cải tiến mô hình trong notebook của mình trước; khi đã chốt mới đưa vào file này.
import copy

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from config import SEED
from utils import get_kfold, rmse


# ======================================================================== ML truyền thống
def get_traditional_models():
    """Bộ mô hình baseline, tham số mặc định/đơn giản. TV2 thêm mô hình & tinh chỉnh."""
    return {
        "LR": LinearRegression(),
        "Ridge": Ridge(alpha=10.0),
        "Lasso": Lasso(alpha=0.0005, max_iter=50000),
        "RF": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED),
        "GBR": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                         max_depth=3, random_state=SEED),
    }


# ======================================================================== MLP (PyTorch)
try:
    import torch
    import torch.nn as nn
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    class MLP(nn.Module):
        """MLP baseline: Linear -> ReLU lặp lại, chưa có regularization."""

        def __init__(self, in_dim, hidden=(256, 128), dropout=0.0):
            super().__init__()
            layers, d = [], in_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU()]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                d = h
            layers.append(nn.Linear(d, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def _t(a):
        return torch.tensor(np.asarray(a, dtype=np.float32), device=DEVICE)

    def train_mlp(x_tr, y_tr, x_va, y_va, hidden=(256, 128), dropout=0.0, lr=1e-3,
                  weight_decay=0.0, batch_size=64, epochs=100, patience=None):
        """
        Huấn luyện 1 mô hình. patience=None -> chạy đủ `epochs` (baseline);
        patience=k -> early stopping, giữ trọng số có val loss nhỏ nhất.
        Trả về (model, history) với history = {"train": [...], "val": [...]}.
        """
        model = MLP(x_tr.shape[1], hidden, dropout).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        loader = DataLoader(TensorDataset(_t(x_tr), _t(y_tr)), batch_size=batch_size, shuffle=True)
        xv, yv = _t(x_va), _t(y_va)
        hist, best, best_state, bad = {"train": [], "val": []}, np.inf, None, 0
        for _ in range(epochs):
            model.train()
            tot = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            model.eval()
            with torch.no_grad():
                val = loss_fn(model(xv), yv).item()
            hist["train"].append(tot / len(x_tr))
            hist["val"].append(val)
            if patience is not None:
                if val < best - 1e-6:
                    best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, hist

    def predict_mlp(model, x):
        model.eval()
        with torch.no_grad():
            return model(_t(x)).cpu().numpy()

    def cv_mlp(x, y, x_test, **cfg):
        """
        5-fold CV cho MLP, dùng chung get_kfold() với mô hình sklearn.
        Mỗi fold: fit imputer + scaler trên fold train; chuẩn hóa target; dự đoán test lấy trung bình.
        Trả về (scores, oof, test_pred, histories).
        """
        x, x_test = np.asarray(x, dtype=float), np.asarray(x_test, dtype=float)
        oof, test_pred, scores, hists = np.zeros(len(y)), np.zeros(len(x_test)), [], []
        kf = get_kfold()
        for tr, va in kf.split(x):
            imp, sc = SimpleImputer(strategy="median"), StandardScaler()
            x_tr = sc.fit_transform(imp.fit_transform(x[tr]))
            x_va = sc.transform(imp.transform(x[va]))
            x_te = sc.transform(imp.transform(x_test))
            mu, sd = y[tr].mean(), y[tr].std()
            model, h = train_mlp(x_tr, (y[tr] - mu) / sd, x_va, (y[va] - mu) / sd, **cfg)
            oof[va] = predict_mlp(model, x_va) * sd + mu
            test_pred += (predict_mlp(model, x_te) * sd + mu) / kf.get_n_splits()
            scores.append(rmse(y[va], oof[va]))
            hists.append(h)
        return np.array(scores), oof, test_pred, hists

except Exception as e:   # không có / lỗi PyTorch: vẫn dùng được phần sklearn
    print(f"[models] Không tải được PyTorch ({type(e).__name__}) -> chỉ dùng được mô hình sklearn.")
    