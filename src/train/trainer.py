"""
Training loop with AdamW + ReduceLROnPlateau + Huber loss + early stopping.
Targets are expected to be log1p-then-StandardScaler scaled; the loop inverts
that transform when computing reportable val metrics.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error

from src.config import EPOCHS, LR, PATIENCE
from src.utils import r2_safe


def train_model(model, train_loader, val_loader, device, scaler_y_mean, scaler_y_scale):
    opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    loss_fn = nn.HuberLoss(delta=1.0)

    best_val     = float("inf")
    patience_ctr = 0
    best_state   = None

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_mae": [], "val_r2": []}

    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'Val MAE':>10}  {'Val R2':>8}")
    print("-" * 58)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_loss += loss.item() * len(yb)
        t_loss /= len(train_loader.dataset)

        model.eval()
        v_loss, preds_s, actuals_s = 0.0, [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                p = model(Xb)
                v_loss += loss_fn(p, yb).item() * len(yb)
                preds_s.extend(p.cpu().numpy())
                actuals_s.extend(yb.cpu().numpy())
        v_loss /= len(val_loader.dataset)

        preds   = np.expm1(np.asarray(preds_s)  * scaler_y_scale + scaler_y_mean)
        actuals = np.expm1(np.asarray(actuals_s) * scaler_y_scale + scaler_y_mean)

        v_mae = mean_absolute_error(actuals, preds)
        v_r2  = r2_safe(actuals, preds)

        sched.step(v_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(t_loss))
        history["val_loss"].append(float(v_loss))
        history["val_mae"].append(float(v_mae))
        history["val_r2"].append(float(v_r2) if np.isfinite(v_r2) else float("nan"))

        print(f"{epoch:>6}  {t_loss:>12.4f}  {v_loss:>10.4f}  {v_mae:>10.4f}  {v_r2:>8.4f}")

        if v_loss < best_val - 1e-5:
            best_val     = v_loss
            patience_ctr = 0
            best_state   = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
