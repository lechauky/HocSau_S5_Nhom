# utils.py — hàm tiện ích dùng chung: đọc/ghi feature set, đánh giá CV, ghi log, tạo file nộp
import glob
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (N_SPLITS, SEED, data_dir, experiments_xlsx, exps_data, feature_dir, logs_dir)


# ======================================================================== seed / CV / metric
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    # Chỉ seed PyTorch khi notebook đã dùng tới nó (notebook MLP import models.py trước).
    # Không tự import torch ở đây -> notebook không dùng MLP không bị lỗi khi torch có vấn đề.
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "manual_seed"):
        torch.manual_seed(seed)


def get_kfold():
    """MỌI mô hình phải dùng đúng cách chia này để điểm so sánh được với nhau."""
    return KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def rmse(y_true, y_pred):
    """y ở thang log1p(SalePrice) -> đúng metric của Kaggle."""
    return float(np.sqrt(np.mean((np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()) ** 2)))


# ======================================================================== dữ liệu
def load_raw():
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train, test


def load_eda():
    """Output của 1.EDA: train + test gộp (cột is_train), kèm danh sách cột số/phân loại."""
    df = pd.read_csv(os.path.join(exps_data, "data_EDA.csv"))
    with open(os.path.join(exps_data, "columns_dtype.json"), encoding="utf-8") as f:
        cols = json.load(f)
    return df, cols["numeric_columns"], cols["category_columns"]


def save_feature_set(feature_set, x_train, y_train, x_test, test_id, info=None):
    """Lưu output của 2.pre-processing vào exps/<feature_set>/."""
    d = feature_dir(feature_set)
    x_train.to_csv(f"{d}/x_train.csv", index=False)
    pd.DataFrame({"y": y_train}).to_csv(f"{d}/y_train.csv", index=False)
    x_test.to_csv(f"{d}/x_test.csv", index=False)
    pd.DataFrame({"Id": test_id}).to_csv(f"{d}/test_id.csv", index=False)
    meta = {"created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "n_train": len(x_train), "n_features": x_train.shape[1], **(info or {})}
    with open(f"{d}/info.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[save] {d}: x_train {x_train.shape}, x_test {x_test.shape}")


def load_feature_set(feature_set, selection=None):
    """
    Đọc exps/<feature_set>/. Nếu selection != None, chỉ giữ các cột trong
    exps/<feature_set>/selected_features.json[selection] (output của 3.feature_selection).
    """
    d = feature_dir(feature_set)
    x_train = pd.read_csv(f"{d}/x_train.csv")
    y_train = pd.read_csv(f"{d}/y_train.csv")["y"].values
    x_test = pd.read_csv(f"{d}/x_test.csv")
    test_id = pd.read_csv(f"{d}/test_id.csv")["Id"].values
    if selection:
        with open(f"{d}/selected_features.json", encoding="utf-8") as f:
            cols = json.load(f)[selection]
        x_train, x_test = x_train[cols], x_test[cols]
    print(f"[load] {feature_set}{'/' + selection if selection else ''}: "
          f"x_train {x_train.shape}, x_test {x_test.shape}")
    return x_train, y_train, x_test, test_id


# ======================================================================== đánh giá
def make_pipeline(model, scale=True):
    """
    Bước cuối trước mô hình: điền NaN còn lại (median) + chuẩn hóa.
    Đặt trong Pipeline để chỉ fit trên fold train -> không rò rỉ dữ liệu khi CV.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def evaluate(model, x, y, scale=True):
    """5-fold CV. Trả về (RMSE từng fold, dự đoán out-of-fold)."""
    pipe = make_pipeline(model, scale)
    oof = cross_val_predict(pipe, x, y, cv=get_kfold())
    scores = np.array([rmse(y[va], oof[va]) for _, va in get_kfold().split(x)])
    return scores, oof


def fit_predict(model, x, y, x_test, scale=True):
    """Fit trên toàn bộ train, dự đoán test (thang log)."""
    return make_pipeline(model, scale).fit(x, y).predict(x_test)


# ======================================================================== log thực nghiệm
LOG_FIELDS = ["time", "member", "notebook", "feature_set", "selection", "model",
              "params", "cv_rmse", "cv_std", "kaggle_score", "note"]


def log_experiment(member, notebook, feature_set, model, scores, params=None,
                   selection="all", note=""):
    """Mỗi người ghi file log riêng exps/logs/experiments_<member>.csv -> không đụng nhau khi làm song song."""
    path = os.path.join(logs_dir, f"experiments_{member}.csv")
    row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "member": member,
           "notebook": notebook, "feature_set": feature_set, "selection": selection or "all",
           "model": model, "params": str(params or ""), "cv_rmse": round(float(np.mean(scores)), 5),
           "cv_std": round(float(np.std(scores)), 5), "kaggle_score": "", "note": note}
    pd.DataFrame([row], columns=LOG_FIELDS).to_csv(path, mode="a", index=False,
                                                    header=not os.path.exists(path))
    print(f"[log] {model:12s} | {feature_set}/{row['selection']} | "
          f"CV RMSE = {row['cv_rmse']:.5f} ± {row['cv_std']:.5f}")


def export_experiments():
    """Gộp log của mọi người thành experiments.xlsx (sắp theo CV RMSE)."""
    files = glob.glob(os.path.join(logs_dir, "experiments_*.csv"))
    if not files:
        print("Chưa có log nào.")
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # Chạy lại notebook nhiều lần -> nhiều dòng trùng. Giữ lần chạy MỚI NHẤT của mỗi thực nghiệm
    # (cùng notebook + feature_set + selection + model), nhưng giữ dòng cũ nếu nó có kaggle_score.
    key = ["member", "notebook", "feature_set", "selection", "model"]
    df["_has_kaggle"] = df["kaggle_score"].notna()
    df = (df.sort_values(["_has_kaggle", "time"])
            .drop_duplicates(subset=key, keep="last")
            .drop(columns="_has_kaggle"))
    df = df.sort_values("cv_rmse").reset_index(drop=True)
    df.insert(0, "No", range(1, len(df) + 1))
    df.to_excel(experiments_xlsx, index=False)
    print(f"[export] {len(df)} thực nghiệm -> {experiments_xlsx}")
    return df


# ======================================================================== lưu dự đoán / file nộp
def save_preds(feature_set, name, oof, test_pred):
    """Lưu dự đoán OOF + test (thang log) vào exps/<fs>/preds/ để 5.improve blend."""
    d = os.path.join(feature_dir(feature_set), "preds")
    np.save(f"{d}/{name}_oof.npy", np.asarray(oof))
    np.save(f"{d}/{name}_test.npy", np.asarray(test_pred))


def save_submission(feature_set, name, test_id, test_pred_log):
    path = os.path.join(feature_dir(feature_set), f"submission_{name}.csv")
    pd.DataFrame({"Id": test_id, "SalePrice": np.expm1(test_pred_log)}).to_csv(path, index=False)
    print(f"[submit] {path}")
    return path
