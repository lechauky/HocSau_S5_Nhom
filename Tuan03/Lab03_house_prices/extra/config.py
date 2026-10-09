# config.py — đường dẫn & tham số DÙNG CHUNG (không tự ý sửa SEED / N_SPLITS)
# Đầu mỗi notebook trong prj/<stage>/:
#     import sys; sys.path.insert(0, "../../extra")
#     from config import *
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")).replace("\\", "/")
extra_dir = os.path.join(root_dir, "extra").replace("\\", "/")
if extra_dir not in sys.path:
    sys.path.insert(0, extra_dir)

# Dữ liệu gốc tải từ Kaggle: train.csv, test.csv, data_description.txt
KAGGLE_DIR = "/kaggle/input/house-prices-advanced-regression-techniques"
data_dir = KAGGLE_DIR if os.path.exists(KAGGLE_DIR) else os.path.join(root_dir, "data").replace("\\", "/")

# Kết quả trung gian giữa các stage
exps_dir = os.path.join(root_dir, "exps").replace("\\", "/")
exps_data = os.path.join(exps_dir, "data").replace("\\", "/")   # output của 1.EDA
logs_dir = os.path.join(exps_dir, "logs").replace("\\", "/")    # log thực nghiệm của từng người
for _d in (exps_dir, exps_data, logs_dir):
    os.makedirs(_d, exist_ok=True)

# Bảng tổng hợp thực nghiệm (giống experiments.xlsx của project mẫu)
experiments_xlsx = os.path.join(root_dir, "experiments.xlsx").replace("\\", "/")

# Đánh giá chung: 5-fold CV, target = log1p(SalePrice), metric = RMSE (đúng metric Kaggle)
SEED = 42
N_SPLITS = 5


def feature_dir(feature_set):
    """exps/<feature_set>/ — nơi lưu x_train, y_train, x_test của một phiên bản tiền xử lý."""
    d = os.path.join(exps_dir, feature_set).replace("\\", "/")
    os.makedirs(os.path.join(d, "preds"), exist_ok=True)
    return d
