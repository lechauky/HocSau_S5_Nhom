# House Prices – Advanced Regression Techniques

Đồ án nhóm 4 người — dự đoán giá nhà (Kaggle, bộ dữ liệu Ames của De Cock 2011).
Tổ chức thư mục theo mẫu của giảng viên: notebook đánh số theo **stage**, kết quả trung gian lưu trong `exps/`, code dùng chung trong `extra/`, bảng tổng hợp `experiments.xlsx`.

## 1. Cấu trúc thư mục
```
house_prices_project/
├── data/                     # dữ liệu gốc từ Kaggle
├── doc/                      # báo cáo
├── extra/                    # code DÙNG CHUNG (import trong mọi notebook)
│   ├── config.py             #   đường dẫn, SEED, số fold
│   ├── common.py             #   import chung
│   ├── utils.py              #   đọc/ghi feature set, CV, ghi log, tạo file nộp
│   └── models.py             #   mô hình sklearn baseline + MLP PyTorch
├── prj/                      # NOTEBOOK — mỗi stage 1 thư mục
│   ├── 1.EDA/                #   EDA.ipynb (baseline), EDA1.ipynb (TV1)
│   ├── 2.pre-processing/     #   pre-processing.ipynb (baseline), pre-processing1 (TV1), pre-processing4 (TV4)
│   ├── 3.feature_selection/  #   feature_selection.ipynb (baseline), feature_selection4 (TV4)
│   ├── 4.model/              #   traditional model / neural network model (baseline), ...2 (TV2), ...3 (TV3)
│   └── 5.improve/            #   improve.ipynb: blend + gộp log -> experiments.xlsx
├── exps/                     # KẾT QUẢ TRUNG GIAN giữa các stage (sinh ra khi chạy notebook)
│   ├── data/                 #   output của 1.EDA: data_EDA.csv, columns_dtype.json
│   ├── feature0/             #   output của pre-processing baseline: x_train, y_train, x_test, preds/, submission_*.csv
│   ├── feature1/ feature4/…  #   output của notebook pre-processing<số>
│   └── logs/                 #   experiments_<thành viên>.csv (mỗi người 1 file)
├── test/                     # nháp, thử code nhanh
└── experiments.xlsx          # bảng tổng hợp mọi thực nghiệm (5.improve sinh ra)
```

**Luồng dữ liệu giữa các stage**
```
data/*.csv ─► 1.EDA ─► exps/data/ ─► 2.pre-processing<k> ─► exps/feature<k>/ ─► 3.feature_selection ─► selected_features.json
                                                                   │                                         │
                                                                   └──────────────► 4.model ◄────────────────┘
                                                                                  │ preds/*.npy, logs
                                                                                  ▼
                                                                         5.improve ─► submission_blend.csv, experiments.xlsx
```
Mỗi stage chỉ đọc file của stage trước trong `exps/` → **ai cũng có thể làm stage của mình ngay**, dùng output của baseline (`feature0`), không phải chờ người làm stage trước.

## 2. Bắt đầu (mỗi thành viên, khoảng 5 phút)
1. `pip install pandas numpy scikit-learn matplotlib seaborn openpyxl torch jupyter`
2. Tải `train.csv`, `test.csv`, `data_description.txt` từ Kaggle vào `data/`.
3. Chạy **lần lượt** các notebook baseline (Run All) để sinh `exps/feature0/`:
   `1.EDA/EDA` → `2.pre-processing/pre-processing` → `3.feature_selection/feature_selection` → `4.model/traditional model` → `4.model/neural network model` → `5.improve/improve`
4. Mở notebook mang **số của mình** và đọc khối **VIỆC CỦA TVx** ở đầu.

## 3. Phân công — làm song song cùng lúc
| | Notebook của mình | Đọc từ | Ghi ra | Việc chính |
|---|---|---|---|---|
| **TV1** | `1.EDA/EDA1`, `2.pre-processing/pre-processing1` | `data/`, `exps/data/` | `exps/feature1/` | Đọc paper; EDA cho báo cáo; xử lý NaN đúng nghĩa, loại ngoại lai, mã hóa thứ bậc |
| **TV2** | `4.model/traditional model2` | `exps/feature0/` | logs, `preds/` | Thêm mô hình sklearn, tinh chỉnh tham số, bảng so sánh |
| **TV3** | `4.model/neural network model3` | `exps/feature0/` | logs, `preds/` | Cải tiến MLP PyTorch: early stopping, dropout, weight decay, kiến trúc |
| **TV4** | `2.pre-processing/pre-processing4`, `3.feature_selection/feature_selection4` | `exps/data/`, `exps/feature4/` | `exps/feature4/` | Feature engineering + ablation, lựa chọn đặc trưng; ghép báo cáo |

### Tiến độ (hạn nộp: **23:59 thứ Tư 7/10** — Classroom không nhận bài trễ; hạn nội bộ: **tối thứ Ba 6/10**)

| Ngày | Cả nhóm | TV1 | TV2 | TV3 | TV4 |
|---|---|---|---|---|---|
| **T6 2/10** | Join Kaggle, tải dữ liệu, chạy 6 notebook baseline (mục 2). Gửi họ tên + MSSV cho nhóm trưởng | Đọc paper De Cock | Đọc `traditional model2` | Đọc `neural network model3` | Đọc `pre-processing4` |
| **T7 3/10 – CN 4/10** | **Làm song song** | `EDA1`; `pre-processing1`: NaN đúng nghĩa, loại ngoại lai, mã hóa thứ bậc → `feature1` | Thêm mô hình, GridSearch trên `feature0` | Early stopping, dropout, weight decay, kiến trúc; learning curve | `pre-processing4`: các nhóm đặc trưng, ablation, giảm skew → `feature4`; `feature_selection4` |
| **T2 5/10** | **Ghép** (tối: chốt kết quả) | Cùng TV4 tạo `pre-processing5` = làm sạch TV1 + FE tốt nhất TV4 → `feature5` | Chạy lại cấu hình tốt nhất với `FEATURE_SET="feature5"` | Chạy lại cấu hình tốt nhất với `feature5` | `5.improve` trên `feature5`: blend, `experiments.xlsx`; **nộp Kaggle**, điền `kaggle_score` |
| **T3 6/10** | **Báo cáo** (tối: ghép xong) | Chương 1–3 | Mục 4.1 | Mục 4.2 | Chương 5–7, ghép báo cáo, trang bìa, bảng phân công |
| **T4 7/10** | **Dự phòng** — rà soát, nhóm trưởng nộp **trước buổi tối** | | | | |

Lưu ý:
- Giai đoạn song song: TV2, TV3 dùng `feature0`; TV1, TV4 tự đo feature set của mình bằng `4.model/traditional model` (đổi `FEATURE_SET`). Cải tiến mô hình trên `feature0` vẫn có giá trị vì mọi người so trên cùng một dữ liệu.
- Kaggle giới hạn số lần nộp mỗi ngày → chỉ nộp các mô hình cuối vào thứ Hai, không dồn sang thứ Tư.
- MLP chậm nếu máy không có GPU → chạy notebook trên Kaggle/Colab (`config.py` tự nhận đường dẫn dữ liệu trên Kaggle).
- **Không Clear Output** notebook trước khi nộp: đề yêu cầu nộp source code kèm quá trình chạy.

### Nộp bài (nhóm trưởng)
Nén thành **`lab03_house_price_hoten_masv.zip`** (họ tên, MSSV của nhóm trưởng), gồm: `prj/` (còn output), `extra/`, `exps/logs/`, `experiments.xlsx`, `README.md`, báo cáo Word trong `doc/` (trang bìa có họ tên + MSSV các thành viên, bảng phân công) và các file `submission_*.csv` cuối. Không kèm dữ liệu Kaggle trong `data/`.

## 4. Quy ước (bắt buộc)
- **Đặt tên notebook:** `<tên stage><số thành viên>.ipynb`; thêm phiên bản bằng `_1, _2`: `pre-processing4_1.ipynb`. Feature set tương ứng: `feature4_1`.
- **Không sửa notebook baseline và notebook của người khác.** Muốn thử thì copy sang notebook mang số của mình.
- **`extra/` là code chung:** chỉ sửa khi cả nhóm thống nhất (sửa sai ảnh hưởng tất cả). Code mới viết trong notebook trước, ổn rồi mới chuyển vào `extra/`.
- **Đánh giá chung:** target `log1p(SalePrice)`, metric RMSE, `get_kfold()` 5-fold seed 42. Không dùng `train_test_split` riêng.
- **Không tính thống kê trên test hay trên cả train trước khi CV** (mean, scaler...) — `make_pipeline()` đã làm trong từng fold.
- **Log:** mỗi lần chạy gọi `log_experiment(MEMBER, ...)` → `exps/logs/experiments_<MEMBER>.csv`. Không ghi đè file log của người khác; `export_experiments()` gộp thành `experiments.xlsx`.
- **Git:** commit `prj/`, `extra/`, `doc/`, `exps/logs/`; không commit `data/` và file lớn trong `exps/feature*/` (đã có `.gitignore`).

## 5. Kết quả baseline (`feature0`, CV RMSE 5-fold)
| Mô hình | CV RMSE |
|---|---|
| Linear Regression | 0.1488 |
| Ridge | 0.1472 |
| Lasso | 0.1443 |
| Random Forest | 0.1441 |
| Gradient Boosting | 0.1302 |
| MLP 256-128 (PyTorch) | 0.1509 |
| Blend (5.improve) | 0.1265 |

