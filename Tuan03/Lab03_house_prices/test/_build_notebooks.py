"""Sinh các notebook baseline + notebook của từng thành viên (chạy 1 lần khi tạo khung)."""
import os

import nbformat as nbf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


SETUP = [
    md("BEGIN"),
    code("%reload_ext autoreload\n%autoreload 2"),
    code('import sys; sys.path.insert(0, "../../extra")\n'
         "from config import *\nfrom common import *\nfrom utils import *\ndisplay.clear_output()"),
]


def params(member, notebook, extra=""):
    return code(f'MEMBER   = "{member}"      # tên/mã thành viên -> file log riêng\n'
                f'NOTEBOOK = "{notebook}"\n' + extra + "\nseed_everything()")


def save(cells, path):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    full = os.path.join(ROOT, "prj", path)
    nbf.write(nb, full)
    print("wrote", path)


# ============================================================================= 1. EDA
def eda_cells(member, notebook, todo=None):
    c = [md(f"# 1. Khám phá dữ liệu (EDA)\n\n**Input:** `data/train.csv`, `data/test.csv`  \n"
            f"**Output:** `exps/data/data_EDA.csv`, `exps/data/columns_dtype.json` (dùng cho 2.pre-processing)"),
         *SETUP, params(member, notebook)]
    if todo:
        c.append(md(todo))
    c += [
        md("## 1.1. Đọc dữ liệu"),
        code("train, test = load_raw()\nprint('train:', train.shape, '| test:', test.shape)\ntrain.head()"),
        md("## 1.2. Biến mục tiêu SalePrice\n"
           "Giá lệch phải mạnh; Kaggle chấm RMSE trên **log(giá)** -> mọi mô hình học trên `log1p(SalePrice)`."),
        code("fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))\n"
             "sns.histplot(train.SalePrice, bins=50, ax=ax[0]).set_title(f'SalePrice, skew={train.SalePrice.skew():.2f}')\n"
             "sns.histplot(np.log1p(train.SalePrice), bins=50, ax=ax[1]).set_title("
             "f'log1p(SalePrice), skew={np.log1p(train.SalePrice).skew():.2f}')\nplt.show()"),
        md("## 1.3. Dữ liệu thiếu\n"
           "Đối chiếu `data/data_description.txt`: nhiều NaN mang nghĩa **“không có”** (không gara, không hầm...)."),
        code("full = pd.concat([train.drop(columns='SalePrice'), test])\n"
             "miss = (full.isna().mean() * 100).sort_values(ascending=False)\nmiss = miss[miss > 0]\n"
             "miss.plot.bar(figsize=(12, 3.5), ylabel='% thiếu'); plt.show()\nmiss.round(1).to_frame('% thiếu').T"),
        md("## 1.4. Tương quan với SalePrice"),
        code("corr = train.select_dtypes('number').corr()['SalePrice'].drop(['SalePrice', 'Id'])\n"
             "top = corr.abs().sort_values(ascending=False).head(12).index\n"
             "corr[top][::-1].plot.barh(figsize=(6, 4)); plt.show()"),
        md("## 1.5. Ngoại lai\nPaper De Cock (2011) khuyến nghị xem xét các căn `GrLivArea > 4000`."),
        code("out = train.GrLivArea > 4000\nplt.figure(figsize=(6, 4))\n"
             "plt.scatter(train.GrLivArea, train.SalePrice, s=6, alpha=.5)\n"
             "plt.scatter(train.GrLivArea[out], train.SalePrice[out], s=30, c='red')\n"
             "plt.xlabel('GrLivArea'); plt.ylabel('SalePrice'); plt.show()\n"
             "train.loc[out, ['GrLivArea', 'SalePrice', 'SaleCondition']]"),
        md("## 1.6. Lưu output cho stage sau"),
        code("df = pd.concat([train.assign(is_train=1), test.assign(is_train=0)], ignore_index=True)\n"
             "feat = [c for c in train.columns if c not in ('Id', 'SalePrice')]\n"
             "numeric_columns  = train[feat].select_dtypes('number').columns.tolist()\n"
             "category_columns = [c for c in feat if c not in numeric_columns]\n"
             "df.to_csv(f'{exps_data}/data_EDA.csv', index=False)\n"
             "with open(f'{exps_data}/columns_dtype.json', 'w') as f:\n"
             "    json.dump({'numeric_columns': numeric_columns, 'category_columns': category_columns}, f, indent=1)\n"
             "print(len(numeric_columns), 'cột số |', len(category_columns), 'cột phân loại ->', exps_data)"),
        md("### Kết luận\n- *(ghi nhận xét của nhóm ở đây)*"),
    ]
    return c


# ============================================================================= 2. PRE-PROCESSING
def prep_cells(member, notebook, feature_set, todo=None, fe_stub=False):
    c = [md(f"# 2. Tiền xử lý dữ liệu\n\n**Input:** `exps/data/data_EDA.csv`  \n"
            f"**Output:** `exps/{feature_set}/` gồm `x_train.csv, y_train.csv, x_test.csv, test_id.csv, info.json`"),
         *SETUP, params(member, notebook, f'FEATURE_SET = "{feature_set}"   # thư mục output trong exps/')]
    if todo:
        c.append(md(todo))
    c += [
        code("df, numeric_columns, category_columns = load_eda()\ndf.shape"),
        md("## 2.1. Làm sạch\n**Baseline = cách của code tham khảo:** bỏ 4 cột thiếu nhiều, "
           "cột phân loại điền mode. Cột số để NaN — `make_pipeline()` ở stage 4 sẽ điền median **trong từng fold**."),
        code("DROP_COLS = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']\n"
             "df = df.drop(columns=DROP_COLS)\n"
             "cat_cols = [c for c in category_columns if c not in DROP_COLS] + ['MSSubClass']\n"
             "df['MSSubClass'] = df['MSSubClass'].astype(str)    # mã loại nhà, không phải số\n"
             "for c in cat_cols:\n"
             "    df[c] = df[c].fillna(df.loc[df.is_train == 1, c].mode()[0])   # mode tính trên train\n"
             "df[cat_cols].isna().sum().sum()"),
        md("## 2.2. Ngoại lai\nBaseline: **chưa** loại."),
        code("outlier_mask = pd.Series(False, index=df.index)   # TODO: GrLivArea > 4000 (chỉ dòng train)\n"
             "df = df[~outlier_mask].reset_index(drop=True)\nprint('Số dòng train:', df.is_train.sum())"),
    ]
    if fe_stub:
        c += [md("## 2.3. Feature engineering (TV4)\nMỗi hàm chỉ dùng thông tin **trong cùng 1 dòng**. "
                 "Bật/tắt từng nhóm để làm ablation."),
              code("def total_sf(d):\n    d['TotalSF'] = d['TotalBsmtSF'] + d['1stFlrSF'] + d['2ndFlrSF']\n    return d\n\n"
                   "# TODO: baths, age, flags, interactions, ...\n"
                   "FE_GROUPS = {'total_sf': total_sf}\nUSE = ['total_sf']      # nhóm đang bật\n"
                   "for g in USE:\n    df = FE_GROUPS[g](df)")]
    c += [
        md("## 2.4. Mã hóa & target\nOne-hot trên train+test gộp (như code tham khảo) để hai tập có cùng cột."),
        code("y_train  = np.log1p(df.loc[df.is_train == 1, 'SalePrice'].values)\n"
             "test_id  = df.loc[df.is_train == 0, 'Id'].values\n"
             "X = pd.get_dummies(df.drop(columns=['Id', 'SalePrice']), columns=cat_cols, dtype=int)\n"
             "x_train = X[X.is_train == 1].drop(columns='is_train').reset_index(drop=True)\n"
             "x_test  = X[X.is_train == 0].drop(columns='is_train').reset_index(drop=True)\n"
             "x_train.shape, x_test.shape"),
        md("## 2.5. Lưu"),
        code("save_feature_set(FEATURE_SET, x_train, y_train, x_test, test_id,\n"
             "                 info={'notebook': NOTEBOOK, 'member': MEMBER, 'note': 'mô tả ngắn thay đổi so với feature0'})"),
        md("### Kết luận\n- *(ghi nhận xét ở đây)*"),
    ]
    return c


# ============================================================================= 3. FEATURE SELECTION
def fs_cells(member, notebook, feature_set, todo=None):
    c = [md(f"# 3. Lựa chọn đặc trưng\n\n**Input:** `exps/{feature_set}/x_train.csv, y_train.csv`  \n"
            f"**Output:** `exps/{feature_set}/selected_features.json` — từ điển {{tên_cách_chọn: [danh sách cột]}}"),
         *SETUP, params(member, notebook, f'FEATURE_SET = "{feature_set}"')]
    if todo:
        c.append(md(todo))
    c += [
        code("from sklearn.linear_model import Lasso, Ridge\nfrom sklearn.ensemble import GradientBoostingRegressor\n"
             "x_train, y_train, x_test, test_id = load_feature_set(FEATURE_SET)"),
        md("## 3.1. Lasso (L1) — giữ các cột có hệ số khác 0"),
        code("lasso = make_pipeline(Lasso(alpha=0.0005, max_iter=50000)).fit(x_train, y_train)\n"
             "coef = pd.Series(lasso.named_steps['model'].coef_, index=x_train.columns)\n"
             "sel_lasso = coef[coef != 0].index.tolist()\n"
             "print(f'Lasso giữ {len(sel_lasso)}/{x_train.shape[1]} cột')\n"
             "coef.abs().sort_values(ascending=False).head(15)[::-1].plot.barh(figsize=(6, 4)); plt.show()"),
        md("## 3.2. Feature importance của Gradient Boosting — top K"),
        code("gbr = make_pipeline(GradientBoostingRegressor(n_estimators=300, random_state=SEED), scale=False)"
             ".fit(x_train, y_train)\n"
             "imp = pd.Series(gbr.named_steps['model'].feature_importances_, index=x_train.columns)\n"
             "K = 50\nsel_gbr = imp.sort_values(ascending=False).head(K).index.tolist()\n"
             "imp.sort_values(ascending=False).head(15)[::-1].plot.barh(figsize=(6, 4)); plt.show()"),
        md("## 3.3. So sánh nhanh bằng Ridge\n"
           "⚠ Đặc trưng được chọn trên **toàn bộ** train rồi mới CV nên điểm hơi lạc quan; "
           "dùng để so sánh tương đối giữa các cách chọn."),
        code("selections = {'all': x_train.columns.tolist(), 'lasso': sel_lasso, f'gbr_top{K}': sel_gbr}\n"
             "for name, cols in selections.items():\n"
             "    scores, _ = evaluate(Ridge(alpha=10), x_train[cols], y_train)\n"
             "    log_experiment(MEMBER, NOTEBOOK, FEATURE_SET, 'Ridge', scores, selection=name)"),
        code("with open(f'{feature_dir(FEATURE_SET)}/selected_features.json', 'w') as f:\n"
             "    json.dump(selections, f, indent=1)\nlist(selections)"),
        md("### Kết luận\n- *(ghi nhận xét ở đây)*"),
    ]
    return c


# ============================================================================= 4. TRADITIONAL MODEL
def trad_cells(member, notebook, feature_set, todo=None):
    c = [md("# 4. Mô hình học máy truyền thống (scikit-learn)\n\n"
            f"**Input:** `exps/{feature_set}/`  \n**Output:** log thực nghiệm, `exps/{feature_set}/preds/*.npy`, "
            "file nộp `submission_*.csv`"),
         *SETUP, params(member, notebook, f'FEATURE_SET = "{feature_set}"\nSELECTION   = None   # None = tất cả cột; hoặc "lasso", "gbr_top50"...')]
    if todo:
        c.append(md(todo))
    c += [
        code("from models import get_traditional_models\n"
             "x_train, y_train, x_test, test_id = load_feature_set(FEATURE_SET, SELECTION)"),
        md("## 4.1. Đánh giá 5-fold CV\nMọi mô hình đều đi qua `make_pipeline()` (median imputer + StandardScaler) và cùng `get_kfold()`."),
        code("results = {}\nfor name, model in get_traditional_models().items():\n"
             "    scores, oof = evaluate(model, x_train, y_train)\n"
             "    results[name] = scores\n"
             "    log_experiment(MEMBER, NOTEBOOK, FEATURE_SET, name, scores, model.get_params(), SELECTION)\n"
             "    save_preds(FEATURE_SET, f'{MEMBER}_{name}', oof, fit_predict(model, x_train, y_train, x_test))"),
        code("pd.DataFrame(results).plot.box(figsize=(7, 3.5), ylabel='RMSE từng fold'); plt.show()\n"
             "pd.DataFrame(results).agg(['mean', 'std']).T.sort_values('mean')"),
        md("## 4.2. Tạo file nộp cho mô hình tốt nhất"),
        code("best = min(results, key=lambda k: results[k].mean())\n"
             "pred = np.load(f'{feature_dir(FEATURE_SET)}/preds/{MEMBER}_{best}_test.npy')\n"
             "save_submission(FEATURE_SET, f'{MEMBER}_{best}', test_id, pred)"),
        md("### Kết luận\n- *(ghi nhận xét ở đây)*"),
    ]
    return c


# ============================================================================= 4. NEURAL NETWORK
def nn_cells(member, notebook, feature_set, todo=None):
    c = [md("# 4. Mô hình mạng nơ-ron MLP (PyTorch)\n\n"
            f"**Input:** `exps/{feature_set}/`  \n**Output:** log thực nghiệm, `exps/{feature_set}/preds/*.npy`, file nộp"),
         *SETUP, params(member, notebook, f'FEATURE_SET = "{feature_set}"\nSELECTION   = None')]
    if todo:
        c.append(md(todo))
    c += [
        code("from models import cv_mlp\n"
             "x_train, y_train, x_test, test_id = load_feature_set(FEATURE_SET, SELECTION)"),
        md("## 4.1. Cấu hình\nBaseline: 2 lớp ẩn, **không** dropout, **không** early stopping, 100 epoch."),
        code("cfg = dict(hidden=(256, 128), dropout=0.0, lr=1e-3, weight_decay=0.0,\n"
             "           batch_size=64, epochs=100, patience=None)\nRUN_NAME = 'MLP_baseline'"),
        md("## 4.2. Huấn luyện 5-fold"),
        code("scores, oof, test_pred, hists = cv_mlp(x_train, y_train, x_test, **cfg)\n"
             "print('RMSE từng fold:', scores.round(4))\n"
             "log_experiment(MEMBER, NOTEBOOK, FEATURE_SET, RUN_NAME, scores, cfg, SELECTION)\n"
             "save_preds(FEATURE_SET, f'{MEMBER}_{RUN_NAME}', oof, test_pred)"),
        md("## 4.3. Learning curve (fold 0)\nTrain loss giảm mãi trong khi val loss đứng yên → overfit."),
        code("plt.figure(figsize=(6, 3.5))\nplt.plot(hists[0]['train'], label='train'); plt.plot(hists[0]['val'], label='val')\n"
             "plt.yscale('log'); plt.xlabel('epoch'); plt.ylabel('MSE (target chuẩn hóa)'); plt.legend(); plt.show()"),
        code("save_submission(FEATURE_SET, f'{MEMBER}_{RUN_NAME}', test_id, test_pred)"),
        md("### Kết luận\n- *(ghi nhận xét ở đây)*"),
    ]
    return c


# ============================================================================= 5. IMPROVE
def improve_cells(member, notebook, feature_set):
    return [
        md("# 5. Cải tiến: kết hợp mô hình & tổng hợp thực nghiệm\n\n"
           f"**Input:** `exps/{feature_set}/preds/*_oof.npy, *_test.npy` (do stage 4 sinh ra)  \n"
           "**Output:** `submission_blend.csv`, `experiments.xlsx`"),
        *SETUP, params(member, notebook, f'FEATURE_SET = "{feature_set}"'),
        code("from scipy.optimize import nnls\n"
             "x_train, y_train, x_test, test_id = load_feature_set(FEATURE_SET)\n"
             "d = f'{feature_dir(FEATURE_SET)}/preds'\n"
             "names = sorted(f[:-8] for f in os.listdir(d) if f.endswith('_oof.npy'))\n"
             "OOF = np.column_stack([np.load(f'{d}/{n}_oof.npy') for n in names])\n"
             "TEST = np.column_stack([np.load(f'{d}/{n}_test.npy') for n in names])\n"
             "pd.Series({n: rmse(y_train, OOF[:, i]) for i, n in enumerate(names)}).sort_values().to_frame('OOF RMSE')"),
        md("## 5.1. Blend: trọng số không âm (NNLS) trên dự đoán out-of-fold\n"
           "Trọng số chọn trên chính OOF nên điểm hơi lạc quan (số tham số nhỏ nên lạc quan ít)."),
        code("w, _ = nnls(OOF, y_train); w = w / w.sum()\n"
             "print({n: round(float(v), 3) for n, v in zip(names, w) if v > 0.001})\n"
             "blend_oof = OOF @ w\nscores = np.array([rmse(y_train[va], blend_oof[va]) for _, va in get_kfold().split(x_train)])\n"
             "log_experiment(MEMBER, NOTEBOOK, FEATURE_SET, 'Blend', scores, {n: round(float(v), 3) for n, v in zip(names, w)})\n"
             "save_submission(FEATURE_SET, 'blend', test_id, TEST @ w)"),
        md("## 5.2. Tổng hợp log của cả nhóm → `experiments.xlsx`"),
        code("export_experiments().head(20)"),
        md("### Kết luận\n- *(ghi nhận xét ở đây)*"),
    ]


# ============================================================================= TODO của từng thành viên
TODO_TV1_EDA = """
## VIỆC CỦA TV1 (EDA mở rộng cho báo cáo Chương 2)
- Đọc paper De Cock (doc/README.md): nguồn dữ liệu, ý nghĩa biến, khuyến nghị về ngoại lai.
- Thêm phân tích: SalePrice theo Neighborhood (boxplot), theo OverallQual; tương quan giữa các biến (đa cộng tuyến).
- Với **mỗi** cột thiếu dữ liệu, tra `data_description.txt`: NaN là *“không có”* hay *thiếu thật*? → lập bảng, chuyển cho pre-processing1.
"""
TODO_TV1_PREP = """
## VIỆC CỦA TV1 — tạo `feature1` (chỉ sửa phần làm sạch, giữ nguyên các bước khác)
1. **NaN = “không có”:** điền `"None"` cho BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1/2, FireplaceQu, Garage*, MasVnrType...; điền 0 cho MasVnrArea, BsmtFinSF1/2, TotalBsmtSF, GarageCars, GarageArea... Khi đó **giữ lại** Alley/PoolQC/Fence/MiscFeature thay vì xóa.
2. **Ngoại lai:** bỏ các dòng train có GrLivArea > 4000 (mục 2.2).
3. **Mã hóa thứ bậc:** Ex/Gd/TA/Fa/Po → 5..1 (None = 0) cho các cột chất lượng, bỏ chúng khỏi `cat_cols`.
4. Mỗi thay đổi: chạy lại notebook này rồi chạy nhanh `4.model/traditional model.ipynb` với `FEATURE_SET="feature1"` để đo tác động; ghi vào bảng cho báo cáo Chương 3.
"""
TODO_TV4_PREP = """
## VIỆC CỦA TV4 — tạo `feature4` (feature engineering), làm song song với TV1
Làm trên phần làm sạch **baseline** (không chờ TV1). Khi ghép (giai đoạn 2) sẽ chép các hàm FE sang notebook của TV1.
1. Viết thêm nhóm đặc trưng ở mục 2.3: `baths` (tổng phòng tắm), `age` (YrSold − YearBuilt, YrSold − YearRemodAdd), `flags` (HasPool, HasGarage, HasBsmt, Has2ndFloor...), `interactions` (OverallQual × GrLivArea...). Tự đề xuất ít nhất 1 nhóm.
2. **Ablation:** bật từng nhóm (`USE = [...]`), lưu thành `feature4_1`, `feature4_2`... hoặc chạy lần lượt rồi đo bằng traditional model → bảng cho báo cáo Chương 5.
3. Thử giảm skew: log1p các cột số có |skew| > 0.75 (tính skew **chỉ trên dòng train**).
"""
TODO_TV4_FS = """
## VIỆC CỦA TV4 — lựa chọn đặc trưng
1. Thêm cách chọn: `SelectKBest(f_regression)`, `RFE`, ngưỡng tương quan, Lasso với nhiều alpha.
2. So sánh số cột giữ lại vs. CV RMSE (vẽ đồ thị K – RMSE).
3. (Nâng cao) Làm selection **bên trong** từng fold để có điểm không lạc quan.
"""
TODO_TV2 = """
## VIỆC CỦA TV2 — làm trên `feature0` ngay từ đầu, khi có feature mới chỉ cần đổi `FEATURE_SET`
1. Thêm mô hình: ElasticNet, SVR, KNN, XGBoost/LightGBM (tùy chọn). Chép `get_traditional_models()` vào notebook rồi sửa.
2. Tinh chỉnh tham số bằng `GridSearchCV(make_pipeline(model), grid, cv=get_kfold(), scoring='neg_root_mean_squared_error')` — tên tham số có tiền tố `model__`.
3. Lập bảng: mô hình × feature set (feature0 → feature1/4 → feature5) cho báo cáo Chương 4, 6.
4. Lưu dự đoán bằng `save_preds()` để 5.improve dùng blend.
"""
TODO_TV3 = """
## VIỆC CỦA TV3 — làm trên `feature0` ngay từ đầu
Baseline bị overfit (xem learning curve). Thử lần lượt, mỗi lần đặt `RUN_NAME` khác nhau:
1. Early stopping: `patience=30, epochs=500`.
2. Regularization: `dropout` 0.1–0.5, `weight_decay` 1e-4–1e-2; thêm BatchNorm (copy class `MLP` từ `extra/models.py` vào notebook rồi sửa).
3. Kiến trúc: (128,), (256,128), (512,256), (256,128,64); lr, batch size.
4. Lập bảng cấu hình – CV RMSE – std; vẽ learning curve trước/sau cho báo cáo Chương 4.
"""

if __name__ == "__main__":
    # ---------------- baseline (người tạo khung, không ai sửa trực tiếp)
    save(eda_cells("baseline", "EDA"), "1.EDA/EDA.ipynb")
    save(prep_cells("baseline", "pre-processing", "feature0"), "2.pre-processing/pre-processing.ipynb")
    save(fs_cells("baseline", "feature_selection", "feature0"), "3.feature_selection/feature_selection.ipynb")
    save(trad_cells("baseline", "traditional model", "feature0"), "4.model/traditional model.ipynb")
    save(nn_cells("baseline", "neural network model", "feature0"), "4.model/neural network model.ipynb")
    save(improve_cells("baseline", "improve", "feature0"), "5.improve/improve.ipynb")
    # ---------------- notebook của từng thành viên (làm song song)
    save(eda_cells("TV1", "EDA1", TODO_TV1_EDA), "1.EDA/EDA1.ipynb")
    save(prep_cells("TV1", "pre-processing1", "feature1", TODO_TV1_PREP), "2.pre-processing/pre-processing1.ipynb")
    save(trad_cells("TV2", "traditional model2", "feature0", TODO_TV2), "4.model/traditional model2.ipynb")
    save(nn_cells("TV3", "neural network model3", "feature0", TODO_TV3), "4.model/neural network model3.ipynb")
    save(prep_cells("TV4", "pre-processing4", "feature4", TODO_TV4_PREP, fe_stub=True), "2.pre-processing/pre-processing4.ipynb")
    save(fs_cells("TV4", "feature_selection4", "feature4", TODO_TV4_FS), "3.feature_selection/feature_selection4.ipynb")
