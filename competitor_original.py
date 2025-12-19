import pandas as pd
import numpy as np
from tqdm import tqdm
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 0. 공통 하이퍼파라미터 (현재 best 세팅)
# ------------------------------------------------------------
PAIR_MAX_LAG = 7
PAIR_MIN_NONZERO = 8           # classifier 학습용 min_nonzero
VAL_MIN_NONZERO = 2            # validation용 min_nonzero

PAIR_LABEL_CORR_THRESHOLD = 0.32  # 0.38 → 0.32 (more positive samples)
VAL_GT_CORR_THRESHOLD = 0.25

NEG_POS_RATIO = 1.5  # 2.0 → 1.5 (better balance)
PAIR_TOP_K = 3000    # 2500 → 3000

TRAIN_END_STR = "2024-12-01"
VAL_START_STR = "2025-01-01"
VAL_END_STR = "2025-04-01"

RANDOM_SEED = 42

# 회귀 feature 고정 (13개)
REG_FEATURE_COLS_FIXED = [
    "b_t", "b_t_1", "b_t_2",
    "b_ma3", "b_change",
    "a_t_lag", "a_t_lag_1",
    "a_ma3", "a_change",
    "ab_value_ratio",
    "max_corr", "best_lag", "corr_stability",
]


# ------------------------------------------------------------
# 1. 유틸 함수
# ------------------------------------------------------------
def safe_corr(a, b):
    mask = (~np.isnan(a)) & (~np.isnan(b))
    if mask.sum() < 3:
        return 0.0
    aa, bb = a[mask], b[mask]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def load_pivot(train_path="train.csv"):
    df = pd.read_csv(train_path)

    monthly = (
        df.groupby(["item_id", "year", "month"], as_index=False)["value"]
          .sum()
    )

    monthly["ym"] = pd.to_datetime(
        monthly["year"].astype(str)
        + "-" +
        monthly["month"].astype(str).str.zfill(2) + "-01"
    )

    pivot = monthly.pivot(index="item_id", columns="ym", values="value")
    pivot = pivot.fillna(0).sort_index(axis=1)

    print("pivot shape:", pivot.shape)
    print("months:", pivot.columns[0].strftime("%Y-%m"),
          "->", pivot.columns[-1].strftime("%Y-%m"))
    return pivot, df


def get_time_indices(pivot):
    months = list(pivot.columns)
    month_to_idx = {m: i for i, m in enumerate(months)}

    train_end = pd.to_datetime(TRAIN_END_STR)
    val_start = pd.to_datetime(VAL_START_STR)
    val_end = pd.to_datetime(VAL_END_STR)

    return (months,
            month_to_idx[train_end],
            month_to_idx[val_start],
            month_to_idx[val_end])


# ------------------------------------------------------------
# 2. Pair feature + classifier
# ------------------------------------------------------------
def build_pair_feature_matrix(
    pivot,
    upto_idx,
    max_lag=7,
    min_nonzero=8,
    corr_threshold_for_label=0.30,
    neg_pos_ratio=2.0
):
    items = pivot.index.to_list()
    months = list(pivot.columns)

    sub_cols = months[:upto_idx + 1]
    pivot_sub = pivot[sub_cols]
    n_sub_months = pivot_sub.shape[1]

    rows_pos = []
    rows_neg = []

    for leader in tqdm(items, desc="build_pair_features"):
        a = pivot_sub.loc[leader].values.astype(float)
        if np.count_nonzero(a) < min_nonzero:
            continue

        for follower in items:
            if leader == follower:
                continue

            b = pivot_sub.loc[follower].values.astype(float)
            if np.count_nonzero(b) < min_nonzero:
                continue

            lag_corrs = []
            best_corr = 0.0
            second_corr = 0.0
            best_lag = None

            for lag in range(1, max_lag + 1):
                if n_sub_months <= lag:
                    lag_corrs.append(0.0)
                    continue

                c = safe_corr(a[:-lag], b[lag:])
                lag_corrs.append(c)

                if abs(c) > abs(best_corr):
                    second_corr = best_corr
                    best_corr = c
                    best_lag = lag
                elif abs(c) > abs(second_corr):
                    second_corr = c

            if best_lag is None:
                continue

            lag_corrs = np.array(lag_corrs, dtype=float)

            feats = {
                "leading_item_id": leader,
                "following_item_id": follower,
                "max_corr": float(best_corr),
                "best_lag": int(best_lag),
                "second_corr": float(second_corr),
                "corr_stability": float(abs(best_corr - second_corr)),
                "corr_mean": float(np.mean(lag_corrs)),
                "corr_std": float(np.std(lag_corrs)),
                "corr_abs_mean": float(np.mean(np.abs(lag_corrs))),
                "nonzero_a": int(np.count_nonzero(a)),
                "nonzero_b": int(np.count_nonzero(b)),
                "sum_a": float(a.sum()),
                "sum_b": float(b.sum())
            }

            label = 1 if abs(best_corr) >= corr_threshold_for_label else 0

            if label == 1:
                rows_pos.append({**feats, "label": 1})
            else:
                rows_neg.append({**feats, "label": 0})

    df_pos = pd.DataFrame(rows_pos)
    df_neg = pd.DataFrame(rows_neg)
    print("pos pairs:", df_pos.shape, "neg pairs:", df_neg.shape)

    if df_pos.empty:
        print("No positive pairs found.")
        return pd.DataFrame()

    # negative sampling
    n_pos = len(df_pos)
    n_neg_keep = int(neg_pos_ratio * n_pos)
    if len(df_neg) > n_neg_keep:
        df_neg = df_neg.sample(n_neg_keep, random_state=RANDOM_SEED)

    df_all = pd.concat([df_pos, df_neg], axis=0).reset_index(drop=True)
    print("pair feature dataset shape:", df_all.shape)
    return df_all


def train_pair_classifier(df_pairs):
    feature_cols = [
        "max_corr", "best_lag", "second_corr",
        "corr_stability", "corr_mean", "corr_std", "corr_abs_mean",
        "nonzero_a", "nonzero_b", "sum_a", "sum_b"
    ]

    df = df_pairs.copy()
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)

    X = df[feature_cols].values
    y = df["label"].values

    clf = XGBClassifier(
        n_estimators=300,     # 200 → 300 (more trees)
        max_depth=5,          # 4 → 5 (deeper)
        learning_rate=0.05,   # 0.08 → 0.05 (slower, more stable)
        subsample=0.85,       # 0.9 → 0.85
        colsample_bytree=0.85,  # 0.9 → 0.85
        reg_alpha=0.3,        # 0.5 → 0.3
        reg_lambda=0.8,       # 1.0 → 0.8
        min_child_weight=2,   # 3 → 2
        gamma=0.1,            # 0.2 → 0.1
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric="logloss"
    )

    clf.fit(X, y)
    return clf, feature_cols


def score_all_pairs_with_classifier(
    pivot, clf, feature_cols,
    max_lag=7, min_nonzero=8, top_k=None
):
    items = pivot.index.to_list()
    months = list(pivot.columns)
    n_months = len(months)

    rows = []

    for leader in tqdm(items, desc="score_all_pairs"):
        a = pivot.loc[leader].values.astype(float)
        if np.count_nonzero(a) < min_nonzero:
            continue

        for follower in items:
            if leader == follower:
                continue

            b = pivot.loc[follower].values.astype(float)
            if np.count_nonzero(b) < min_nonzero:
                continue

            lag_corrs = []
            best_corr = 0.0
            second_corr = 0.0
            best_lag = None

            for lag in range(1, max_lag + 1):
                if n_months <= lag:
                    lag_corrs.append(0.0)
                    continue

                c = safe_corr(a[:-lag], b[lag:])
                lag_corrs.append(c)

                if abs(c) > abs(best_corr):
                    second_corr = best_corr
                    best_corr = c
                    best_lag = lag
                elif abs(c) > abs(second_corr):
                    second_corr = c

            if best_lag is None:
                continue

            feats = {
                "max_corr": float(best_corr),
                "best_lag": int(best_lag),
                "second_corr": float(second_corr),
                "corr_stability": float(abs(best_corr - second_corr)),
                "corr_mean": float(np.mean(lag_corrs)),
                "corr_std": float(np.std(lag_corrs)),
                "corr_abs_mean": float(np.mean(np.abs(lag_corrs))),
                "nonzero_a": int(np.count_nonzero(a)),
                "nonzero_b": int(np.count_nonzero(b)),
                "sum_a": float(a.sum()),
                "sum_b": float(b.sum())
            }

            x_vec = np.array([[feats[col] for col in feature_cols]], dtype=float)
            prob = float(clf.predict_proba(x_vec)[0, 1])

            rows.append({
                "leading_item_id": leader,
                "following_item_id": follower,
                "best_lag": int(best_lag),
                "max_corr": float(best_corr),
                "corr_stability": float(abs(best_corr - second_corr)),
                "clf_prob": prob
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if top_k is not None and len(df) > top_k:
        df = df.sort_values("clf_prob", ascending=False).head(top_k)

    return df.reset_index(drop=True)


# ------------------------------------------------------------
# 3. Validation용 GT 계산
# ------------------------------------------------------------
def get_val_gt_pairs(
    pivot,
    max_lag=7,
    min_nonzero=8,
    corr_threshold=0.20,
    start_idx=None,
    end_idx=None
):
    items = pivot.index.to_list()
    months = list(pivot.columns)

    sub_cols = months[start_idx:end_idx + 1]
    pivot_sub = pivot[sub_cols]
    n_sub_months = pivot_sub.shape[1]

    gt_set = set()

    for leader in tqdm(items, desc="GT pairs(val)"):
        a = pivot_sub.loc[leader].values.astype(float)
        if np.count_nonzero(a) < min_nonzero:
            continue

        for follower in items:
            if leader == follower:
                continue

            b = pivot_sub.loc[follower].values.astype(float)
            if np.count_nonzero(b) < min_nonzero:
                continue

            best_corr = 0.0
            best_lag = None

            for lag in range(1, max_lag + 1):
                if n_sub_months <= lag:
                    continue

                c = safe_corr(a[:-lag], b[lag:])
                if abs(c) > abs(best_corr):
                    best_corr = c
                    best_lag = lag

            if best_lag is not None and abs(best_corr) >= corr_threshold:
                gt_set.add((leader, follower))

    print("val GT size:", len(gt_set))
    return gt_set


# ------------------------------------------------------------
# 4. Regression dataset 생성
# ------------------------------------------------------------
def build_pair_dataset(pivot, pairs, target_start_idx, target_end_idx):
    months = list(pivot.columns)
    n_months = len(months)

    rows = []

    for row in tqdm(pairs.itertuples(index=False), desc="build_pair_dataset"):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)

        a = pivot.loc[leader].values.astype(float)
        b = pivot.loc[follower].values.astype(float)

        for t in range(lag + 2, n_months - 1):
            target_idx = t + 1
            if target_idx < target_start_idx or target_idx > target_end_idx:
                continue

            if t - 2 < 0 or (t - lag - 1) < 0:
                continue

            b_t = b[t]
            b_t_1 = b[t - 1]
            b_t_2 = b[t - 2]

            a_t_lag = a[t - lag]
            a_t_lag_1 = a[t - lag - 1]

            b_ma3 = np.mean([b_t, b_t_1, b_t_2])

            if (t - lag - 2) >= 0:
                a_ma3 = np.mean([a_t_lag, a_t_lag_1, a[t - lag - 2]])
            else:
                a_ma3 = np.mean([a_t_lag, a_t_lag_1])

            b_change = (b_t - b_t_1) / (b_t_1 + 1)
            a_change = (a_t_lag - a_t_lag_1) / (a_t_lag_1 + 1)

            ab_ratio = b_t / (a_t_lag + 1)

            target = b[target_idx]

            rows.append({
                "leading_item_id": leader,
                "following_item_id": follower,
                "b_t": b_t,
                "b_t_1": b_t_1,
                "b_t_2": b_t_2,
                "b_ma3": b_ma3,
                "b_change": b_change,
                "a_t_lag": a_t_lag,
                "a_t_lag_1": a_t_lag_1,
                "a_ma3": a_ma3,
                "a_change": a_change,
                "ab_value_ratio": ab_ratio,
                "max_corr": row.max_corr,
                "best_lag": lag,
                "corr_stability": row.corr_stability,
                "target": target,
            })

    df = pd.DataFrame(rows)
    print("reg dataset shape:", df.shape)
    return df


# ------------------------------------------------------------
# 5. Regression model + NMAE
# ------------------------------------------------------------
def train_xgb_regressor(df_train):
    feature_cols = REG_FEATURE_COLS_FIXED

    df_train = df_train.replace([np.inf, -np.inf], 0).fillna(0)

    X = df_train[feature_cols].values
    y = df_train["target"].values

    model = XGBRegressor(
        n_estimators=300,     # 200 → 300
        max_depth=5,          # 4 → 5
        learning_rate=0.05,   # 0.08 → 0.05
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,   # 5 → 3
        gamma=0.1,            # 0.2 → 0.1
        reg_alpha=0.3,        # 0.5 → 0.3
        reg_lambda=0.8,       # 1.0 → 0.8
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(X, y)
    return model, feature_cols


def compute_nmae(model, df_val, feature_cols=None):
    if feature_cols is None:
        feature_cols = REG_FEATURE_COLS_FIXED

    df_val = df_val.replace([np.inf, -np.inf], 0).fillna(0)

    X = df_val[feature_cols].values
    y_true = df_val["target"].values

    y_pred = model.predict(X)
    y_pred = np.clip(y_pred, 0, None)

    eps = 1e-6
    nmae = np.mean(
        np.minimum(1.0, np.abs(y_true - y_pred) / (np.abs(y_true) + eps))
    )
    return float(nmae)


# ------------------------------------------------------------
# 6. Union metric (tau 기준)
# ------------------------------------------------------------
def eval_threshold_union(
    pivot,
    pairs_all,
    reg_model,
    reg_feature_cols,
    val_gt_set,
    val_start_idx,
    val_end_idx,
    tau=0.5,
):
    pairs_tau = pairs_all[pairs_all["clf_prob"] >= tau].copy().reset_index(drop=True)

    pred_set = set(
        (r.leading_item_id, r.following_item_id)
        for r in pairs_tau.itertuples(index=False)
    )

    tp = len(val_gt_set & pred_set)
    fp = len(pred_set - val_gt_set)
    fn = len(val_gt_set - pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    df_val_tau = build_pair_dataset(
        pivot,
        pairs_tau,
        target_start_idx=val_start_idx,
        target_end_idx=val_end_idx
    )

    if len(df_val_tau) == 0:
        nmae_union = 1.0
    else:
        df = df_val_tau.replace([np.inf, -np.inf], 0).fillna(0)

        X = df[reg_feature_cols].values
        y_true = df["target"].values

        y_pred = reg_model.predict(X)
        y_pred = np.clip(y_pred, 0, None)

        eps = 1e-6
        df["err_row"] = np.minimum(1.0, np.abs(y_true - y_pred) / (np.abs(y_true) + eps))

        pair_err = (
            df.groupby(["leading_item_id", "following_item_id"])["err_row"]
            .mean()
            .to_dict()
        )

        U = val_gt_set | pred_set
        errs = []
        for pair in U:
            if pair in val_gt_set and pair in pred_set:
                errs.append(pair_err.get(pair, 1.0))
            else:
                errs.append(1.0)

        nmae_union = float(np.mean(errs))

    score = 0.6 * f1 + 0.4 * (1.0 - nmae_union)
    print(f"[tau={tau:.3f}] F1={f1:.4f}, union NMAE={nmae_union:.4f}, Score={score:.4f}")
    return f1, nmae_union, score


# ------------------------------------------------------------
# 7. ✅ Validation 전체 파이프라인 (파일 저장 X)
# ------------------------------------------------------------
def run_validation():
    """
    로컬 검증용:
    - Pair classifier 학습
    - Regression 학습
    - Forecast NMAE 출력
    - tau 스캔해서 union Score 출력
    """
    pivot, raw = load_pivot("train.csv")
    months, train_end_idx, val_start_idx, val_end_idx = get_time_indices(pivot)

    # 1) pair classifier 학습용 feature
    df_pairs_label = build_pair_feature_matrix(
        pivot,
        upto_idx=train_end_idx,
        max_lag=PAIR_MAX_LAG,
        min_nonzero=PAIR_MIN_NONZERO,
        corr_threshold_for_label=PAIR_LABEL_CORR_THRESHOLD,
        neg_pos_ratio=NEG_POS_RATIO
    )

    if df_pairs_label.empty:
        raise RuntimeError("No pair features for classifier.")

    pair_clf, pair_feature_cols = train_pair_classifier(df_pairs_label)

    # 2) 전체 쌍 scoring
    pairs_all = score_all_pairs_with_classifier(
        pivot,
        clf=pair_clf,
        feature_cols=pair_feature_cols,
        max_lag=PAIR_MAX_LAG,
        min_nonzero=VAL_MIN_NONZERO,
        top_k=None
    )

    if pairs_all.empty:
        raise RuntimeError("No scored pairs.")

    pairs_pred = (
        pairs_all.sort_values("clf_prob", ascending=False)
        .head(PAIR_TOP_K)
        .reset_index(drop=True)
    )

    # 3) validation용 GT 쌍
    val_gt_set = get_val_gt_pairs(
        pivot,
        max_lag=PAIR_MAX_LAG,
        min_nonzero=VAL_MIN_NONZERO,
        corr_threshold=VAL_GT_CORR_THRESHOLD,
        start_idx=val_start_idx,
        end_idx=val_end_idx
    )

    # 4) regression train/val dataset + NMAE
    df_train = build_pair_dataset(
        pivot, pairs_pred,
        target_start_idx=0,
        target_end_idx=train_end_idx
    )
    reg_model, reg_feature_cols = train_xgb_regressor(df_train)

    df_val = build_pair_dataset(
        pivot, pairs_pred,
        target_start_idx=val_start_idx,
        target_end_idx=val_end_idx
    )
    nmae_val = compute_nmae(reg_model, df_val, reg_feature_cols)
    print(f"[Forecast] NMAE(val) = {nmae_val:.4f}")

    # 5) tau별 union metric
    taus = np.linspace(0.30, 0.55, 11)
    best_score = -1
    best_tau = None
    best_result = None

    for tau in taus:
        f1_t, nmae_t, score_t = eval_threshold_union(
            pivot,
            pairs_all=pairs_all,
            reg_model=reg_model,
            reg_feature_cols=reg_feature_cols,
            val_gt_set=val_gt_set,
            val_start_idx=val_start_idx,
            val_end_idx=val_end_idx,
            tau=tau
        )
        if score_t > best_score:
            best_score = score_t
            best_tau = tau
            best_result = (f1_t, nmae_t, score_t)

    print("\n=== [Validation] BEST tau ===")
    if best_result is not None:
        print(f"tau={best_tau:.3f}, F1={best_result[0]:.4f}, "
            f"NMAE={best_result[1]:.4f}, Score={best_result[2]:.4f}")
    else:
        print("No valid tau result")


# ------------------------------------------------------------
# 8. ✅ 제출용: submission 파일 생성
# ------------------------------------------------------------
def run_submission(out_path="submission.csv"):
    """
    실제 제출용 CSV 생성:
    - train_end까지로 pair classifier + regressor 학습
    - 마지막 월 기준으로 다음달(2025-08) 예측
    - out_path로 submission 저장
    """
    pivot, raw = load_pivot("train.csv")
    months, train_end_idx, _, _ = get_time_indices(pivot)

    # 1) pair classifier 학습
    df_pairs_label = build_pair_feature_matrix(
        pivot,
        upto_idx=train_end_idx,
        max_lag=PAIR_MAX_LAG,
        min_nonzero=PAIR_MIN_NONZERO,
        corr_threshold_for_label=PAIR_LABEL_CORR_THRESHOLD,
        neg_pos_ratio=NEG_POS_RATIO
    )
    if df_pairs_label.empty:
        raise RuntimeError("No pair features for classifier.")

    pair_clf, pair_feature_cols = train_pair_classifier(df_pairs_label)

    # 2) 전체 쌍 scoring
    pairs_all = score_all_pairs_with_classifier(
        pivot,
        clf=pair_clf,
        feature_cols=pair_feature_cols,
        max_lag=PAIR_MAX_LAG,
        min_nonzero=PAIR_MIN_NONZERO,
        top_k=None
    )
    if pairs_all.empty:
        raise RuntimeError("No scored pairs.")

    pairs_pred = (
        pairs_all.sort_values("clf_prob", ascending=False)
        .head(PAIR_TOP_K)
        .reset_index(drop=True)
    )

    # 3) regression train (train_end까지 전부 사용)
    df_train = build_pair_dataset(
        pivot, pairs_pred,
        target_start_idx=0,
        target_end_idx=train_end_idx
    )
    reg_model, reg_feature_cols = train_xgb_regressor(df_train)

    # 4) 제출 feature 생성 (마지막 관측 월 → 다음달)
    last_idx = len(months) - 1
    sub_rows = []

    for row in pairs_pred.itertuples(index=False):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)

        a = pivot.loc[leader].values.astype(float)
        b = pivot.loc[follower].values.astype(float)

        if last_idx - 2 < 0 or (last_idx - lag - 1) < 0:
            continue

        b_t = b[last_idx]
        b_t_1 = b[last_idx - 1]
        b_t_2 = b[last_idx - 2]

        a_t_lag = a[last_idx - lag]
        a_t_lag_1 = a[last_idx - lag - 1]

        b_ma3 = np.mean([b_t, b_t_1, b_t_2])
        if (last_idx - lag - 2) >= 0:
            a_ma3 = np.mean([a_t_lag, a_t_lag_1, a[last_idx - lag - 2]])
        else:
            a_ma3 = np.mean([a_t_lag, a_t_lag_1])

        b_change = (b_t - b_t_1) / (b_t_1 + 1)
        a_change = (a_t_lag - a_t_lag_1) / (a_t_lag_1 + 1)

        ab_ratio = b_t / (a_t_lag + 1)

        feat = {
            "b_t": b_t,
            "b_t_1": b_t_1,
            "b_t_2": b_t_2,
            "b_ma3": b_ma3,
            "b_change": b_change,
            "a_t_lag": a_t_lag,
            "a_t_lag_1": a_t_lag_1,
            "a_ma3": a_ma3,
            "a_change": a_change,
            "ab_value_ratio": ab_ratio,
            "max_corr": row.max_corr,
            "best_lag": int(row.best_lag),
            "corr_stability": row.corr_stability,
        }

        X = np.array([[feat[col] for col in reg_feature_cols]])
        pred = float(reg_model.predict(X)[0])
        pred = max(pred, 0.0)

        sub_rows.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": int(round(pred))
        })

    df_sub = pd.DataFrame(sub_rows).drop_duplicates(
        ["leading_item_id", "following_item_id"]
    )

    df_sub.to_csv(out_path, index=False)
    print("Saved submission:", out_path, "shape:", df_sub.shape)


# ------------------------------------------------------------
# 9. 실행
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Run Validation")
    print("=" * 60)
    run_validation()

    print("\n" + "=" * 60)
    print("Run Submission")
    print("=" * 60)
    run_submission("submission_improved.csv")
