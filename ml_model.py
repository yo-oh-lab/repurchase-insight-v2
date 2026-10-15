"""범용 데이터용 선택적 재구매 예측 모델. 시간 기준 분리로 미래 정보 누수를 줄인다."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_DAYS = 90


def train_repurchase_model(df: pd.DataFrame, order_col: str, customer_col: str, date_col: str, max_customers: int = 5000) -> dict:
    if not date_col or date_col not in df.columns:
        raise ValueError("향후 90일 예측에는 날짜 컬럼이 필요합니다.")
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    x = x.dropna(subset=[date_col, order_col, customer_col]).copy()
    x[order_col] = x[order_col].astype(str)
    x[customer_col] = x[customer_col].astype(str)
    x = x.sort_values(date_col)
    # 주문 단위 dedup; feature columns are numeric/categorical available in uploaded file
    x = x.drop_duplicates([order_col, customer_col], keep="last")
    if x[customer_col].nunique() > max_customers:
        keep_ids = x[customer_col].drop_duplicates().sample(n=max_customers, random_state=42)
        x = x[x[customer_col].isin(set(keep_ids))]
    start, end = x[date_col].min(), x[date_col].max() - pd.Timedelta(days=TARGET_DAYS)
    if pd.isna(start) or start >= end or x[customer_col].nunique() < 30:
        raise ValueError("예측 모델에는 충분한 고객 수와 90일 이상의 관측 기간이 필요합니다.")
    cutoffs = pd.date_range(start + pd.Timedelta(days=30), end, periods=min(6, max(2, (end-start).days // 60 + 1)))
    numeric = [c for c in x.columns if c not in {order_col, customer_col, date_col} and pd.api.types.is_numeric_dtype(x[c])]
    # numeric-looking strings
    for c in x.columns:
        if c in {order_col, customer_col, date_col} or c in numeric:
            continue
        conv = pd.to_numeric(x[c], errors="coerce")
        if x[c].notna().sum() and conv.notna().sum() / x[c].notna().sum() >= .85:
            x[c] = conv
            numeric.append(c)
    categorical = [c for c in x.columns if c not in {order_col, customer_col, date_col} and c not in numeric]
    categorical = [c for c in categorical if x[c].nunique(dropna=True) <= 50]
    numeric = numeric[:20]
    categorical = categorical[:10]
    rows = []
    for cutoff in cutoffs:
        hist = x[x[date_col] <= cutoff]
        future = x[(x[date_col] > cutoff) & (x[date_col] <= cutoff + pd.Timedelta(days=TARGET_DAYS))]
        future_ids = set(future[customer_col].unique())
        for cid, group in hist.groupby(customer_col, sort=False):
            row = {"customer_id": cid, "snapshot_date": cutoff, "target": int(cid in future_ids), "order_count": group[order_col].nunique(), "days_since_last_order": max(0, int((cutoff - group[date_col].max()).days))}
            if numeric:
                for c in numeric:
                    vals = pd.to_numeric(group[c], errors="coerce")
                    row[f"mean__{c}"] = vals.mean()
                    row[f"last__{c}"] = vals.iloc[-1] if len(vals) else np.nan
            if categorical:
                for c in categorical:
                    row[f"last__{c}"] = str(group[c].dropna().iloc[-1]) if group[c].notna().any() else np.nan
            rows.append(row)
    snapshots = pd.DataFrame(rows)
    if snapshots.empty or snapshots["target"].nunique() < 2:
        raise ValueError("학습 데이터에 향후 90일 내 재구매/비재구매 사례가 모두 있어야 합니다.")
    generated_numeric = [c for c in snapshots.columns if c in {"order_count", "days_since_last_order"} or c.startswith("mean__") or (c.startswith("last__") and c.removeprefix("last__") in numeric)]
    generated_cat = [c for c in snapshots.columns if c.startswith("last__") and c not in generated_numeric]
    feature_cols = generated_numeric + generated_cat
    # 시간 순서로 train/test split; if only one distinct cutoff, reject rather than claim future validation
    dates = sorted(snapshots["snapshot_date"].unique())
    if len(dates) < 2:
        raise ValueError("시간순 평가를 위해 최소 2개의 기준 시점이 필요합니다.")
    split_date = dates[max(0, int(len(dates) * .8) - 1)]
    train = snapshots[snapshots["snapshot_date"] <= split_date]
    test = snapshots[snapshots["snapshot_date"] > split_date]
    if train.empty or test.empty or train["target"].nunique() < 2:
        split = max(1, int(len(snapshots) * .8))
        train, test = snapshots.iloc[:split], snapshots.iloc[split:]
    if train.empty or test.empty or train["target"].nunique() < 2:
        raise ValueError("모델 평가용 학습/검증 표본을 나눌 수 없습니다.")
    transformers = []
    if generated_numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), generated_numeric))
    if generated_cat:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=50))]), generated_cat))
    model = Pipeline([("preprocessor", ColumnTransformer(transformers)), ("classifier", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42))])
    model.fit(train[feature_cols], train["target"].astype(int))
    y = test["target"].astype(int)
    probs = model.predict_proba(test[feature_cols])[:, 1]
    auc = float(roc_auc_score(y, probs)) if y.nunique() == 2 else np.nan
    # latest feature row per customer at the latest observed date
    cutoff = x[date_col].max()
    latest_rows = []
    for cid, group in x.groupby(customer_col, sort=False):
        group = group.sort_values(date_col)
        row = {"customer_id": cid, "order_count": group[order_col].nunique(), "days_since_last_order": max(0, int((cutoff-group[date_col].max()).days))}
        for c in numeric:
            if f"mean__{c}" in feature_cols:
                row[f"mean__{c}"] = pd.to_numeric(group[c], errors="coerce").mean()
            if f"last__{c}" in feature_cols:
                vals = pd.to_numeric(group[c], errors="coerce")
                row[f"last__{c}"] = vals.iloc[-1] if len(vals) else np.nan
        for c in categorical:
            if f"last__{c}" in feature_cols:
                row[f"last__{c}"] = str(group[c].dropna().iloc[-1]) if group[c].notna().any() else np.nan
        latest_rows.append(row)
    latest = pd.DataFrame(latest_rows)
    for c in feature_cols:
        if c not in latest:
            latest[c] = np.nan
    latest = latest[["customer_id"] + feature_cols]
    # Global model coefficient magnitude and customer-specific linear contribution are separate outputs.
    global_importance = pd.DataFrame()
    try:
        names = model.named_steps["preprocessor"].get_feature_names_out()
        coefs = model.named_steps["classifier"].coef_[0]
        global_importance = pd.DataFrame({"특성": names, "계수": coefs, "절대계수": np.abs(coefs)})
        global_importance["특성"] = global_importance["특성"].astype(str).str.replace(r"^(numeric|categorical)__", "", regex=True)
        global_importance = global_importance.sort_values("절대계수", ascending=False).head(20).reset_index(drop=True)
        transformed = model.named_steps["preprocessor"].transform(latest[feature_cols])
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        contributions = pd.DataFrame(transformed * coefs, columns=names)
        contributions.insert(0, "customer_id", latest["customer_id"].values)
    except Exception:
        contributions = pd.DataFrame()
    return {"model": model, "latest_rows": latest, "feature_cols": feature_cols, "orders": x, "order_col": order_col, "customer_col": customer_col, "date_col": date_col, "accuracy": float(accuracy_score(y, probs >= .5)), "roc_auc": auc, "brier_score": float(brier_score_loss(y, probs)), "training_rows": len(train), "test_rows": len(test), "snapshot_rows": len(snapshots), "target_days": TARGET_DAYS, "data_start": x[date_col].min(), "data_end": x[date_col].max(), "contributions": contributions, "global_importance": global_importance}


def predict_customer_repurchase(ml_result: dict, customer_id: str) -> dict | None:
    latest = ml_result["latest_rows"]
    matched = latest[latest["customer_id"].astype(str) == str(customer_id)]
    if matched.empty:
        return None
    model = ml_result["model"]
    features = ml_result["feature_cols"]
    score = float(model.predict_proba(matched[features])[0, 1])
    factors = pd.DataFrame()
    all_contrib = ml_result.get("contributions")
    if isinstance(all_contrib, pd.DataFrame) and not all_contrib.empty:
        row = all_contrib[all_contrib["customer_id"].astype(str) == str(customer_id)]
        if not row.empty:
            vals = row.drop(columns="customer_id").iloc[0]
            factors = pd.DataFrame({"특성": vals.index, "기여도": vals.values}).sort_values("기여도", ascending=False)
            factors["특성"] = factors["특성"].str.replace(r"^(numeric|categorical)__", "", regex=True)
    orders = ml_result["orders"]
    customer_col = ml_result["customer_col"]
    customer_orders = orders[orders[customer_col].astype(str) == str(customer_id)].sort_values(ml_result["date_col"])
    return {"customer_id": customer_id, "repurchase_probability": score, "features": matched.iloc[0].to_dict(), "orders": customer_orders, "order_count": int(matched.iloc[0].get("order_count", 0)), "positive_factors": factors.head(5), "negative_factors": factors.tail(5).sort_values("기여도") if not factors.empty else factors}
