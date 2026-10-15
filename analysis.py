"""범용 주문 데이터 재구매 분석 모듈."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

@dataclass
class RepurchaseMetrics:
    total_orders: int
    total_customers: int
    repurchase_customers: int
    repurchase_rate: float
    order_distribution: pd.DataFrame
    segments: dict[str, pd.DataFrame]
    first_orders: pd.DataFrame
    order_table: pd.DataFrame
    warnings: list[str]


def prepare_orders(df: pd.DataFrame, order_col: str, customer_col: str, date_col: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """주문행/상품행 데이터를 주문 단위로 정리한다. 내부 표준 컬럼은 order_id/customer_id/order_date."""
    if order_col not in df or customer_col not in df:
        raise ValueError("주문 ID와 고객 ID 컬럼을 선택해야 합니다.")
    x = df.copy()
    x = x.rename(columns={order_col: "order_id", customer_col: "customer_id"})
    warnings = []
    if date_col and date_col in x.columns:
        x = x.rename(columns={date_col: "order_date"})
        x["order_date"] = pd.to_datetime(x["order_date"], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        x["order_date"] = pd.NaT
        warnings.append("날짜 컬럼이 없어 최초 주문 시점 기반 분석과 시간순 분석은 제한됩니다.")
    x = x.dropna(subset=["order_id", "customer_id"]).copy()
    x["order_id"] = x["order_id"].astype(str)
    x["customer_id"] = x["customer_id"].astype(str)
    if x.empty:
        raise ValueError("주문 ID와 고객 ID가 모두 존재하는 행이 없습니다.")

    # 숫자형 값은 상품행 합산, 범주형 값은 주문 안의 고유값을 합쳐 주문 단위로 만든다.
    exclude = {"order_id", "customer_id", "order_date"}
    numeric_cols = [c for c in x.columns if c not in exclude and pd.api.types.is_numeric_dtype(x[c])]
    # 숫자로 저장된 문자열(금액 등)도 일정 비율 이상 변환되면 숫자로 처리
    for c in x.columns:
        if c in exclude or c in numeric_cols:
            continue
        converted = pd.to_numeric(x[c], errors="coerce")
        non_null = x[c].notna().sum()
        if non_null and converted.notna().sum() / non_null >= 0.85:
            x[c] = converted
            numeric_cols.append(c)
    categorical_cols = [c for c in x.columns if c not in exclude and c not in numeric_cols]
    aggregations = {"customer_id": "first", "order_date": "min"}
    for c in numeric_cols:
        aggregations[c] = "sum"
    for c in categorical_cols:
        aggregations[c] = lambda s: " / ".join(pd.Series(s.dropna().astype(str).unique()).head(5)) if s.notna().any() else np.nan
    orders = x.groupby("order_id", as_index=False, dropna=False).agg(aggregations)
    # 동일 주문 ID에 고객이 여러 명 매핑된 경우 데이터 오류로 안내
    customer_counts = x.groupby("order_id")["customer_id"].nunique()
    if (customer_counts > 1).any():
        warnings.append(f"고객 ID가 여러 개 연결된 주문 {int((customer_counts > 1).sum())}건이 있어 첫 고객 ID를 사용했습니다.")
    orders = orders.dropna(subset=["customer_id"]).sort_values("order_date", na_position="last").reset_index(drop=True)
    if orders["order_id"].duplicated().any():
        orders = orders.drop_duplicates("order_id", keep="first")
    return orders, warnings


def _safe_bins(series: pd.Series, max_bins: int = 5) -> pd.Series:
    """이상치에 덜 민감한 중복 없는 분위 구간. 표본이 적거나 값이 상수면 구간화를 생략한다."""
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    out = pd.Series(index=series.index, dtype="object")
    if valid.nunique() < 2 or len(valid) < 10:
        out.loc[valid.index] = "전체 값"
        return out
    q = min(max_bins, max(2, int(valid.nunique() ** 0.5)), 5)
    try:
        bins = pd.qcut(valid, q=q, duplicates="drop")
        labels = []
        for interval in bins.cat.categories:
            labels.append(f"{interval.left:,.2f} ~ {interval.right:,.2f}")
        mapped = bins.cat.rename_categories(labels).astype("object")
        out.loc[valid.index] = mapped
    except (ValueError, TypeError):
        out.loc[valid.index] = "전체 값"
    return out


def _segment_table(first: pd.DataFrame, col: str, max_categories: int = 12) -> pd.DataFrame:
    tmp = first[[col, "repurchased"]].copy().dropna(subset=[col])
    if tmp.empty:
        return pd.DataFrame()
    if pd.api.types.is_numeric_dtype(tmp[col]):
        tmp["구간"] = _safe_bins(tmp[col])
    else:
        values = tmp[col].astype(str).str.strip().replace("", "미상")
        top = values.value_counts().head(max_categories).index
        tmp["구간"] = values.where(values.isin(top), "기타")
    tmp = tmp.dropna(subset=["구간"])
    if tmp.empty:
        return pd.DataFrame()
    table = tmp.groupby("구간", observed=True).agg(고객수=("repurchased", "size"), 재구매고객수=("repurchased", "sum"), 재구매율=("repurchased", "mean")).reset_index()
    table["재구매율(%)"] = (table["재구매율"] * 100).round(2)
    table = table.drop(columns="재구매율")
    # 카이제곱 검정은 2x2 이상 교차표가 성립할 때만 계산
    contingency = pd.crosstab(tmp["구간"], tmp["repurchased"])
    p = np.nan
    if contingency.shape[0] >= 2 and contingency.shape[1] == 2 and (contingency.sum(axis=0) > 0).all():
        try:
            _, p, _, _ = chi2_contingency(contingency)
        except ValueError:
            pass
    table["p-value"] = round(float(p), 5) if pd.notna(p) else np.nan
    table["통계적 유의성(5%)"] = bool(p < 0.05) if pd.notna(p) else False
    return table.sort_values("고객수", ascending=False).reset_index(drop=True)


def calculate_repurchase_kpis(df: pd.DataFrame, order_col: str, customer_col: str, date_col: str | None = None, min_group_size: int = 20) -> RepurchaseMetrics:
    orders, warnings = prepare_orders(df, order_col, customer_col, date_col)
    counts = orders.groupby("customer_id")["order_id"].nunique()
    repurchased_ids = set(counts[counts >= 2].index)
    rate = 100 * len(repurchased_ids) / len(counts) if len(counts) else 0.0
    distribution = counts.value_counts().sort_index().rename_axis("주문 횟수").reset_index(name="고객 수")
    distribution["비중(%)"] = (distribution["고객 수"] / max(len(counts), 1) * 100).round(2)
    first_orders = orders.sort_values("order_date", na_position="last").drop_duplicates("customer_id", keep="first").copy()
    first_orders["repurchased"] = first_orders["customer_id"].isin(repurchased_ids).astype(int)
    # 1회 구매 고객도 충분한 재구매 관측 기간이 없을 수 있으므로 해석 주의
    if first_orders["order_date"].notna().sum() == 0:
        warnings.append("날짜 정보가 없어 첫 주문은 파일에 등장한 순서가 아닌 주문 ID 그룹의 첫 행 기준입니다.")
    candidate_cols = [c for c in first_orders.columns if c not in {"order_id", "customer_id", "order_date", "repurchased"}]
    segments = {}
    for col in candidate_cols:
        if first_orders[col].notna().sum() < min_group_size:
            continue
        if first_orders[col].nunique(dropna=True) < 2:
            continue
        table = _segment_table(first_orders, col)
        if not table.empty:
            # 지나치게 작은 구간은 결과표에서 제외
            table = table[table["고객수"] >= max(5, min_group_size // 2)].reset_index(drop=True)
            if not table.empty:
                segments[col] = table
    return RepurchaseMetrics(
        total_orders=int(orders["order_id"].nunique()),
        total_customers=int(len(counts)),
        repurchase_customers=int(len(repurchased_ids)),
        repurchase_rate=float(rate),
        order_distribution=distribution,
        segments=segments,
        first_orders=first_orders,
        order_table=orders,
        warnings=warnings,
    )
