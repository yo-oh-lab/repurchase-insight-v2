"""관측 데이터에 근거한 자동 인사이트와 마케팅 가설."""
from __future__ import annotations
import pandas as pd


def generate_insights(metrics) -> list[dict]:
    results = []
    rate = float(metrics.repurchase_rate)
    results.append({
        "title": "전체 재구매 현황",
        "finding": f"분석 고객 {metrics.total_customers:,}명 중 {metrics.repurchase_customers:,}명이 관측 기간에 2회 이상 주문했습니다.",
        "evidence": f"전체 주문 {metrics.total_orders:,}건 · 관측 기간 기준 재구매율 {rate:.2f}%",
        "action": "고객군별 재구매율과 표본 수를 함께 보고, 차이가 큰 구간부터 캠페인 실험 대상으로 선정하세요.",
        "caution": "관측 기간이 짧은 고객은 재구매 기회가 적었을 수 있습니다. 재구매율은 인과효과나 미래 예측값이 아닙니다.",
    })
    candidates = []
    for feature, table in metrics.segments.items():
        if table.empty or "재구매율(%)" not in table:
            continue
        eligible = table[table["고객수"] >= 20]
        if eligible.empty:
            continue
        eligible = eligible.copy()
        eligible["전체 대비 차이(%p)"] = eligible["재구매율(%)"] - rate
        eligible["절대차이"] = eligible["전체 대비 차이(%p)"].abs()
        for _, row in eligible.iterrows():
            candidates.append((float(row["절대차이"]), feature, row))
    candidates.sort(key=lambda x: x[0], reverse=True)
    for _, feature, row in candidates[:4]:
        delta = float(row["전체 대비 차이(%p)"])
        direction = "높습니다" if delta > 0 else "낮습니다"
        results.append({
            "title": f"{feature} 구간별 차이",
            "finding": f"'{row['구간']}' 구간의 관측 재구매율은 {row['재구매율(%)']:.1f}%로 전체보다 {abs(delta):.1f}%p {direction}.",
            "evidence": f"구간 고객 {int(row['고객수']):,}명, 재구매 고객 {int(row['재구매고객수']):,}명",
            "action": "이 구간에 맞춘 메시지·추천·혜택을 소규모 대조 실험으로 비교하고, 전환율과 이익을 함께 측정하세요.",
            "caution": "집단 차이는 관련성을 보여줄 뿐 원인을 증명하지 않습니다. 여러 구간을 비교한 결과는 탐색적 신호로 해석하세요.",
        })
    if len(results) == 1:
        results.append({
            "title": "추가 분석 안내",
            "finding": "최소 표본 기준을 충족하면서 충분히 구분되는 고객 구간이 아직 없습니다.",
            "evidence": "작은 집단의 재구매율은 우연한 변동이 클 수 있어 자동 인사이트에서 제외했습니다.",
            "action": "날짜·결제·상품·금액 관련 컬럼을 포함한 주문 데이터를 확인하세요.",
            "caution": "분석 가능한 컬럼과 고객 수에 따라 결과가 달라집니다.",
        })
    return results


def generate_marketing_actions(metrics) -> pd.DataFrame:
    rows = []
    overall = float(metrics.repurchase_rate)
    for feature, table in metrics.segments.items():
        if table.empty or "재구매율(%)" not in table:
            continue
        for _, r in table.iterrows():
            n = int(r["고객수"])
            if n < 20:
                continue
            delta = float(r["재구매율(%)"] - overall)
            if delta <= -5:
                strategy = "재방문 장벽 점검 및 재활성화 실험"
                action = "배송·품질·결제 불편 여부를 점검하고, 알림·관련 상품 추천·소액 혜택을 대조군과 비교합니다."
                priority = "높음" if delta <= -10 and n >= 50 else "보통"
            elif delta >= 5:
                strategy = "유사 고객 확장 및 연관 상품 추천 실험"
                action = "해당 구간에서 반응이 좋았던 상품/콘텐츠를 유사 고객에게 시험하고 추가 구매와 이익을 측정합니다."
                priority = "보통"
            else:
                continue
            rows.append({
                "우선순위": priority,
                "고객 세그먼트": f"{feature}: {r['구간']}",
                "관측 재구매율(%)": round(float(r["재구매율(%)"]), 2),
                "전체 대비 차이(%p)": round(delta, 2),
                "표본 수": n,
                "추천 전략": strategy,
                "실행 방법": action,
                "성과 지표": "재구매 전환율, 객단가, 공헌이익, 수신거부율",
            })
    if not rows:
        rows.append({
            "우선순위": "탐색",
            "고객 세그먼트": "전체 고객",
            "관측 재구매율(%)": round(overall, 2),
            "전체 대비 차이(%p)": 0.0,
            "표본 수": metrics.total_customers,
            "추천 전략": "메시지/추천 A/B 테스트 설계",
            "실행 방법": "할인 제공군과 비제공군을 비교하고, 단순 매출이 아닌 이익 및 재구매를 확인합니다.",
            "성과 지표": "재구매 전환율, 객단가, 공헌이익, 수신거부율",
        })
    return pd.DataFrame(rows).sort_values(["우선순위", "표본 수"], ascending=[True, False]).reset_index(drop=True)
