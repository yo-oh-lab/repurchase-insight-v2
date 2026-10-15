from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from analysis import calculate_repurchase_kpis
from insights import generate_insights, generate_marketing_actions
from ml_model import train_repurchase_model, predict_customer_repurchase

st.set_page_config(page_title="범용 재구매 분석 대시보드", page_icon="📊", layout="wide")
st.title("📊 재구매 분석 대시보드")
st.caption("CSV 컬럼을 자동으로 추정하고, 필요하면 직접 지정해 분석합니다. 관측된 차이와 예측 결과를 구분합니다.")

@st.cache_data(show_spinner=False)
def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp949", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV 인코딩을 판별하지 못했습니다.")

def guess_column(columns, tokens):
    lowered = {c: str(c).lower().replace(" ", "_").replace("-", "_") for c in columns}
    for token in tokens:
        for c, name in lowered.items():
            if token in name:
                return c
    return "없음"

def choose(label, columns, guess, help_text=None):
    options = ["없음"] + list(columns)
    index = options.index(guess) if guess in options else 0
    return st.selectbox(label, options, index=index, help=help_text)

uploaded = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
if uploaded is None:
    local_files = sorted(Path(__file__).parent.glob("*.csv"))
    if local_files:
        chosen = st.sidebar.selectbox("또는 프로젝트 폴더의 CSV", [p.name for p in local_files])
        try:
            df = pd.read_csv(Path(__file__).parent / chosen, low_memory=False)
            data_name = chosen
        except UnicodeDecodeError:
            df = pd.read_csv(Path(__file__).parent / chosen, encoding="cp949", low_memory=False)
            data_name = chosen
    else:
        st.info("왼쪽 메뉴에서 CSV 파일을 업로드해 주세요.")
        st.stop()
else:
    try:
        df = read_csv_bytes(uploaded.getvalue())
        data_name = uploaded.name
    except Exception as exc:
        st.error(f"CSV를 읽지 못했습니다: {exc}")
        st.stop()

if df.empty:
    st.error("CSV에 분석할 행이 없습니다.")
    st.stop()
# duplicate column labels can make pandas mapping ambiguous
if df.columns.duplicated().any():
    df = df.loc[:, ~df.columns.duplicated()].copy()
columns = list(df.columns)
st.sidebar.success(f"데이터 로드 완료: {data_name}")
st.sidebar.caption(f"행 {len(df):,}개 · 컬럼 {len(columns):,}개")

st.sidebar.subheader("분석 컬럼 설정")
order_guess = guess_column(columns, ["order_id", "orderid", "transaction_id", "transactionid", "주문번호", "주문_id", "거래번호", "주문"])
customer_guess = guess_column(columns, ["customer_unique_id", "customer_id", "customerid", "user_id", "userid", "회원번호", "고객번호", "고객_id", "사용자_id", "고객"])
date_guess = guess_column(columns, ["order_purchase_timestamp", "order_date", "purchase_date", "transaction_date", "created_at", "주문일", "구매일", "결제일", "날짜", "timestamp"])
order_col = choose("주문 ID 컬럼", columns, order_guess, "한 주문을 고유하게 식별하는 컬럼입니다. 상품 행 단위 CSV도 주문 ID가 필요합니다.")
customer_col = choose("고객 ID 컬럼", columns, customer_guess, "고객을 반복해서 식별할 수 있는 ID가 필요합니다. 고객 ID가 없으면 재구매율을 계산할 수 없습니다.")
date_col = choose("주문 날짜 컬럼(선택)", columns, date_guess, "날짜는 첫 주문 기준 비교와 향후 90일 예측에 필요합니다.")
if order_col == "없음" or customer_col == "없음":
    st.warning("재구매 분석을 위해 주문 ID와 고객 ID 컬럼을 지정해 주세요. 컬럼명이 다르면 위 선택 메뉴에서 직접 고르면 됩니다.")
    st.subheader("CSV 미리보기")
    st.dataframe(df.head(30), use_container_width=True, hide_index=True)
    st.stop()

try:
    metrics = calculate_repurchase_kpis(df, order_col, customer_col, None if date_col == "없음" else date_col)
except Exception as exc:
    st.error(f"분석을 진행할 수 없습니다: {exc}")
    st.write("현재 컬럼:", columns)
    st.stop()

for warning in metrics.warnings:
    st.warning(warning)

c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 주문 수", f"{metrics.total_orders:,}")
c2.metric("전체 고객 수", f"{metrics.total_customers:,}")
c3.metric("재구매 고객 수", f"{metrics.repurchase_customers:,}")
c4.metric("관측 재구매율", f"{metrics.repurchase_rate:.2f}%")
st.caption("관측 재구매율 = 파일 내에서 2회 이상 주문한 고객 비율입니다. 관측 기간이 짧은 고객은 재구매 기회가 적을 수 있습니다.")

st.divider()
st.header("1. 전체 재구매 분석")
left, right = st.columns(2)
with left:
    st.subheader("고객별 주문 횟수")
    st.dataframe(metrics.order_distribution, use_container_width=True, hide_index=True)
    if not metrics.order_distribution.empty:
        st.plotly_chart(px.bar(metrics.order_distribution, x="주문 횟수", y="고객 수", title="주문 횟수별 고객 수"), use_container_width=True)
with right:
    st.subheader("자동 분석 인사이트")
    for item in generate_insights(metrics):
        with st.container(border=True):
            st.markdown(f"**{item['title']}**")
            st.write(item["finding"])
            st.caption(f"근거: {item['evidence']}")
            st.write(f"실행 제안: {item['action']}")
            st.caption(f"주의: {item['caution']}")

st.subheader("첫 주문 특성별 실제 재구매율")
if metrics.segments:
    labels = {c: c for c in metrics.segments}
    selected_feature = st.selectbox("분석할 특성", list(labels), format_func=lambda c: labels[c])
    segment_table = metrics.segments[selected_feature]
    st.dataframe(segment_table, use_container_width=True, hide_index=True)
    if not segment_table.empty:
        st.plotly_chart(px.bar(segment_table, x="구간", y="재구매율(%)", hover_data=["고객수", "재구매고객수"], title=f"{selected_feature} 구간별 재구매율"), use_container_width=True)
else:
    st.info("표본 수와 값의 다양성을 충족하는 특성별 구간이 없습니다. 더 많은 고객 또는 구매 특성 컬럼이 있는 데이터를 사용해 보세요.")

st.subheader("세그먼트별 마케팅 실행안")
actions = generate_marketing_actions(metrics)
st.dataframe(actions, use_container_width=True, hide_index=True)
st.caption("마케팅 제안은 관측된 집단 차이에 기반한 실험 가설입니다. 실행 후 대조군과 비교해 실제 효과를 검증하세요.")

st.divider()
st.header("2. 향후 90일 재구매 예측(선택 기능)")
if date_col == "없음":
    st.info("예측 기능은 날짜 컬럼이 필요합니다. 왼쪽 메뉴에서 날짜 컬럼을 선택해 주세요.")
else:
    st.write("과거 기준 시점에서 향후 90일 이내 재구매 여부를 구성해 로지스틱 회귀 모델을 평가합니다. 데이터 기간과 고객 수가 부족하면 실행되지 않을 수 있습니다.")
    if st.button("머신러닝 모델 학습 및 예측 준비", type="primary"):
        try:
            with st.spinner("시점별 학습 데이터를 만들고 모델을 평가하고 있습니다."):
                st.session_state["generic_ml_result"] = train_repurchase_model(df, order_col, customer_col, date_col)
            st.success("모델 준비가 완료되었습니다.")
        except Exception as exc:
            st.session_state.pop("generic_ml_result", None)
            st.error(f"모델 준비 실패: {exc}")
    ml_result = st.session_state.get("generic_ml_result")
    if ml_result:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("정확도", f"{ml_result['accuracy']:.3f}")
        m2.metric("ROC-AUC", "계산 불가" if pd.isna(ml_result["roc_auc"]) else f"{ml_result['roc_auc']:.3f}")
        m3.metric("Brier score", f"{ml_result['brier_score']:.3f}")
        m4.metric("평가 표본", f"{ml_result['test_rows']:,}")
        st.caption("시간순 검증 지표이며, 데이터 기간·표본 구성에 따라 달라집니다. 예측은 구매나 마케팅 효과를 보장하지 않습니다.")
        st.subheader("전체 모델의 주요 특성")
        st.caption("절대계수는 모델 전체에서 각 변수가 가진 상대적 가중치 크기입니다. 영향 방향과 고객별 기여도를 뜻하지는 않습니다.")
        global_importance = ml_result.get("global_importance", pd.DataFrame())
        if isinstance(global_importance, pd.DataFrame) and not global_importance.empty:
            st.dataframe(global_importance, use_container_width=True, hide_index=True)
        ids = ml_result["latest_rows"]["customer_id"].astype(str).tolist()
        selected_customer = st.selectbox("예측할 고객 ID", ids)
        if st.button("선택 고객 예측"):
            prediction = predict_customer_repurchase(ml_result, selected_customer)
            if prediction is None:
                st.warning("선택 고객을 찾지 못했습니다.")
            else:
                st.session_state["generic_customer_prediction"] = prediction
        prediction = st.session_state.get("generic_customer_prediction")
        if prediction and str(prediction.get("customer_id")) == str(selected_customer):
            st.metric("향후 90일 재구매 예측 확률", f"{prediction['repurchase_probability']:.1%}")
            st.write(f"모델에 기록된 주문 수: {prediction['order_count']}")
            st.caption("고객별 기여도는 모델 점수에 대한 선형 기여도이며, 인과관계나 행동의 원인을 의미하지 않습니다.")
            if not prediction["positive_factors"].empty:
                st.write("점수를 높이는 방향의 모델 기여")
                st.dataframe(prediction["positive_factors"], use_container_width=True, hide_index=True)
            if not prediction["negative_factors"].empty:
                st.write("점수를 낮추는 방향의 모델 기여")
                st.dataframe(prediction["negative_factors"], use_container_width=True, hide_index=True)

with st.expander("분석에 사용된 주문 단위 데이터 미리보기"):
    st.dataframe(metrics.order_table.head(100), use_container_width=True, hide_index=True)
