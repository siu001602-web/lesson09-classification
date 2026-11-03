import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="채점표 - 뇌졸중 예측 실습실",
    page_icon="📋",
    layout="wide"
)

st.title("📋 모델 성능 채점표 및 오예측 분석")
st.markdown("학습된 예측 모델의 혼동행렬, 성능 지표(정확도, 재현율, 정밀도, F1) 및 실제 오예측 목록을 확인합니다.")

# 2. 데이터 로드 (최상단 데이터 로드 및 캐싱)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df_raw = load_data()

# 3. 데이터 정렬 및 분할 ('분류 모델' 페이지와 동일한 규칙 적용)
# 번호(id) 순으로 정렬 후 10명씩 묶어, 앞 3명(index 0,1,2)을 테스트용, 나머지 7명을 훈련용으로 활용
df_sorted = df_raw.sort_values(by="id").reset_index(drop=True)

test_mask = (df_sorted.index % 10) < 3
train_df = df_sorted[~test_mask].copy()
test_df = df_sorted[test_mask].copy()

# 기본 속성 4개 고정: age, avg_glucose_level, hypertension, heart_disease
selected_cols = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

X_train = train_df[selected_cols].copy()
y_train = train_df['stroke'].values

X_test = test_df[selected_cols].copy()
y_test = test_df['stroke'].values

# 스케일링 (로지스틱 회귀용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 모델 학습 및 예측
# (1) 로지스틱 회귀 (확률로 답하는 모델)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)

# (2) 의사결정트리 (질문으로 답하는 모델)
dt_clf = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=5,
    random_state=42
)
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)

# (3) 다수결 모델 (훈련용에서 많은 쪽으로만 답하는 모델)
majority_class = int(pd.Series(y_train).mode()[0])
y_pred_base = np.full_like(y_test, fill_value=majority_class)

# 5. 혼동행렬 및 지표 계산 함수
def calc_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (tp + fp) == 0:
        precision_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"
    else:
        precision = tp / (tp + fp)
        precision_str = f"{precision:.4f}"
        if (precision + recall) == 0:
            f1_str = "계산할 수 없음"
        else:
            f1 = (2 * precision * recall) / (precision + recall)
            f1_str = f"{f1:.4f}"
            
    recall_str = f"{recall:.4f}"
    accuracy_str = f"{accuracy:.4f}"
    
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "acc": accuracy_str, "recall": recall_str,
        "precision": precision_str, "f1": f1_str
    }

m_log = calc_metrics(y_test, y_pred_log)
m_dt = calc_metrics(y_test, y_pred_dt)
m_base = calc_metrics(y_test, y_pred_base)

# 6. 상단 요약 정보
total_test = len(test_df)
stroke_test = int(np.sum(y_test == 1))
st.info(f"📌 **테스트 데이터 세트 정보**: 총 인원 **{total_test:,}명** 중 실제 뇌졸중 환자는 **{stroke_test}명**입니다.")

st.markdown("---")

# 7. 혼동행렬 (2x2 표) 표시
st.subheader("🟩🟨 2×2 혼동행렬 (Confusion Matrix)")
col_cm1, col_cm2 = st.columns(2)

def render_cm_table(title, m):
    row_stroke_1_total = m["tp"] + m["fn"]
    row_stroke_0_total = m["fp"] + m["tn"]
    col_pred_1_total = m["tp"] + m["fp"]
    col_pred_0_total = m["fn"] + m["tn"]
    total_all = row_stroke_1_total + row_stroke_0_total
    
    cm_df = pd.DataFrame({
        "예측: 뇌졸중(1)": [
            f"TP (참 양성): {m['tp']}명",
            f"FP (거짓 양성): {m['fp']}명",
            f"합계: {col_pred_1_total}명"
        ],
        "예측: 정상(0)": [
            f"FN (거짓 음성): {m['fn']}명",
            f"TN (참 음성): {m['tn']}명",
            f"합계: {col_pred_0_total}명"
        ],
        "합계": [
            f"{row_stroke_1_total}명",
            f"{row_stroke_0_total}명",
            f"{total_all}명"
        ]
    }, index=["실제: 뇌졸중(1)", "실제: 정상(0)", "합계"])
    
    st.markdown(f"#### {title}")
    st.table(cm_df)

with col_cm1:
    render_cm_table("로지스틱 회귀(확률로 답하는 모델)", m_log)

with col_cm2:
    render_cm_table("의사결정트리(질문으로 답하는 모델)", m_dt)

st.markdown("---")

# 8. 종합 평가 지표 표 나란히 비교
st.subheader("📊 성능 평가 지표 비교")

metrics_summary_df = pd.DataFrame({
    "모델": [
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)",
        "훈련용에서 많은 쪽으로만 답하는 모델"
    ],
    "정확도 (Accuracy)": [m_log["acc"], m_dt["acc"], m_base["acc"]],
    "재현율 (Recall)": [m_log["recall"], m_dt["recall"], m_base["recall"]],
    "정밀도 (Precision)": [m_log["precision"], m_dt["precision"], m_base["precision"]],
    "F1 점수 (F1-Score)": [m_log["f1"], m_dt["f1"], m_base["f1"]]
})

st.table(metrics_summary_df)

st.markdown("---")

# 9. 오예측 환자 명단 상세 비교 (놓친 환자 vs 헛짚은 사람)
st.subheader("🔍 오예측 데이터 분석")

selected_model_name = st.selectbox(
    "오예측 명단을 확인할 모델을 선택하세요:",
    ["로지스틱 회귀(확률로 답하는 모델)", "의사결정트리(질문으로 답하는 모델)"],
    key="eval_model_selector"
)

if selected_model_name == "로지스틱 회귀(확률로 답하는 모델)":
    pred_values = y_pred_log
    current_m = m_log
else:
    pred_values = y_pred_dt
    current_m = m_dt

# 테스트 데이터프레임에 예측값 반영
test_eval_df = test_df.copy()
test_eval_df['pred'] = pred_values

# 놓친 환자: 실제 뇌졸중(1)인데 예측 정상(0) -> FN
fn_df = test_eval_df[(test_eval_df['stroke'] == 1) & (test_eval_df['pred'] == 0)].copy()

# 헛짚은 사람: 실제 정상(0)인데 예측 뇌졸중(1) -> FP
fp_df = test_eval_df[(test_eval_df['stroke'] == 0) & (test_eval_df['pred'] == 1)].copy()

# 나이 내림차순 정렬
fn_df = fn_df.sort_values(by="age", ascending=False)
fp_df = fp_df.sort_values(by="age", ascending=False)

# 표시할 열 설정 및 한글 이름 변경
display_cols = ["id", "age", "avg_glucose_level", "hypertension", "heart_disease"]
korean_col_names = {
    "id": "번호",
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부"
}

fn_display = fn_df[display_cols].rename(columns=korean_col_names)
fp_display = fp_df[display_cols].rename(columns=korean_col_names)

col_list1, col_list2 = st.columns(2)

with col_list1:
    st.markdown(f"### 🚨 놓친 환자 [총 {len(fn_df)}명]")
    if len(fn_display) > 20:
        st.caption("※ 인원이 20명을 초과하여 나이가 많은 상위 20명만 표시합니다.")
        st.dataframe(fn_display.head(20), use_container_width=True, hide_index=True)
    else:
        st.dataframe(fn_display, use_container_width=True, hide_index=True)

with col_list2:
    st.markdown(f"### ⚠️ 헛짚은 사람 [총 {len(fp_df)}명]")
    if len(fp_display) > 20:
        st.caption("※ 인원이 20명을 초과하여 나이가 많은 상위 20명만 표시합니다.")
        st.dataframe(fp_display.head(20), use_container_width=True, hide_index=True)
    else:
        st.dataframe(fp_display, use_container_width=True, hide_index=True)

st.markdown(
    f"💡 **참고**: 선택하신 **{selected_model_name}** 기준 '놓친 환자'의 전체 인원수(**{len(fn_df)}명**)는 "
    f"위 혼동행렬의 **FN 칸({current_m['fn']}명)**과 같고, "
    f"'헛짚은 사람'의 전체 인원수(**{len(fp_df)}명**)는 **FP 칸({current_m['fp']}명)**과 정확히 일치합니다."
)
