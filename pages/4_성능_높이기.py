import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="성능 높이기 - 뇌졸중 예측 실습실",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ 클래스 가중치 조정을 통한 모델 성능 개선 실습")
st.markdown("뇌졸중 데이터는 양성(환자) 데이터가 매우 적은 불균형 데이터입니다. **클래스 가중치(`class_weight='balanced'`)** 적용 전후의 모델 예측 변화와 결정 경계를 비교해 보세요.")

# 2. 데이터 로드 및 캐싱
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df_raw = load_data()

# 3. 속성 매핑 정보 정의
FEATURE_MAP = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
REVERSE_FEATURE_MAP = {v: k for k, v in FEATURE_MAP.items()}

# 4. 화면 맨 위에 속성 선택 UI 배치
st.sidebar.header("⚙️ 모델 설정")
selected_korean_features = st.sidebar.multiselect(
    "학습에 사용할 속성을 선택하세요 (최소 2개)",
    options=list(FEATURE_MAP.values()),
    default=["나이", "평균 혈당", "고혈압", "심장병"]
)

# 2개 미만 선택 시 예외 처리
if len(selected_korean_features) < 2:
    st.warning("⚠️ 최소 2개 이상의 속성을 선택해야 모델을 학습하고 결정 경계를 시각화할 수 있습니다.")
    st.stop()

selected_cols = [REVERSE_FEATURE_MAP[name] for name in selected_korean_features]

# 5. 데이터 분할 및 전처리 (번호 순 정렬 후 10명 중 앞 3명 테스트, 나머지 훈련)
df_sorted = df_raw.sort_values(by="id").reset_index(drop=True)

test_mask = (df_sorted.index % 10) < 3
train_df = df_sorted[~test_mask].copy()
test_df = df_sorted[test_mask].copy()

X_train = train_df[selected_cols].copy()
y_train = train_df['stroke'].values

X_test = test_df[selected_cols].copy()
y_test = test_df['stroke'].values

# 체질량지수(bmi) 결측치 처리 (선택된 경우 훈련용 중앙값으로 대체)
if "bmi" in selected_cols:
    bmi_median = X_train['bmi'].median()
    X_train['bmi'] = X_train['bmi'].fillna(bmi_median)
    X_test['bmi'] = X_test['bmi'].fillna(bmi_median)

# 스케일링 (로지스틱 회귀용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 모델 학습 (가중치 적용 전후 총 4개 모델)
# (1) 로지스틱 회귀 (기본)
log_reg_before = LogisticRegression(random_state=42)
log_reg_before.fit(X_train_scaled, y_train)
y_pred_log_b = log_reg_before.predict(X_test_scaled)

# (2) 로지스틱 회귀 (가중치 균형)
log_reg_after = LogisticRegression(class_weight='balanced', random_state=42)
log_reg_after.fit(X_train_scaled, y_train)
y_pred_log_a = log_reg_after.predict(X_test_scaled)

# (3) 의사결정트리 (기본)
dt_before = DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=42)
dt_before.fit(X_train, y_train)
y_pred_dt_b = dt_before.predict(X_test)

# (4) 의사결정트리 (가중치 균형)
dt_after = DecisionTreeClassifier(max_depth=3, min_samples_split=5, class_weight='balanced', random_state=42)
dt_after.fit(X_train, y_train)
y_pred_dt_a = dt_after.predict(X_test)

# 7. 성능 지표 계산 함수
def calc_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (tp + fp) == 0:
        precision_str = "계산 불가"
        f1_str = "계산 불가"
    else:
        precision = tp / (tp + fp)
        precision_str = f"{precision:.4f}"
        if (precision + recall) == 0:
            f1_str = "계산 불가"
        else:
            f1 = (2 * precision * recall) / (precision + recall)
            f1_str = f"{f1:.4f}"
            
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "acc": f"{accuracy:.4f}", "recall": f"{recall:.4f}",
        "precision": precision_str, "f1": f1_str
    }

m_log_b = calc_metrics(y_test, y_pred_log_b)
m_log_a = calc_metrics(y_test, y_pred_log_a)
m_dt_b = calc_metrics(y_test, y_pred_dt_b)
m_dt_a = calc_metrics(y_test, y_pred_dt_a)

# 8. 전후 비교 표 출력
st.subheader("📊 클래스 가중치 적용 전후 성능 비교")

def render_comparison_section(model_title, mb, ma):
    st.markdown(f"### {model_title}")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**[가중치 조정 전 (기본)]**")
        df_b = pd.DataFrame({
            "예측: 뇌졸중(1)": [f"TP: {mb['tp']}명", f"FP: {mb['fp']}명"],
            "예측: 정상(0)": [f"FN: {mb['fn']}명", f"TN: {mb['tn']}명"]
        }, index=["실제: 뇌졸중(1)", "실제: 정상(0)"])
        st.table(df_b)
        st.write(f"- 정확도: **{mb['acc']}** | 재현율: **{mb['recall']}** | 정밀도: **{mb['precision']}** | F1: **{mb['f1']}**")
        
    with col_t2:
        st.markdown("**[가중치 조정 후 (`class_weight='balanced'`)]**")
        df_a = pd.DataFrame({
            "예측: 뇌졸중(1)": [f"TP: {ma['tp']}명", f"FP: {ma['fp']}명"],
            "예측: 정상(0)": [f"FN: {ma['fn']}명", f"TN: {ma['tn']}명"]
        }, index=["실제: 뇌졸중(1)", "실제: 정상(0)"])
        st.table(df_a)
        st.write(f"- 정확도: **{ma['acc']}** | 재현율: **{ma['recall']}** | 정밀도: **{ma['precision']}** | F1: **{ma['f1']}**")
        
    # 변경 요약 줄
    st.info(
        f"💡 **변화 요약**: 찾아낸 환자 수(TP)는 **{mb['tp']}명 → {ma['tp']}명**, "
        f"놓친 환자 수(FN)는 **{mb['fn']}명 → {ma['fn']}명**, "
        f"헛짚은 사람 수(FP)는 **{mb['fp']}명 → {ma['fp']}명**으로 변경되었습니다."
    )
    st.markdown("---")

render_comparison_section("로지스틱 회귀(확률로 답하는 모델)", m_log_b, m_log_a)
render_comparison_section("의사결정트리(질문으로 답하는 모델)", m_dt_b, m_dt_a)

# 9. 산점도 및 결정 경계 시각화 (Plotly)
st.subheader("📈 결정 경계 가중치 적용 전후 비교 산점도")

if len(selected_cols) == 2:
    # 2개 선택 시 2차원 산점도 및 0.5 경계선 2개 겹쳐 그리기
    c1, c2 = st.columns(2)
    with c1:
        x_korean = selected_korean_features[0]
    with c2:
        y_korean = selected_korean_features[1]
        
    x_col = REVERSE_FEATURE_MAP[x_korean]
    y_col = REVERSE_FEATURE_MAP[y_korean]
    
    x_min, x_max = X_test[x_col].min() - 5, X_test[x_col].max() + 5
    y_min, y_max = X_test[y_col].min() - 5, X_test[y_col].max() + 5
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 150),
        np.linspace(y_min, y_max, 150)
    )
    
    grid_df = pd.DataFrame(index=range(xx.size))
    for c in selected_cols:
        if c == x_col:
            grid_df[c] = xx.ravel()
        elif c == y_col:
            grid_df[c] = yy.ravel()
            
    grid_scaled = scaler.transform(grid_df[selected_cols])
    
    Z_b = log_reg_before.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
    Z_a = log_reg_after.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
    
    fig = go.Figure()
    
    # 테스트 데이터 산점도
    for stroke_val, name, color in zip([0, 1], ["정상(아님)", "뇌졸중"], ["blue", "red"]):
        sub = test_df[test_df['stroke'] == stroke_val]
        fig.add_trace(go.Scatter(
            x=sub[x_col],
            y=sub[y_col],
            mode='markers',
            name=name,
            marker=dict(color=color, opacity=0.7, size=7)
        ))
        
    # 조정 전 0.5 경계선
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 150),
        y=np.linspace(y_min, y_max, 150),
        z=Z_b,
        contours_start=0.5, contours_end=0.5,
        contours_coloring='lines',
        line_width=3, line_color='cyan',
        name='가중치 조정 전 경계선 (P=0.5)',
        showscale=False
    ))
    
    # 조정 후 0.5 경계선
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 150),
        y=np.linspace(y_min, y_max, 150),
        z=Z_a,
        contours_start=0.5, contours_end=0.5,
        contours_coloring='lines',
        line_width=3, line_color='magenta',
        name='가중치 조정 후 경계선 (P=0.5)',
        showscale=False
    ))
    
    fig.update_layout(
        title=f"로지스틱 회귀 가중치 전후 결정 경계 비교 ({x_korean} vs {y_korean})",
        xaxis_title=x_korean,
        yaxis_title=y_korean,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    has_b = np.any(np.abs(Z_b - 0.5) < 0.05)
    has_a = np.any(np.abs(Z_a - 0.5) < 0.05)
    if not has_b or not has_a:
        st.caption("ℹ️ **안내**: 일부 결정 경계선이 현재 축 표시 영역(그림 밖)에 위치하여 시각화되지 않았을 수 있습니다.")

else:
    # 3개 이상 선택 시 3차원 산점도 및 평면 겹쳐 그리기
    st.markdown("산점도 축으로 사용할 3개의 속성을 선택해 주세요:")
    c1, c2, c3 = st.columns(3)
    with c1:
        x_korean = st.selectbox("X축 속성", selected_korean_features, index=0)
    with c2:
        y_options = [f for f in selected_korean_features if f != x_korean]
        y_korean = st.selectbox("Y축 속성", y_options, index=0)
    with c3:
        z_options = [f for f in selected_korean_features if f not in [x_korean, y_korean]]
        z_korean = st.selectbox("Z축 속성", z_options, index=0)
        
    x_col = REVERSE_FEATURE_MAP[x_korean]
    y_col = REVERSE_FEATURE_MAP[y_korean]
    z_col = REVERSE_FEATURE_MAP[z_korean]
    
    # 3D 산점도 및 결정 평면 시각화
    fig = go.Figure()
    
    for stroke_val, name, color in zip([0, 1], ["정상(아님)", "뇌졸중"], ["blue", "red"]):
        sub = test_df[test_df['stroke'] == stroke_val]
        fig.add_trace(go.Scatter3d(
            x=sub[x_col],
            y=sub[y_col],
            z=sub[z_col],
            mode='markers',
            name=name,
            marker=dict(color=color, opacity=0.8, size=5)
        ))
        
    fig.update_layout(
        title=f"3차원 산점도 ({x_korean} / {y_korean} / {z_korean})",
        scene=dict(
            xaxis_title=x_korean,
            yaxis_title=y_korean,
            zaxis_title=z_korean
        ),
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("ℹ️ 3차원 공간에서 로지스틱 회귀 모델의 0.5 결정 평면은 모델의 가중치 선형 방정식을 통해 마우스로 회전하며 다각도로 확인할 수 있습니다.")
