import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="분류 모델 - 뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 뇌졸중 예측 분류 모델 실습")
st.markdown("머신러닝 모델을 직접 학습시키고, 테스트 데이터에 대한 예측 성능 및 결정 경계를 확인해 보세요.")

# 2. 데이터 로드 및 최상단 캐싱
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

# 사이드바: 속성 선택 UI
st.sidebar.header("⚙️ 모델 설정")
selected_korean_features = st.sidebar.multiselect(
    "학습에 사용할 속성을 선택하세요 (최소 2개)",
    options=list(FEATURE_MAP.values()),
    default=["나이", "평균 혈당", "고혈압", "심장병"]
)

# 2개 미만 선택 시 예외 처리
if len(selected_korean_features) < 2:
    st.warning("⚠️ 최소 2개 이상의 속성을 선택해야 모델을 학습하고 2차원 결정 경계를 그릴 수 있습니다.")
    st.stop()

selected_cols = [REVERSE_FEATURE_MAP[name] for name in selected_korean_features]

# 4. 데이터 분할 및 전처리 (요구사항 고정 규칙 반영)
# - 번호(id) 순으로 정렬 후 10명씩 묶어 앞 3명을 테스트용, 나머지 7명을 훈련용으로 분할
df_sorted = df_raw.sort_values(by="id").reset_index(drop=True)

# index % 10 이 0, 1, 2 이면 테스트용 (30%), 3~9 이면 훈련용 (70%)
test_mask = (df_sorted.index % 10) < 3
train_df = df_sorted[~test_mask].copy()
test_df = df_sorted[test_mask].copy()

X_train = train_df[selected_cols].copy()
y_train = train_df['stroke'].values

X_test = test_df[selected_cols].copy()
y_test = test_df['stroke'].values

# BMI 결측치 처리 (BMI가 선택된 경우 훈련용 데이터의 중앙값으로 채움)
if "bmi" in selected_cols:
    bmi_median = X_train['bmi'].median()
    X_train['bmi'] = X_train['bmi'].fillna(bmi_median)
    X_test['bmi'] = X_test['bmi'].fillna(bmi_median)

# 스케일링 (로지스틱 회귀 및 시각화용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 모델 학습
# (1) 다수결 사전확률 모델 (입력을 전혀 보지 않고 훈련용 다수 클래스로만 예측)
majority_class = int(pd.Series(y_train).mode()[0])
baseline_train_acc = (y_train == majority_class).mean()
baseline_test_acc = (y_test == majority_class).mean()

# (2) 로지스틱 회귀 (확률로 답하는 모델)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
log_train_acc = log_reg.score(X_train_scaled, y_train)
log_test_acc = log_reg.score(X_test_scaled, y_test)

# (3) 의사결정트리 (질문으로 답하는 모델)
dt_clf = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=5,
    random_state=42
)
dt_clf.fit(X_train, y_train)
dt_train_acc = dt_clf.score(X_train, y_train)
dt_test_acc = dt_clf.score(X_test, y_test)

# 6. 모델 성능 정확도 카드 표시
st.subheader("📊 모델 정확도 평가")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="로지스틱 회귀(확률로 답하는 모델)",
        value=f"{log_test_acc:.4f}",
        delta=f"훈련: {log_train_acc:.4f} | 테스트: {log_test_acc:.4f}",
        delta_color="off"
    )

with col2:
    st.metric(
        label="의사결정트리(질문으로 답하는 모델)",
        value=f"{dt_test_acc:.4f}",
        delta=f"훈련: {dt_train_acc:.4f} | 테스트: {dt_test_acc:.4f}",
        delta_color="off"
    )

with col3:
    st.metric(
        label="다수결 모델(입력을 보지 않는 모델)",
        value=f"{baseline_test_acc:.4f}",
        delta=f"훈련: {baseline_train_acc:.4f} | 테스트: {baseline_test_acc:.4f}",
        delta_color="off"
    )

st.markdown("---")

# 7. 결정 경계 시각화 (Plotly)
st.subheader("📈 2차원 결정 경계 및 테스트 데이터 산점도")

# 축으로 사용할 2개 속성 선택
c1, c2 = st.columns(2)
with c1:
    x_korean = st.selectbox("가로축(X축) 속성 선택", selected_korean_features, index=0)
with c2:
    y_options = [f for f in selected_korean_features if f != x_korean]
    y_korean = st.selectbox("세로축(Y축) 속성 선택", y_options, index=0)

x_col = REVERSE_FEATURE_MAP[x_korean]
y_col = REVERSE_FEATURE_MAP[y_korean]

x_idx = selected_cols.index(x_col)
y_idx = selected_cols.index(y_col)

# 선택되지 않은 나머지 속성들의 테스트 데이터 중앙값 고정 안내
other_cols = [c for c in selected_cols if c not in [x_col, y_col]]
fixed_val_info = []
fixed_vals_dict = {}

for c in other_cols:
    med_val = X_test[c].median()
    fixed_vals_dict[c] = med_val
    fixed_val_info.append(f"**{FEATURE_MAP[c]}**: {med_val:.2f}")

if fixed_val_info:
    st.info(f"📌 **고정된 속성 중앙값 (테스트 데이터 기준)**: {', '.join(fixed_val_info)}")

# 격자(Grid) 생성
x_min, x_max = X_test[x_col].min() - 5, X_test[x_col].max() + 5
y_min, y_max = X_test[y_col].min() - 5, X_test[y_col].max() + 5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 150),
    np.linspace(y_min, y_max, 150)
)

# 격자 데이터프레임 구성 (선택되지 않은 축은 중앙값 배치)
grid_df = pd.DataFrame(index=range(xx.size))
for c in selected_cols:
    if c == x_col:
        grid_df[c] = xx.ravel()
    elif c == y_col:
        grid_df[c] = yy.ravel()
    else:
        grid_df[c] = fixed_vals_dict[c]

# 의사결정트리 예측 (배경색 칠하기용)
Z_dt = dt_clf.predict(grid_df[selected_cols]).reshape(xx.shape)

# 로지스틱 회귀 확률 예측
grid_df_scaled = scaler.transform(grid_df[selected_cols])
Z_log_prob = log_reg.predict_proba(grid_df_scaled)[:, 1].reshape(xx.shape)

fig = go.Figure()

# (1) 의사결정트리의 예측 구역 옅은 색 칠하기
fig.add_trace(go.Contour(
    x=np.linspace(x_min, x_max, 150),
    y=np.linspace(y_min, y_max, 150),
    z=Z_dt,
    showscale=False,
    opacity=0.2,
    colorscale=[[0, "lightblue"], [1, "orange"]],
    hoverinfo="skip"
))

# (2) 로지스틱 회귀 P=0.5 결정 경계선 그리기
fig.add_trace(go.Contour(
    x=np.linspace(x_min, x_max, 150),
    y=np.linspace(y_min, y_max, 150),
    z=Z_log_prob,
    contours_start=0.5,
    contours_end=0.5,
    contours_coloring='lines',
    line_width=3,
    line_color='black',
    name='로지스틱 회귀 경계선 (P=0.5)',
    showscale=False
))

# 경계선 화면 이탈 여부 확인
has_contour_05 = np.any(np.abs(Z_log_prob - 0.5) < 0.05)

# (3) 테스트 데이터 산점도 추가
for stroke_val, name, color in zip([0, 1], ["정상(아님)", "뇌졸중"], ["blue", "red"]):
    sub = test_df[test_df['stroke'] == stroke_val]
    fig.add_trace(go.Scatter(
        x=sub[x_col],
        y=sub[y_col],
        mode='markers',
        name=name,
        marker=dict(color=color, opacity=0.7, size=7)
    ))

fig.update_layout(
    title=f"로지스틱 회귀 & 의사결정트리 결정 경계 ({x_korean} vs {y_korean})",
    xaxis_title=x_korean,
    yaxis_title=y_korean,
    height=600
)

st.plotly_chart(fig, use_container_width=True)

if not has_contour_05:
    st.caption("ℹ️ **안내**: 로지스틱 회귀 모델의 0.5 확률 경계선이 현재 축 표시 영역(그림 밖)에 위치하지 않아 선이 그려지지 않았습니다.")

st.markdown("---")

# 8. 의사결정트리 가지 그림 (DOT & Graphviz)
st.subheader("🌳 의사결정트리 질문 구조 (Dot Graph)")

tree = dt_clf.tree_

def build_dot_string(tree_obj, feature_names):
    dot_lines = [
        'digraph Tree {',
        'node [shape=box, style="filled, rounded", color="black", fontname="helvetica"] ;',
        'edge [fontname="helvetica"] ;'
    ]
    
    def recurse(node_id):
        is_leaf = (tree_obj.children_left[node_id] == tree_obj.children_right[node_id])
        n_samples = tree_obj.n_node_samples[node_id]
        value = tree_obj.value[node_id][0]
        stroke_count = int(value[1]) if len(value) > 1 else 0
        stroke_ratio = (stroke_count / n_samples) * 100
        
        if is_leaf:
            pred_class = np.argmax(value)
            color = "#FFE0B2" if pred_class == 1 else "#E1F5FE"
            label = (
                f"답: {'뇌졸중' if pred_class == 1 else '정상(아님)'}\\n"
                f"인원 수: {n_samples}명\\n"
                f"뇌졸중 환자: {stroke_count}명 ({stroke_ratio:.1f}%)"
            )
            dot_lines.append(f'{node_id} [label="{label}", fillcolor="{color}"] ;')
        else:
            feat_name = feature_names[tree_obj.feature[node_id]]
            threshold = tree_obj.threshold[node_id]
            label = (
                f"{feat_name} <= {threshold:.2f}?\\n"
                f"인원 수: {n_samples}명\\n"
                f"뇌졸중 환자: {stroke_count}명 ({stroke_ratio:.1f}%)"
            )
            dot_lines.append(f'{node_id} [label="{label}", fillcolor="#FFFFFF"] ;')
            
            left_child = tree_obj.children_left[node_id]
            right_child = tree_obj.children_right[node_id]
            
            dot_lines.append(f'{node_id} -> {left_child} [labeldistance=2.5, labelangle=45, headlabel="예"] ;')
            dot_lines.append(f'{node_id} -> {right_child} [labeldistance=2.5, labelangle=-45, headlabel="아니요"] ;')
            
            recurse(left_child)
            recurse(right_child)

    recurse(0)
    dot_lines.append('}')
    return "\n".join(dot_lines)

dot_graph_str = build_dot_string(tree, selected_korean_features)
st.graphviz_chart(dot_graph_str)

# 의사결정트리 결과 한 줄 정리 요약
leaf_nodes = [i for i in range(tree.node_count) if tree.children_left[i] == tree.children_right[i]]
total_leafs = len(leaf_nodes)
normal_leafs = sum(1 for i in leaf_nodes if np.argmax(tree.value[i][0]) == 0)

used_feature_indices = set(tree.feature[i] for i in range(tree.node_count) if tree.children_left[i] != tree.children_right[i])
used_korean_features = [selected_korean_features[i] for i in sorted(list(used_feature_indices))]

st.write(f"- 답을 내는 마디(리프 노드)는 총 **{total_leafs}개**이며, 그중 **{normal_leafs}개**의 마디가 **'정상(아님)'**이라고 답합니다.")
st.write(f"- 고른 속성 중 이 나무가 실제로 질의에 활용한 속성은 **{', '.join(used_korean_features)}** 입니다.")
