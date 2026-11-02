import streamlit as st
import pandas as pd
import plotly.express as px

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="탐색 - 뇌졸중 예측 실습실",
    page_icon="🔍",
    layout="wide"
)

# 2. 데이터 로딩 (동일 데이터 URL)
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df = load_data()

st.title("🔍 데이터 탐색 (EDA)")
st.markdown("데이터의 주요 특징과 변수 간의 관계를 그래프와 표로 확인해보세요.")
st.markdown("---")

# ---------------------------------------------------------
# 1. 나이와 평균 혈당의 분포 (히스토그램 2개)
# ---------------------------------------------------------
st.subheader("📊 1. 나이 및 평균 혈당 분포")
col1, col2 = st.columns(2)

with col1:
    fig_age_hist = px.histogram(
        df, 
        x="age", 
        nbins=30, 
        title="나이(age) 분포",
        labels={"age": "나이", "count": "인원 수"},
        color_discrete_sequence=["#4F46E5"]
    )
    st.plotly_chart(fig_age_hist, use_container_width=True)

with col2:
    fig_glucose_hist = px.histogram(
        df, 
        x="avg_glucose_level", 
        nbins=30, 
        title="평균 혈당(avg_glucose_level) 분포",
        labels={"avg_glucose_level": "평균 혈당", "count": "인원 수"},
        color_discrete_sequence=["#10B981"]
    )
    st.plotly_chart(fig_glucose_hist, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 2. 뇌졸중 여부에 따른 나이 및 평균 혈당 비교 (상자그림 & 평균 표)
# ---------------------------------------------------------
st.subheader("📦 2. 뇌졸중 여부에 따른 나이 및 평균 혈당 비교")

# stroke 값 레이블 변환 (시각화용)
df_display = df.copy()
df_display["뇌졸중 여부"] = df_display["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})

col3, col4 = st.columns(2)

with col3:
    fig_age_box = px.box(
        df_display, 
        x="뇌졸중 여부", 
        y="age", 
        color="뇌졸중 여부",
        title="뇌졸중 여별 나이 분포",
        labels={"age": "나이"},
        color_discrete_map={"정상 (0)": "#6366F1", "뇌졸중 (1)": "#EF4444"}
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with col4:
    fig_glucose_box = px.box(
        df_display, 
        x="뇌졸중 여부", 
        y="avg_glucose_level", 
        color="뇌졸중 여부",
        title="뇌졸중 여별 평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당"},
        color_discrete_map={"정상 (0)": "#10B981", "뇌졸중 (1)": "#F59E0B"}
    )
    st.plotly_chart(fig_glucose_box, use_container_width=True)

# 두 그룹의 평균값 표 작성
group_means = df.groupby("stroke")[["age", "avg_glucose_level"]].mean().reset_index()
group_means["stroke"] = group_means["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})
group_means.columns = ["그룹 (stroke)", "평균 나이", "평균 혈당"]

st.markdown("**💡 그룹별 평균값 요약**")
st.dataframe(group_means.style.format({"평균 나이": "{:.2f} 세", "평균 혈당": "{:.2f}"}), use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 3. 기저질환(고혈압, 심장병) 유무에 따른 뇌졸중 비율 (막대그래프)
# ---------------------------------------------------------
st.subheader("🩸 3. 고혈압 및 심장병 유무에 따른 뇌졸중 비율")

col5, col6 = st.columns(2)

# 고혈압 뇌졸중 비율 계산
ht_stroke = df.groupby("hypertension")["stroke"].mean().reset_index()
ht_stroke["hypertension"] = ht_stroke["hypertension"].map({0: "없음 (0)", 1: "있음 (1)"})
ht_stroke["stroke_rate"] = ht_stroke["stroke"] * 100

with col5:
    fig_ht = px.bar(
        ht_stroke,
        x="hypertension",
        y="stroke_rate",
        text=ht_stroke["stroke_rate"].apply(lambda x: f"{x:.2f}%"),
        title="고혈압 유무별 뇌졸중 비율 (%)",
        labels={"hypertension": "고혈압", "stroke_rate": "뇌졸중 비율 (%)"},
        color="hypertension",
        color_discrete_map={"없음 (0)": "#3B82F6", "있음 (1)": "#EF4444"}
    )
    fig_ht.update_traces(textposition='outside')
    st.plotly_chart(fig_ht, use_container_width=True)

# 심장병 뇌졸중 비율 계산
hd_stroke = df.groupby("heart_disease")["stroke"].mean().reset_index()
hd_stroke["heart_disease"] = hd_stroke["heart_disease"].map({0: "없음 (0)", 1: "있음 (1)"})
hd_stroke["stroke_rate"] = hd_stroke["stroke"] * 100

with col6:
    fig_hd = px.bar(
        hd_stroke,
        x="heart_disease",
        y="stroke_rate",
        text=hd_stroke["stroke_rate"].apply(lambda x: f"{x:.2f}%"),
        title="심장병 유무별 뇌졸중 비율 (%)",
        labels={"heart_disease": "심장병", "stroke_rate": "뇌졸중 비율 (%)"},
        color="heart_disease",
        color_discrete_map={"없음 (0)": "#3B82F6", "있음 (1)": "#EF4444"}
    )
    fig_hd.update_traces(textposition='outside')
    st.plotly_chart(fig_hd, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 4. 체질량지수(bmi) 결측치 그룹과 전체 그룹의 뇌졸중 비율 비교
# ---------------------------------------------------------
st.subheader("🚨 4. BMI 결측치 그룹 vs 전체 뇌졸중 비율 비교")

bmi_null_df = df[df["bmi"].isnull()]
null_count = len(bmi_null_df)
null_stroke_rate = (bmi_null_df["stroke"].mean() * 100) if null_count > 0 else 0
total_stroke_rate = df["stroke"].mean() * 100

bmi_comparison_df = pd.DataFrame([
    {
        "구분": "BMI 결측치 그룹 (Null)",
        "대상 인원 수": f"{null_count:,} 명",
        "뇌졸중 환자 수": f"{bmi_null_df['stroke'].sum():,} 명",
        "뇌졸중 비율": f"{null_stroke_rate:.2f} %"
    },
    {
        "구분": "전체 데이터 그룹 (Overall)",
        "대상 인원 수": f"{len(df):,} 명",
        "뇌졸중 환자 수": f"{df['stroke'].sum():,} 명",
        "뇌졸중 비율": f"{total_stroke_rate:.2f} %"
    }
])

st.dataframe(bmi_comparison_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------
# 5. 흡연 상태 범주별 인원 수
# ---------------------------------------------------------
st.subheader("🚬 5. 흡연 상태(smoking_status)별 인원 수")

smoke_counts = df["smoking_status"].value_counts().reset_index()
smoke_counts.columns = ["흡연 상태 (smoking_status)", "인원 수"]
smoke_counts["비율"] = (smoke_counts["인원 수"] / len(df) * 100).map("{:.2f} %".format)

st.dataframe(smoke_counts, use_container_width=True, hide_index=True)
