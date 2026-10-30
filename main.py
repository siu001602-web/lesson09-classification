import json
import os
import pandas as pd
import plotly.express as px
import streamlit as st

# 1. 페이지 기본 설정 (브라우저 탭 제목 및 아이콘)
st.set_page_config(
    page_title="뇌졸중 예측 실습실", page_icon="🧠", layout="wide"
)

# 파일 저장 경로 설정
DATA_DIR = "내_기록"
JSON_FILE = os.path.join(DATA_DIR, "user_notes.json")


# JSON 파일 저장 함수
def save_user_data(data):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# JSON 파일 불러오기 함수
def load_user_data():
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# 저장된 사용자 입력 데이터 로드
saved_data = load_user_data()

# 세션 상태 초기화 (저장된 값이 있으면 불러오고, 없으면 기본값 설정)
columns_list = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]

for col in columns_list:
    key = f"meaning_{col}"
    if key not in st.session_state:
        st.session_state[key] = saved_data.get(key, "")

if "source_text" not in st.session_state:
    st.session_state["source_text"] = saved_data.get("source_text", "")


# 저장 버튼 클릭 시 실행될 콜백 함수
def save_all_inputs():
    new_data = {}
    for col in columns_list:
        key = f"meaning_{col}"
        new_data[key] = st.session_state[key]
    new_data["source_text"] = st.session_state["source_text"]
    save_user_data(new_data)
    st.toast("기록이 성공적으로 저장되었습니다!", icon="💾")


# 2. 데이터 불러오기 (캐싱 적용)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/siu001602-web/data/refs/heads/main/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df


df = load_data()

# 3. 앱 헤더
st.title("🧠 뇌졸중 예측 실습실")
st.markdown("---")

# 4. 데이터셋 소개 섹션
st.header("📊 데이터셋 소개")

# 상단 메트릭 카드 4개
col1, col2, col3, col4 = st.columns(4)

total_people = len(df)
total_cols = len(df.columns)
stroke_cases = int(df["stroke"].sum())
stroke_ratio = (stroke_cases / total_people) * 100

col1.metric(label="전체 사람 수", value=f"{total_people:,} 명")
col2.metric(label="열 개수", value=f"{total_cols} 개")
col3.metric(label="뇌졸중 환자 수 (stroke=1)", value=f"{stroke_cases:,} 명")
col4.metric(label="뇌졸중 비율", value=f"{stroke_ratio:.2f} %")

st.markdown("---")

# 5. 열 정보 안내 및 입력 표
st.subheader("📋 데이터 열(Column) 정보 및 의미 작성")
st.caption(
    "교재를 참고하여 각 열의 우리말 뜻을 입력한 후, 하단의 '기록 저장하기' 버튼을 눌러주세요."
)

# 입력값을 저장하기 위한 데이터 프레임 구성
col_info = []
for col in columns_list:
    dtype_str = str(df[col].dtype)
    null_count = int(df[col].isnull().sum())
    meaning_val = st.session_state[f"meaning_{col}"]
    col_info.append(
        {
            "열 이름": col,
            "우리말 뜻": meaning_val,
            "값의 종류": dtype_str,
            "빈 값 개수": null_count,
        }
    )

# 데이터 에디터를 통해 우리말 뜻을 직접 편집 가능하도록 구현
info_df = pd.DataFrame(col_info)

edited_df = st.data_editor(
    info_df,
    column_config={
        "열 이름": st.column_config.Column(disabled=True),
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻 (편집 가능)"),
        "값의 종류": st.column_config.Column(disabled=True),
        "빈 값 개수": st.column_config.Column(disabled=True),
    },
    hide_index=True,
    use_container_width=True,
)

# 에디터에서 변경된 내용을 세션 상태에 즉시 반영
for idx, row in edited_df.iterrows():
    col_name = row["열 이름"]
    st.session_state[f"meaning_{col_name}"] = row["우리말 뜻"]

st.markdown("---")

# 6. 상위 5개 데이터 미리보기
st.subheader("🔍 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.markdown("---")

# 7. 데이터 출처 작성 영역 및 저장 버튼
st.subheader("📝 데이터 출처 및 메모")
st.text_area(
    label="출처 및 교재 내용 작성",
    key="source_text",
    height=100,
    placeholder="교재에 있는 데이터 출처나 필요한 메모를 입력하세요.",
)

# 저장 버튼
st.button("💾 내 기록 저장하기", on_click=save_all_inputs, type="primary")
