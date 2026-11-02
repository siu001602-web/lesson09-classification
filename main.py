import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# 2. 내_기록 폴더 및 JSON 파일 경로 설정
SAVE_DIR = "내_기록"
SAVE_FILE = os.path.join(SAVE_DIR, "notes.json")

def load_user_notes():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_user_notes(notes):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=4)

# 3. 데이터 로딩
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df = load_data()

# 저장된 사용자 기록 불러오기
saved_notes = load_user_notes()

# 4. 앱 헤더
st.title("🧠 뇌졸중 예측 실습실")
st.markdown("---")

# 5. 주요 지표 카드 (Metric Cards)
total_count = len(df)
col_count = len(df.columns)
stroke_count = int(df['stroke'].sum()) if 'stroke' in df.columns else 0
stroke_ratio = (stroke_count / total_count * 100) if total_count > 0 else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric(label="전체 사람 수", value=f"{total_count:,} 명")
m2.metric(label="열 개수", value=f"{col_count} 개")
m3.metric(label="뇌졸중 환자 수 (stroke=1)", value=f"{stroke_count:,} 명")
m4.metric(label="뇌졸중 환자 비율", value=f"{stroke_ratio:.2f} %")

st.markdown("---")

# 6. 열 정보 및 우리말 뜻 입력 표
st.subheader("📋 데이터 열 정보 및 명세")

# 기존 저장된 우리말 뜻 가져오기
column_meanings_saved = saved_notes.get("column_meanings", {})

# 표 형태 데이터 구성
cols_info = []
for col in df.columns:
    cols_info.append({
        "열 이름": col,
        "우리말 뜻": column_meanings_saved.get(col, ""),
        "값의 종류": str(df[col].dtype),
        "빈 값 개수": int(df[col].isnull().sum())
    })

info_df = pd.DataFrame(cols_info)

# data_editor를 통한 사용자 직접 수정 가능 표
edited_info_df = st.data_editor(
    info_df,
    column_config={
        "열 이름": st.column_config.Column(disabled=True),
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻 (교재를 보고 입력하세요)", help="직접 작성 시 자동 저장됩니다."),
        "값의 종류": st.column_config.Column(disabled=True),
        "빈 값 개수": st.column_config.Column(disabled=True),
    },
    hide_index=True,
    use_container_width=True,
    key="editor_column_meanings"
)

# 데이터 변경 사항 감지 및 자동 저장
new_column_meanings = {row["열 이름"]: row["우리말 뜻"] for row in edited_info_df.to_dict("records")}

st.markdown("---")

# 7. 상위 5개 행 데이터 미리보기
st.subheader("🔍 데이터 미리보기 (상위 5개 행)")
st.dataframe(df.head(5), use_container_width=True)

st.markdown("---")

# 8. 데이터 출처 입력란
st.subheader("📌 데이터 출처")
saved_source = saved_notes.get("data_source", "")
source_input = st.text_area(
    "데이터 출처를 입력하세요 (교재 참고):",
    value=saved_source,
    placeholder="여기에 데이터 출처 관련 설명을 적어주세요...",
    key="input_data_source"
)

# 저장 처리 (우리말 뜻 변경 또는 출처 변경 시 저장)
if (new_column_meanings != column_meanings_saved) or (source_input != saved_source):
    updated_notes = {
        "column_meanings": new_column_meanings,
        "data_source": source_input
    }
    save_user_notes(updated_notes)
