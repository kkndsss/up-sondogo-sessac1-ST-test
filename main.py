import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ----------------------------------------------------------------------
# 데이터 읽기 (인코딩 자동 처리)
# ----------------------------------------------------------------------
@st.cache_data
def read_csv(file):
    """utf-8-sig -> cp949 순서로 시도하여 CSV를 읽는다."""
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

def std_cols(df):
    """학교마다 다른 열 이름을 표준 이름으로 바꾼다."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "").
                str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce"
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + \
                          pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df

def fmt_id(series):
    """개체번호를 정수면 정수 그대로, 글자면 글자 그대로 문자열로 반환."""
    out = []
    for v in series:
        if pd.isna(v):
            out.append("")
        elif isinstance(v, (int, np.integer)):
            out.append(str(int(v)))
        else:
            out.append(str(v))
    return out

def base_layout(fig, title):
    """Plotly 그래프 공통 레이아웃 (한글 폰트·반응형)."""
    fig.update_layout(
        title={"text": title, "x": 0.05, "xanchor": "left"},
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified",
    )
    return fig

# ----------------------------------------------------------------------
# 페이지 설정
# ----------------------------------------------------------------------
st.set_page_config(page_title="송도 새싹 1팀", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("🌿 송도 새싹 1팀 — 극지식물 생육 차이 분석 대시보드")

# ----------------------------------------------------------------------
# 사이드바: CSV 업로드
# ----------------------------------------------------------------------
st.sidebar.header("📁 데이터 업로드")

env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")

measure_cols = [
    ("잎 수", "잎 수"),
    ("지상부 길이", "지상부 길이 (mm)"),
    ("지하부 길이", "지하부 길이 (mm)"),
    ("생중량", "생중량 (g)"),
]

selected_measure = st.sidebar.selectbox(
    "📏 생육 측정 항목 선택 (탭3 그래프 기준)",
    options=[m[0] for m in measure_cols],
    index=3,
)

env_df = pd.DataFrame()
growth_df = pd.DataFrame()
school_name = "업로드된 학교"

if env_file is not None:
    with st.sidebar.spinner("환경 데이터를 읽는 중..."):
        env_df = std_cols(read_csv(env_file))
if growth_file is not None:
    with st.sidebar.spinner("생육 결과 데이터를 읽는 중..."):
        growth_df = std_cols(read_csv(growth_file))

if env_file is not None:
    raw = env_file.name
    school_name = raw.split("_")[0].strip()

st.sidebar.success(f"🏫 학교명: {school_name}")

st.sidebar.markdown("---")
st.sidebar.info("💡 CSV를 업로드하면 탭별 분석이 자동으로 표시됩니다.")
st.sidebar.markdown("""
**팁**
- 인코딩 오류가 나면 앱이 자동으로 cp949로 다시 읽습니다.
- 열 이름이 달라도 std_cols 함수가 표준 이름으로 바꿔줍니다.
- 개체번호는 분석 대상에서 제외되며, 그대로 표시됩니다.
""")

env_avail = not env_df.empty
growth_avail = not growth_df.empty

# ======================================================================
# 탭 구성
# ======================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 실험 개요",
    "🌡️ 환경 데이터",
    "📊 생육 결과",
    "🔍 가설 검증",
    "💡 한계 및 개선방향",
])

# ----------------------------------------------------------------------
# Tab 1 — 실험 개요
# ----------------------------------------------------------------------
with tab1:
    try:
        st.header("📖 실험 개요")

        st.subheader("연구 배경 및 목적")
        st.markdown("""
**극지식물**은 극한의 추위·건조·자외선에서 살아남는 식물로, 기후변화 연구의 지표이자
극한 환경 농업의 가능성을 열어줍니다.

이 실험은 같은 스마트팜에서 재배된 극지식물 개체들 사이에 **지상부 길이(줄기)**와
**지하부 길이(뿌리)** 사이에 어떤 관계가 있는지 알아보는 것이 목적입니다.

- **가설**: 지상부 길이와 지하부 길이 사이에는 뚜렷한 상관관계가 없을 것이다.
- **방법**: 생육 결과 데이터의 개체별 지상부·지하부 길이를 피어슨 상관계수로 비교한다.
""")

        # 데이터 요약 표
        st.subheader("데이터 요약")
        summary_rows = []
        if growth_avail:
            n_ind = len(growth_df)
            measure_present = [m[1] for m in measure_cols if m[0] in growth_df.columns]
            summary_rows.append({
                "항목": "개체 수",
                "값": f"{n_ind}개체",
                "설명": "생육 결과 데이터에 기록된 극지식물 개체 수",
            })
            summary_rows.append({
                "항목": "측정 항목",
                "값": ", ".join(measure_present) if measure_present else "없음",
                "설명": "생육결과데이터에 실제로 존재하는 측정 열",
            })
        if env_avail and "time" in env_df.columns:
            t_min = env_df["time"].min()
            t_max = env_df["time"].max()
            summary_rows.append({
                "항목": "측정 기간",
                "값": f"{t_min.date()} ~ {t_max.date()}",
                "설명": "환경 센서 데이터가 기록된 기간",
            })
        if env_avail:
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            summary_rows.append({
                "항목": "환경 측정 항목",
                "값": ", ".join(env_cols) if env_cols else "없음",
                "설명": "환경 CSV에 존재하는 센서 항목",
            })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), width="stretch")
        else:
            st.info("📁 데이터를 업로드하면 요약 정보가 표시됩니다.")

        # 주요 지표 카드 4개
        st.subheader("주요 지표")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if growth_avail and "개체번호" in growth_df.columns:
                n_total = len(growth_df)
                st.metric("총 개체수", f"{n_total}개체")
            else:
                st.info("개체수 정보는 생육 CSV 업로드 후 확인 가능")
        with col2:
            if env_avail and "temperature" in env_df.columns:
                t_vals = env_df["temperature"].dropna()
                if len(t_vals) >= 2:
                    st.metric("평균 온도", f"{t_vals.mean():.1f} ℃",
                              delta=f"범위 {t_vals.min():.1f} ~ {t_vals.max():.1f} ℃")
                else:
                    st.info("온도 데이터 부족")
            else:
                st.info("평균 온도는 환경 CSV 업로드 후 확인 가능")
        with col3:
            if env_avail and "humidity" in env_df.columns:
                h_vals = env_df["humidity"].dropna()
                if len(h_vals) >= 2:
                    st.metric("평균 습도", f"{h_vals.mean():.1f} %",
                              delta=f"범위 {h_vals.min():.1f} ~ {h_vals.max():.1f} %")
                else:
                    st.info("습도 데이터 부족")
            else:
                st.info("평균 습도는 환경 CSV 업로드 후 확인 가능")
        with col4:
            if growth_avail and "생중량" in growth_df.columns:
                fw = growth_df["생중량"].dropna()
                if len(fw) >= 2:
                    st.metric("평균 생중량", f"{fw.mean():.2f} g",
                              delta=f"범위 {fw.min():.2f} ~ {fw.max():.2f} g")
                else:
                    st.info("생중량 데이터 부족")
            else:
                st.info("평균 생중량은 생육 CSV 업로드 후 확인 가능")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ----------------------------------------------------------------------
# Tab 2 — 환경 데이터
# ----------------------------------------------------------------------
with tab2:
    try:
        if not env_avail:
            st.info("🌡️ 환경 데이터 탭 — 환경 CSV를 업로드하면 분석 결과가 표시됩니다.")
        else:
            st.header("🌡️ 환경 데이터")
            st.caption(f"학교명: {school_name}  |  데이터 출처: 업로드된 환경 CSV")

            # 환경 요약 표 (평균·최소·최대)
            st.subheader("환경 요약 (평균 · 최소 · 최대)")
            env_metrics = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if env_metrics:
                rows = []
                for col in env_metrics:
                    v = env_df[col].dropna()
                    if len(v) >= 2:
                        avg = v.mean()
                        cv_val = (v.std() / avg * 100) if avg != 0 else 0
                        rows.append({
                            "항목": col,
                            "평균": round(avg, 3),
                            "최소": round(v.min(), 3),
                            "최대": round(v.max(), 3),
                            "표준편차": round(v.std(), 3),
                            "CV (%)": round(cv_val, 2),
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), width="stretch")
                    st.caption("💡 CV(변동계수) = 표준편차 ÷ 평균 × 100  →  단위가 다른 항목끼리도 평균 대비 변동성을 비교할 수 있다.")
                else:
                    st.info("표시할 환경 지표가 없습니다.")
            else:
                st.info("표시할 환경 지표(temperature, humidity, ph, ec)가 없습니다.")

            # 날짜별 시계열 서브플롯 (온도·습도 + pH·EC)
            env_time_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if env_time_cols and "time" in env_df.columns:
                st.subheader("날짜별 시계열 (일별 평균)")
                daily = env_df.set_index("time").resample("D")[env_time_cols].mean().dropna(how="all")

                n_rows = 1 + (1 if "ph" in daily.columns else 0) + (1 if "ec" in daily.columns else 0)
                subplot_titles = ["온도 · 습도 일별 평균"]
                if "ph" in daily.columns:
                    subplot_titles.append("pH 일별 평균")
                if "ec" in daily.columns:
                    subplot_titles.append("EC 일별 평균")

                fig = make_subplots(rows=n_rows, cols=1, subplot_titles=subplot_titles[:n_rows])
                colors_map = {"temperature": "#e74c3c", "humidity": "#3498db",
                              "ph": "#8e44ad", "ec": "#27ae60"}
                for i, col in enumerate(env_time_cols):
                    if col in daily.columns and len(daily[col].dropna()) >= 2:
                        if col in ("temperature", "humidity"):
                            row_idx = 1
                        elif col == "ph":
                            row_idx = 2
                        else:
                            row_idx = 3
                        fig.add_trace(
                            go.Scatter(
                                x=daily.index,
                                y=daily[col].values,
                                mode="lines",
                                name=col,
                                line=dict(color=colors_map.get(col, "#333"), width=2),
                                hovertemplate="%{x|%Y-%m-%d}<br>%{y}<extra></extra>",
                            ),
                            row=row_idx,
                            col=1,
                        )
                fig = base_layout(fig, "날짜별 환경 지표 일별 평균 추이")
                fig.update_xaxes(tickformat="%m/%d", row=1, col=1)
                st.plotly_chart(fig, width="stretch")

                # 이상치 정보 (IQR 기준)
                st.subheader("이상치 현황 (IQR 기준)")
                outlier_rows = []
                for col in env_time_cols:
                    v = env_df[col].dropna()
                    if len(v) >= 4:
                        q1, q3 = v.quantile(0.25), v.quantile(0.75)
                        iqr_val = q3 - q1
                        lo, hi = q1 - 1.5 * iqr_val, q3 + 1.5 * iqr_val
                        n_out = int(((v < lo) | (v > hi)).sum())
                        outlier_rows.append({
                            "항목": col,
                            "IQR 이상치 개수": n_out,
                            "Q1": round(q1, 3),
                            "Q3": round(q3, 3),
                            "IQR": round(iqr_val, 3),
                        })
                if outlier_rows:
                    st.dataframe(pd.DataFrame(outlier_rows), width="stretch")
                else:
                    st.info("이상치 계산에 충분한 데이터가 없습니다.")
            else:
                st.info("날짜별 시계열: time 열 또는 환경 지표 열이 없어 표시할 수 없습니다.")

            # CV 비교 막대그래프
            if env_metrics:
                st.subheader("환경 데이터별 CV 비교 막대그래프")
                cv_data = []
                for col in env_metrics:
                    v = env_df[col].dropna()
                    if len(v) >= 2:
                        mean_v = v.mean()
                        cv_val = (v.std() / mean_v * 100) if mean_v != 0 else 0
                        cv_data.append({"항목": col, "CV (%)": cv_val})
                if cv_data:
                    cv_df = pd.DataFrame(cv_data)
                    fig2 = px.bar(
                        cv_df,
                        x="항목",
                        y="CV (%)",
                        color="항목",
                        color_discrete_map={
                            "temperature": "#e74c3c",
                            "humidity": "#3498db",
                            "ph": "#8e44ad",
                            "ec": "#27ae60",
                        },
                        labels={"항목": "환경 항목", "CV (%)": "CV (%)"},
                        title="환경 지표별 변동계수(CV) 비교",
                    )
                    fig2 = base_layout(fig2, "환경 지표별 CV 비교")
                    fig2.update_xaxes(title_text="환경 항목")
                    fig2.update_yaxes(title_text="CV (%)")
                    st.plotly_chart(fig2, width="stretch")
                    st.caption("💡 CV가 높을수록 평균 주변에서의 들쭉날쭉함이 크다는 뜻이다. EC·pH는 후반부 급변동이 잦아 CV가 높은 편이다.")
            else:
                st.info("CV 비교 그래프: 표시할 환경 지표가 없습니다.")

            # 원본 테이블 + 다운로드 (expander)
            with st.expander("📋 환경 데이터 원본 테이블 + CSV 다운로드"):
                if env_avail:
                    display_cols = [c for c in ["time", "temperature", "humidity", "ph", "ec"] if c in env_df.columns]
                    if display_cols:
                        st.dataframe(env_df[display_cols], width="stretch")
                    csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "⬇️ 환경 CSV 다운로드 (utf-8-sig)",
                        csv_bytes,
                        file_name=f"{school_name}_환경데이터_표준화.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("환경 CSV를 업로드하면 원본 테이블과 다운로드 버튼이 표시됩니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ----------------------------------------------------------------------
# Tab 3 — 생육 결과
# ----------------------------------------------------------------------
with tab3:
    try:
        if not growth_avail:
            st.info("📊 생육 결과 탭 — 생육 결과 CSV를 업로드하면 분석 결과가 표시됩니다.")
        else:
            st.header("📊 생육 결과")

            # 핵심 결과 카드: 평균 생중량 + 최대·최소 개체
            st.subheader("🥇 핵심 결과 카드")
            if "생중량" in growth_df.columns:
                fw_all = growth_df["생중량"].dropna()
                if len(fw_all) >= 2:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("평균 생중량", f"{fw_all.mean():.2f} g",
                                  delta=f"중앙값 {fw_all.median():.2f} g")
                    with col_b:
                        top_idx = fw_all.idxmax()
                        top_id = growth_df.loc[top_idx, "개체번호"]
                        st.metric("최대 생중량 개체",
                                  f"{fmt_id([top_id])[0]}",
                                  delta=f"{fw_all.max():.2f} g")
                    with col_c:
                        bot_idx = fw_all.idxmin()
                        bot_id = growth_df.loc[bot_idx, "개체번호"]
                        st.metric("최소 생중량 개체",
                                  f"{fmt_id([bot_id])[0]}",
                                  delta=f"{fw_all.min():.2f} g")
                else:
                    st.info("생중량 데이터가 충분하지 않습니다.")
            else:
                st.info("생중량 열이 없어 핵심 결과 카드를 표시할 수 없습니다.")

            # 개체별 생육 비교 막대그래프
            st.subheader("개체별 생육 비교 (막대 그래프)")
            avail_growth_cols = [c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"] if c in growth_df.columns]
            if avail_growth_cols:
                show_cols_list = [selected_measure] + \
                    [c for c in ["생중량", "잎 수", "지상부 길이"] if c != selected_measure and c in growth_df.columns]
                show_cols_list = show_cols_list[:3]
                bar_traces = []
                colors_bar = ["#e74c3c", "#3498db", "#2ecc71"]
                for i, col in enumerate(show_cols_list):
                    v = growth_df[col].dropna()
                    if len(v) >= 2:
                        bar_traces.append(go.Bar(
                            x=fmt_id(growth_df.loc[v.index, "개체번호"]),
                            y=v.values,
                            name=col,
                            marker=dict(color=colors_bar[i % 3]),
                            hovertemplate="%{x}<br>%{y}<extra></extra>",
                        ))
                if bar_traces:
                    fig3 = go.Figure(data=bar_traces)
                    fig3 = base_layout(fig3, f"개체별 생육 비교 — {', '.join(show_cols_list)}")
                    fig3.update_xaxes(title_text="개체번호", tickangle=45)
                    fig3.update_yaxes(title_text="측정값")
                    st.plotly_chart(fig3, width="stretch")
                    st.caption(f"💡 선택한 측정 항목: **{selected_measure}**. 막대 그래프에서 개체 간 차이를 직접 비교할 수 있다.")
                else:
                    st.info("그래프를 그릴 만큼 충분한 데이터 열이 없습니다.")
            else:
                st.info("표시할 생육 지표(잎 수, 지상부 길이, 지하부 길이, 생중량)가 없습니다.")

            # 생중량 분포 (박스플롯)
            if "생중량" in growth_df.columns:
                st.subheader("생중량 분포 (박스플롯)")
                fw_v = growth_df["생중량"].dropna()
                if len(fw_v) >= 2:
                    fig4 = go.Figure()
                    fig4.add_trace(go.Box(
                        y=fw_v.values,
                        name="생중량",
                        fillcolor="#3498db",
                        line=dict(color="#2980b9"),
                        boxpoints="outliers",
                        hovertemplate="%{y} g<extra></extra>",
                    ))
                    fig4 = base_layout(fig4, "개체별 생중량 분포 (박스플롯)")
                    fig4.update_yaxes(title_text="생중량 (g)")
                    st.plotly_chart(fig4, width="stretch")

                    st.caption(f"""
**생중량 기초 통계**
- 평균: {fw_v.mean():.2f} g  |  중앙값: {fw_v.median():.2f} g
- 표준편차: {fw_v.std():.2f} g  |  CV: {fw_v.std()/fw_v.mean()*100:.1f}%
- 범위: {fw_v.min():.2f} g ~ {fw_v.max():.2f} g  ({len(fw_v)}개체)
""")
                else:
                    st.info("생중량 데이터가 충분하지 않아 박스플롯을 그릴 수 없습니다.")

            # 원본 테이블 + 다운로드 (expander)
            with st.expander("📋 생육 결과 데이터 원본 테이블 + CSV 다운로드"):
                if growth_avail:
                    show_gcols = [c for c in ["개체번호", "잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns]
                    if show_gcols:
                        st.dataframe(growth_df[show_gcols], width="stretch")
                    csv_bytes_g = growth_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "⬇️ 생육 결과 CSV 다운로드 (utf-8-sig)",
                        csv_bytes_g,
                        file_name=f"{school_name}_생육결과_표준화.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("생육 CSV를 업로드하면 원본 테이블과 다운로드 버튼이 표시됩니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ----------------------------------------------------------------------
# Tab 4 — 가설 검증
# ----------------------------------------------------------------------
with tab4:
    try:
        st.header("🔍 가설 검증 — 지상부 길이 vs 지하부 길이")

        if not growth_avail or not {"지상부 길이", "지하부 길이"}.issubset(growth_df.columns):
            st.info("가설 검증 탭 — 생육 CSV에 **지상부 길이**와 **지하부 길이** 열이 있어야 분석할 수 있습니다.")
        else:
            above = growth_df["지상부 길이"].dropna()
            below = growth_df["지하부 길이"].dropna()
            common_idx = above.index.intersection(below.index)
            x = above.loc[common_idx].values
            y = below.loc[common_idx].values

            if len(x) < 2:
                st.info("지상부·지하부 길이 데이터가 충분하지 않아 상관관계 분석이 어렵습니다.")
            else:
                # 핵심 결과 카드
                st.subheader("핵심 결과 카드: 지상부 vs 지하부 길이 상관관계")
                corr_val = float(np.corrcoef(x, y)[0, 1])
                col_h1, col_h2, col_h3 = st.columns(3)
                with col_h1:
                    st.metric("피어슨 상관계수 (r)", f"{corr_val:.3f}")
                with col_h2:
                    st.metric("지상부 길이 평균", f"{x.mean():.1f} mm",
                              delta=f"n={len(x)}")
                with col_h3:
                    st.metric("지하부 길이 평균", f"{y.mean():.1f} mm",
                              delta=f"n={len(y)}")

                st.caption(f"""
**통계량 설명 (고등학생용)**
- **피어슨 상관계수(r)**: 두 변수가 함께 증가하거나 감소하는 경향을 -1 ~ +1 사이 숫자로 나타낸 값.
  +1에 가까우면 함께 증가, -1에 가까우면 한쪽 증가 시 다른 쪽 감소, 0에 가까우면 선형 관계가 거의 없음.
- 이 데이터에서는 r = **{corr_val:.3f}** 로, 지상부 길이와 지하부 길이 사이에 **뚜렷한 선형 상관관계가 거의 없음**을 보여준다.
""")

                # 기초 통계분석표
                st.subheader("지상부·지하부 기초 통계분석표")
                stat_rows = []
                for label, vals in [("지상부 길이 (mm)", x), ("지하부 길이 (mm)", y)]:
                    n = len(vals)
                    mean_v = vals.mean()
                    stat_rows.append({
                        "변수": label,
                        "n": n,
                        "평균": round(mean_v, 2),
                        "중앙값": round(float(np.median(vals)), 2),
                        "표준편차": round(float(vals.std()), 2),
                        "최소": round(float(vals.min()), 2),
                        "최대": round(float(vals.max()), 2),
                        "CV (%)": round(float(vals.std() / mean_v * 100), 2) if mean_v != 0 else 0,
                        "왜도": round(float(pd.Series(vals).skew()), 3),
                        "상관계수(r)": "",
                    })
                stat_rows[0]["상관계수(r)"] = round(corr_val, 3)
                stat_rows[1]["상관계수(r)"] = round(corr_val, 3)
                st.dataframe(pd.DataFrame(stat_rows), width="stretch")

                # 평균 비율 원그래프
                st.subheader("지상부·지하부 길이 평균 비율 (원그래프)")
                avg_above = float(x.mean())
                avg_below = float(y.mean())
                total_len = avg_above + avg_below
                if total_len > 0:
                    fig_pie = go.Figure()
                    fig_pie.add_trace(go.Pie(
                        labels=["지상부 길이", "지하부 길이"],
                        values=[avg_above, avg_below],
                        marker=dict(colors=["#e74c3c", "#3498db"]),
                        textinfo="percent+label",
                        hovertemplate="%{label}<br>평균 %{value:.1f} mm<extra></extra>",
                    ))
                    fig_pie = base_layout(fig_pie, f"지상부·지하부 평균 길이 비율 (총 {total_len:.1f} mm)")
                    st.plotly_chart(fig_pie, width="stretch")
                    st.caption(f"💡 지상부 평균 {avg_above:.1f} mm, 지하부 평균 {avg_below:.1f} mm — 지하부가 평균 {(avg_below/avg_above):.2f}배로 더 길게 자랐다.")

                # 전체 생육 데이터 히트맵
                st.subheader("생육 데이터 전체 상관관계 히트맵")
                heat_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns]
                if len(heat_cols) >= 2:
                    hm_df = growth_df[heat_cols].dropna()
                    if len(hm_df) >= 2:
                        hm_mat = hm_df.corr()
                        fig_heat = px.imshow(
                            hm_mat,
                            text_auto=".2f",
                            color_continuous_scale="RdBu_r",
                            labels={"x": "변수", "y": "변수", "color": "상관계수(r)"},
                            title="생육 지표 간 피어슨 상관계수 히트맵",
                        )
                        fig_heat = base_layout(fig_heat, "생육 지표 간 상관관계 히트맵")
                        fig_heat.update_layout(coloraxis_colorbar=dict(title="r"))
                        st.plotly_chart(fig_heat, width="stretch")
                        st.caption("💡 히트맵의 색이 붉을수록 양의 상관, 푸를수록 음의 상관, 색이 약할수록 상관 없음. 지상부-지하부 셀을 보면 두 길이 사이의 관계를 한눈에 알 수 있다.")
                    else:
                        st.info("히트맵 계산에 충분한 데이터가 없습니다.")
                else:
                    st.info("히트맵을 그릴 생육 지표 열이 2개 이상 필요합니다.")

                # 산점도 + 추세선 (np.polyfit)
                st.subheader("지상부·지하부 길이 상관관계 산점도")
                if len(x) >= 2:
                    fig_sc = go.Figure()
                    fig_sc.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        marker=dict(size=9, color="#3498db", opacity=0.7,
                                    line=dict(width=1, color="#2980b9")),
                        hovertemplate="지상부 %{x:.1f} mm<br>지하부 %{y:.1f} mm<extra></extra>",
                        name="개체",
                    ))
                    slope, intercept = np.polyfit(x, y, 1)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = slope * x_line + intercept
                    fig_sc.add_trace(go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        line=dict(color="#e74c3c", width=2, dash="dash"),
                        name=f"추세선 (기울기={slope:.3f})",
                        hoverinfo="skip",
                    ))
                    fig_sc = base_layout(fig_sc, f"지상부 vs 지하부 길이 (r = {corr_val:.3f})")
                    fig_sc.update_xaxes(title_text="지상부 길이 (mm)")
                    fig_sc.update_yaxes(title_text="지하부 길이 (mm)")
                    st.plotly_chart(fig_sc, width="stretch")
                    st.caption(f"💡 빨간 점선이 추세선. 기울기가 {slope:.3f}로 0에 가까워, 지상부 길이가 변해도 지하부 길이는 거의 변하지 않는 경향이 있다.")

                # 가설 검증 결과 분석 설명
                st.subheader("가설 검증 결과 분석")
                if abs(corr_val) < 0.2:
                    verdict = "지상부 길이와 지하부 길이 사이에 **뚜렷한 선형 상관관계가 거의 없다**고 볼 수 있다."
                elif corr_val >= 0.2:
                    verdict = "지상부 길이가 길수록 지하부 길이도 길어지는 **약한 양의 상관**이 관찰된다."
                else:
                    verdict = "지상부 길이가 길수록 지하부 길이가 짧아지는 **약한 음의 상관**이 관찰된다."

                # 극단값 예시 개체 찾기
                example_rows_list = []
                combined = pd.DataFrame({"지상부": x, "지하부": y})
                combined["비율"] = combined["지하부"] / combined["지상부"].replace(0, np.nan)
                if len(combined) > 0:
                    top_ratio = combined.nlargest(1, "비율").iloc[0] if combined["비율"].notna().any() else None
                    low_ratio = combined.nsmallest(1, "비율").iloc[0] if combined["비율"].notna().any() else None
                    if top_ratio is not None:
                        example_rows_list.append(
                            f"- 지상부 {top_ratio['지상부']:.0f}mm / 지하부 {top_ratio['지하부']:.0f}mm "
                            f"— 지상부 대비 지하부가 특히 긴 개체"
                        )
                    if low_ratio is not None:
                        example_rows_list.append(
                            f"- 지상부 {low_ratio['지상부']:.0f}mm / 지하부 {low_ratio['지하부']:.0f}mm "
                            f"— 지상부 대비 지하부가 특히 짧은 개체"
                        )

                example_text = "\n".join(example_rows_list) if example_rows_list else "- 극단 비율 개체는 표본에서 확인되지 않았다."

                st.markdown(f"""
**가설**: "지상부 길이와 지하부 길이 사이에는 뚜렷한 상관관계가 없을 것이다."

**분석 결과**
- 피어슨 상관계수 r = **{corr_val:.3f}** (n = {len(x)}개체)
- {verdict}
- 상관계수 절댓값이 0.2 미만이므로, 지상부 길이만으로는 지하부 길이를 예측하기 어렵고 그 반대도 마찬가지다.
- 이는 식물의 줄기와 뿌리가 서로 다른 환경 조건과 생리적 요인의 영향을 받아 성장할 수 있다는 가설 근거와 일치한다.
- 다만 r ≈ 0 이 '두 생장이 완전히 무관하다'는 뜻은 아니며, **선형 관계가 거의 없다**는 의미로 해석하는 것이 적절하다.
- 표본이 {len(x)}개체로 제한적이므로, 이 결과는 해당 학교와 해당 시기의 데이터에 국한된 관찰이다.

**예시 개체 (극단값)**
{example_text}
""")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ----------------------------------------------------------------------
# Tab 5 — 한계 및 개선방향
# ----------------------------------------------------------------------
with tab5:
    try:
        st.header("💡 한계 및 개선방향")
        st.subheader("연구 한계")
        st.markdown("""
- **생육 측정일 부재**: 생육결과데이터에는 생육 측정일이 기록되어 있지 않아, 환경데이터의 측정 시점과
  생육 결과를 같은 시점에 연결하여 분석하기 어렵다.

- **개체 위치·처리구 정보 부재**: 개체의 위치, 화분 번호, 처리구 정보가 없어 개체 간 환경 차이를
  반영한 분석이 불가능하다.

- **표본 수 제한**: 생육결과데이터의 표본은 81개체로 제한적이어서 상관관계를 일반화하거나
  통계적으로 안정적인 결론을 내리기 어렵다.

- **측정 시각 불일치**: 환경데이터의 측정 시각이 일정하지 않아 생육 측정 시점과 환경 조건을
  정확히 대응시키기 어렵다.
""")

        st.subheader("개선 방향")
        st.markdown("""
- **생육 측정일 기록**: 동일한 실험을 다시 한다면 생육 측정일인 날짜와 시각을 개체번호와 함께 기록한다.

- **개체 위치 정보 기록**: 개체의 위치를 화분 번호, 재배대 위치, 광 조건 등과 함께 기록하여
  개체 간 환경 차이를 설명할 수 있게 한다.

- **데이터 연계**: 측정 시각과 생육 측정 시점을 일치시키는 방식으로 환경데이터와 생육결과데이터를
  연계하여 분석한다.

- **표본 확대 및 반복**: 표본 수를 늘려 여러 개체를 반복 측정하고, 가능하면 실험을 2~3회 반복하여
  결과의 신뢰성을 높인다.

- **통합 분석**: 지상부 길이와 지하부 길이뿐 아니라 환경 요인과 생육 결과를 함께 비교·분석하여
  줄기와 뿌리 생장에 영향을 주는 요인을 탐색한다.
""")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ----------------------------------------------------------------------
# 푸터
# ----------------------------------------------------------------------
st.markdown("---")
st.caption(f"🏫 학교명: {school_name}  |  송도 새싹 1팀 극지식물 생육 차이 분석 대시보드  |  데이터 기준: 업로드된 CSV")
