import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── 한글 폰트 CSS ──────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── 데이터 로딩 (캐시) ──────────────────────────────────────────────────────
@st.cache_data
def load_env(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

@st.cache_data
def load_growth(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ── 열 이름 표준화 ──────────────────────────────────────────────────────────
def std_cols(df):
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
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
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
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df


def safe_pdf_download(df, label, filename):
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    return st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


def plotly_font(fig):
    fig.update_layout(font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 페이지 설정 및 제목
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="송도 새싹 1팀", layout="wide")
st.title("🌱 송도 새싹 1팀 — 극지식물 개체별 생육 차이 분석")

# ── 사이드바: CSV 업로드 ────────────────────────────────────────────────────
env_file = st.sidebar.file_uploader("📁 환경 데이터 CSV 업로드", type="csv")
growth_file = st.sidebar.file_uploader("📁 생육 결과 데이터 CSV 업로드", type="csv")

env_df = pd.DataFrame()
growth_df = pd.DataFrame()

if env_file is not None:
    with st.sidebar.spinner("환경 데이터 로딩 중…"):
        env_df = std_cols(load_env(env_file))
        school_name = env_file.name.split("_")[0] if "_" in env_file.name else env_file.name
        st.sidebar.success(f"✅ {school_name} 환경 데이터 로드 완료 ({len(env_df)}행)")

if growth_file is not None:
    with st.sidebar.spinner("생육 데이터 로딩 중…"):
        growth_df = std_cols(load_growth(growth_file))
        school_name_g = growth_file.name.split("_")[0] if "_" in growth_file.name else growth_file.name
        st.sidebar.success(f"✅ {school_name_g} 생육 데이터 로드 완료 ({len(growth_df)}행)")

if env_df.empty and growth_df.empty:
    st.info("CSV 파일을 업로드하면 대시보드가 표시됩니다.")

# 생육 측정 항목 선택 (사이드바)
growth_items = ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
selected_items = []
if not growth_df.empty:
    selected_items = st.sidebar.multiselect(
        "📏 생육 측정 항목 선택",
        growth_items,
        default=growth_items,
    )

# 탭 선언
tabs = st.tabs([
    "📖 실험 개요",
    "🌡️ 환경 데이터",
    "📊 생육 결과",
    "🔍  가설 검증",
    "⚠️ 한계 및 개선방향",
])

# ═══════════════════════════════════════════════════════════════════════════
# 탭 1 — 실험 개요
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    try:
        st.markdown("""
**연구 배경 및 목적**

극지식물은 극한의 추위·건조·자외선 환경에서 살아남는 식물로, 기후변화 연구의 지표이자
극한 환경 농업의 가능성을 열어줍니다.

이 연구는 같은 스마트팜에서 재배한 극지식물 개체들을 비교하여,
**지상부 길이(줄기)와 지하부 길이(뿌리) 사이에 어떤 상관관계가 나타나는지**,
그리고 개체별로 생육 차이가 얼마나 큰지를 분석하는 것이 목적입니다.

**주요 가설:** 지상부 길이와 지하부 길이 사이에는 뚜렷한 상관관계가 없을 것이다.
""")
        st.divider()

        if not env_df.empty or not growth_df.empty:
            summary_rows = []

            if not growth_df.empty:
                n_individuals = len(growth_df)
                if "time" in growth_df.columns:
                    tg = growth_df["time"].dropna()
                    if len(tg) > 0:
                        g_period = f"{tg.min().strftime('%Y-%m-%d')} ~ {tg.max().strftime('%Y-%m-%d')}"
                    else:
                        g_period = "기록 없음"
                else:
                    g_period = "기록 없음 (생육 측정일 미기록)"

                g_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량",
                                       "지상부 질량", "지하부 질량"] if c in growth_df.columns]
                summary_rows.append({
                    "구분": "생육 결과 데이터",
                    "지표": "총 개체 수",
                    "값": f"{n_individuals}개체",
                    "비고": f"개체번호 범위: {growth_df['개체번호'].min()} ~ {growth_df['개체번호'].max()}",
                })
                summary_rows.append({
                    "구분": "생육 결과 데이터",
                    "지표": "측정 기간",
                    "값": g_period,
                    "비고": "생육 측정일 열이 없어 환경데이터와 시점 연결 불가",
                })
                summary_rows.append({
                    "구분": "생육 결과 데이터",
                    "지표": "측정 항목",
                    "값": ", ".join(g_cols),
                    "비고": "",
                })

            if not env_df.empty:
                te = env_df["time"].dropna()
                if len(te) > 0:
                    e_period = f"{te.min().strftime('%Y-%m-%d %H:%M')} ~ {te.max().strftime('%Y-%m-%d %H:%M')}"
                else:
                    e_period = "기록 없음"

                e_cols = [c for c in ["temperature", "humidity", "ph", "ec",
                                       "water_temp", "co2"] if c in env_df.columns]
                col_labels = {
                    "temperature": "온도(℃)", "humidity": "습도(%)",
                    "ph": "pH", "ec": "EC(mS/cm)",
                    "water_temp": "수온(℃)", "co2": "CO₂(ppm)",
                }
                summary_rows.append({
                    "구분": "환경 데이터",
                    "지표": "측정 기간",
                    "값": e_period,
                    "비고": f"총 {len(env_df)}행",
                })
                summary_rows.append({
                    "구분": "환경 데이터",
                    "지표": "측정 항목",
                    "값": ", ".join(col_labels.get(c, c) for c in e_cols),
                    "비고": "",
                })

            if summary_rows:
                st.subheader("📋 데이터 요약")
                st.dataframe(pd.DataFrame(summary_rows), width="stretch", use_container_width=True)
                st.caption("※ 생육 결과 데이터에는 생육 측정일이 기록되어 있지 않아 환경데이터와 시점을 연결할 수 없습니다.")
        else:
            st.info("데이터를 업로드하면 요약 정보가 표시됩니다.")

        st.divider()
        st.subheader("📊 주요 지표")

        if not growth_df.empty and not env_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("총 개체 수", f"{len(growth_df)} 개체")

            if "temperature" in env_df.columns:
                t_vals = env_df["temperature"].dropna()
                if len(t_vals) >= 2:
                    col2.metric("평균 온도", f"{t_vals.mean():.1f} ℃")
                else:
                    col2.metric("평균 온도", "데이터 부족")
            else:
                col2.metric("평균 온도", "해당 열 없음")

            if "humidity" in env_df.columns:
                h_vals = env_df["humidity"].dropna()
                if len(h_vals) >= 2:
                    col3.metric("평균 습도", f"{h_vals.mean():.1f} %")
                else:
                    col3.metric("평균 습도", "데이터 부족")
            else:
                col3.metric("평균 습도", "해당 열 없음")

            if "생중량" in growth_df.columns:
                gw = growth_df["생중량"].dropna()
                if len(gw) >= 2:
                    col4.metric("평균 생중량", f"{gw.mean():.2f} g")
                else:
                    col4.metric("평균 생중량", "데이터 부족")
            else:
                col4.metric("평균 생중량", "해당 열 없음")

        elif not growth_df.empty:
            st.info("환경 데이터가 업로드되면 평균 온도·습도 카드를 표시할 수 있습니다.")
            col1, col2 = st.columns(2)
            col1.metric("총 개체 수", f"{len(growth_df)} 개체")
            if "생중량" in growth_df.columns:
                gw = growth_df["생중량"].dropna()
                if len(gw) >= 2:
                    col2.metric("평균 생중량", f"{gw.mean():.2f} g")
                else:
                    col2.metric("평균 생중량", "데이터 부족")
            else:
                col2.metric("평균 생중량", "해당 열 없음")
        elif not env_df.empty:
            st.info("생육 결과 데이터가 업로드되면 개체수·생중량 카드를 표시할 수 있습니다.")
            col1, col2 = st.columns(2)
            if "temperature" in env_df.columns:
                t_vals = env_df["temperature"].dropna()
                if len(t_vals) >= 2:
                    col1.metric("평균 온도", f"{t_vals.mean():.1f} ℃")
                else:
                    col1.metric("평균 온도", "데이터 부족")
            else:
                col1.metric("평균 온도", "해당 열 없음")
            if "humidity" in env_df.columns:
                h_vals = env_df["humidity"].dropna()
                if len(h_vals) >= 2:
                    col2.metric("평균 습도", f"{h_vals.mean():.1f} %")
                else:
                    col2.metric("평균 습도", "데이터 부족")
            else:
                col2.metric("평균 습도", "해당 열 없음")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 탭 2 — 환경 데이터
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    try:
        if env_df.empty:
            st.info("사이드바에서 환경 데이터 CSV를 업로드해주세요.")
        else:
            st.subheader("🌡️ 환경 데이터 요약")

            env_num_cols = [
                c for c in ["temperature", "humidity", "ph", "ec",
                            "water_temp", "co2"]
                if c in env_df.columns
            ]
            label_map = {
                "temperature": "온도(℃)", "humidity": "습도(%)",
                "ph": "pH", "ec": "EC(mS/cm)",
                "water_temp": "수온(℃)", "co2": "CO₂(ppm)",
            }

            if env_num_cols:
                rows = []
                for c in env_num_cols:
                    s = env_df[c].dropna()
                    if len(s) >= 2:
                        mean_v = s.mean()
                        std_v = s.std()
                        rows.append({
                            "항목": label_map.get(c, c),
                            "평균": round(mean_v, 3),
                            "최소": round(s.min(), 3),
                            "최대": round(s.max(), 3),
                            "표준편차": round(std_v, 3),
                            "CV(%)": round(std_v / mean_v * 100, 2) if mean_v != 0 else np.nan,
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), width="stretch", use_container_width=True)
                    st.caption("※ CV(변동계수) = 표준편차 ÷ 평균 × 100. 단위가 다른 항목끼리 평균 대비 들쭉날쭉함을 비교할 수 있습니다.")
                    st.caption("※ 분산: 각 값이 평균에서 얼마나 떨어져 있는지 제곱해 평균낸 값. 표준편차는 분산의 제곱근입니다.")
                else:
                    st.info("표시할 수 있는 충분한 데이터가 없습니다.")
            else:
                st.info("표시할 수 있는 수치형 환경 열이 없습니다.")

            st.divider()
            st.subheader("📈 날짜별 시계열")

            if "time" in env_df.columns:
                if "temperature" in env_df.columns:
                    t_series = env_df.set_index("time").sort_index()["temperature"]
                    t_daily = t_series.resample("d").mean().dropna()
                    if len(t_daily) >= 2:
                        t_min = t_series.resample("d").min().dropna()
                        t_max = t_series.resample("d").max().dropna()
                        t_df = pd.DataFrame({"평균": t_daily, "최소": t_min, "최대": t_max}).dropna()

                        fig_t = go.Figure()
                        fig_t.add_trace(go.Scatter(
                            x=t_df.index, y=t_df["평균"],
                            mode="lines", name="평균 온도",
                            line=dict(color="#e74c3c", width=2),
                        ))
                        fig_t.add_trace(go.Scatter(
                            x=t_df.index, y=t_df["최소"],
                            mode="lines", name="최저 온도",
                            line=dict(color="#e74c3c", width=1, dash="dot"), opacity=0.6,
                        ))
                        fig_t.add_trace(go.Scatter(
                            x=t_df.index, y=t_df["최대"],
                            mode="lines", name="최고 온도",
                            line=dict(color="#e74c3c", width=1, dash="dot"), opacity=0.6,
                        ))
                        fig_t.update_layout(
                            title="날짜별 평균·최저·최고 온도",
                            xaxis_title="날짜", yaxis_title="온도 (℃)",
                            height=280,
                        )
                        plotly_font(fig_t)
                        st.plotly_chart(fig_t, width="stretch")
                    else:
                        st.info("온도 데이터의 일별 평균 계산에 필요한 데이터가 부족합니다.")

                if "humidity" in env_df.columns:
                    h_series = env_df.set_index("time").sort_index()["humidity"]
                    h_daily = h_series.resample("d").mean().dropna()
                    if len(h_daily) >= 2:
                        h_min = h_series.resample("d").min().dropna()
                        h_max = h_series.resample("d").max().dropna()
                        h_df = pd.DataFrame({"평균": h_daily, "최소": h_min, "최대": h_max}).dropna()

                        fig_h = go.Figure()
                        fig_h.add_trace(go.Scatter(
                            x=h_df.index, y=h_df["평균"],
                            mode="lines", name="평균 습도",
                            line=dict(color="#3498db", width=2),
                        ))
                        fig_h.add_trace(go.Scatter(
                            x=h_df.index, y=h_df["최소"],
                            mode="lines", name="최저 습도",
                            line=dict(color="#3498db", width=1, dash="dot"), opacity=0.6,
                        ))
                        fig_h.add_trace(go.Scatter(
                            x=h_df.index, y=h_df["최대"],
                            mode="lines", name="최고 습도",
                            line=dict(color="#3498db", width=1, dash="dot"), opacity=0.6,
                        ))
                        fig_h.update_layout(
                            title="날짜별 평균·최저·최고 습도",
                            xaxis_title="날짜", yaxis_title="습도 (%)",
                            height=280,
                        )
                        plotly_font(fig_h)
                        st.plotly_chart(fig_h, width="stretch")
                    else:
                        st.info("습도 데이터의 일별 평균 계산에 필요한 데이터가 부족합니다.")

                ph_ec_cols = [c for c in ["ph", "ec"] if c in env_df.columns]
                if ph_ec_cols:
                    pe_series = env_df.set_index("time").sort_index()[ph_ec_cols]
                    pe_daily = pe_series.resample("d").mean().dropna()
                    if len(pe_daily) >= 2:
                        colors_pe = {"ph": "#9b59b6", "ec": "#2ecc71"}
                        fig_pe = go.Figure()
                        for c in ph_ec_cols:
                            fig_pe.add_trace(go.Scatter(
                                x=pe_daily.index,
                                y=pe_daily[c],
                                mode="lines",
                                name=f"{c.upper()}",
                                line=dict(color=colors_pe.get(c, "#888"), width=2),
                            ))
                        fig_pe.update_layout(
                            title="날짜별 pH · EC 평균 추이",
                            xaxis_title="날짜", yaxis_title="값",
                            height=280,
                        )
                        plotly_font(fig_pe)
                        st.plotly_chart(fig_pe, width="stretch")
                    else:
                        st.info("pH·EC 데이터의 일별 평균 계산에 필요한 데이터가 부족합니다.")
                else:
                    st.info("pH 또는 EC 열이 없어 해당 시계열을 표시할 수 없습니다.")

                # 이상치 탐지 (IQR)
                st.divider()
                st.subheader("🔍 이상치 탐지 (IQR 기준)")
                outlier_rows = []
                for c in env_num_cols:
                    s = env_df[c].dropna()
                    if len(s) < 4:
                        continue
                    q1 = s.quantile(0.25)
                    q3 = s.quantile(0.75)
                    iqr = q3 - q1
                    lo = q1 - 1.5 * iqr
                    hi = q3 + 1.5 * iqr
                    n_out = int((s < lo).sum() + (s > hi).sum())
                    outlier_rows.append({
                        "항목": label_map.get(c, c),
                        "Q1": round(q1, 3),
                        "Q3": round(q3, 3),
                        "IQR": round(iqr, 3),
                        "이상치 개수": n_out,
                        "하한": round(lo, 3),
                        "상한": round(hi, 3),
                    })
                if outlier_rows:
                    st.dataframe(pd.DataFrame(outlier_rows), width="stretch", use_container_width=True)
                else:
                    st.info("이상치를 계산할 수 있는 충분한 데이터가 없습니다.")

                # CV 비교 막대그래프
                st.divider()
                st.subheader("📊 환경 데이터별 CV(변동계수) 비교")
                cv_rows = []
                for c in env_num_cols:
                    s = env_df[c].dropna()
                    if len(s) >= 2 and s.mean() != 0:
                        cv_rows.append({
                            "항목": label_map.get(c, c),
                            "CV(%)": round(s.std() / s.mean() * 100, 2),
                        })
                if cv_rows:
                    fig_cv = px.bar(
                        cv_rows, x="항목", y="CV(%)",
                        color="CV(%)", color_continuous_scale="RdYlGn_r",
                        labels={"항목": "항목", "CV(%)": "CV (%)"},
                        title="환경 측정 항목별 변동계수(CV) 비교",
                        height=320,
                    )
                    fig_cv.update_layout(xaxis_title="")
                    plotly_font(fig_cv)
                    st.plotly_chart(fig_cv, width="stretch")
                else:
                    st.info("CV를 계산할 수 있는 충분한 데이터가 없습니다.")
            else:
                st.info("환경 데이터에 time 열이 없어 시계열 분석을 표시할 수 없습니다.")

            with st.expander("📄 환경 데이터 원본 테이블 및 CSV 다운로드"):
                st.dataframe(env_df, width="stretch", use_container_width=True)
                safe_pdf_download(env_df, "📥 환경 데이터 CSV 다운로드", "환경데이터.csv")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 탭 3 — 생육 결과
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    try:
        if growth_df.empty:
            st.info("사이드바에서 생육 결과 데이터 CSV를 업로드해주세요.")
        else:
            # 핵심 결과 카드
            st.subheader("🥇 핵심 결과")
            c1, c2, c3 = st.columns(3)

            if "생중량" in growth_df.columns:
                gw = growth_df["생중량"].dropna()
                if len(gw) >= 2:
                    avg_gw = gw.mean()
                    c1.metric("📦 평균 생중량", f"{avg_gw:.2f} g")
                    max_idx = gw.idxmax()
                    min_idx = gw.idxmin()
                    max_id = growth_df.loc[max_idx, "개체번호"]
                    min_id = growth_df.loc[min_idx, "개체번호"]
                    c2.metric("🏆 최대 생중량", f"{gw.max():.2f} g",
                              delta=f"개체 #{max_id}")
                    c3.metric("📉 최소 생중량", f"{gw.min():.2f} g",
                              delta=f"개체 #{min_id}")
                else:
                    c1.metric("📦 평균 생중량", "데이터 부족")
                    c2.metric("🏆 최대 생중량", "데이터 부족")
                    c3.metric("📉 최소 생중량", "데이터 부족")
            else:
                st.info("생중량 열이 없어 핵심 결과 카드를 표시할 수 없습니다.")

            st.divider()

            # 개체별 생육 비교 막대 그래프
            st.subheader("📊 개체별 생육 비교")
            if selected_items:
                available = [it for it in selected_items
                             if it in growth_df.columns
                             and growth_df[it].dropna().shape[0] >= 2]
                if available:
                    color_map = {
                        "생중량": "#2ecc71",
                        "잎 수": "#3498db",
                        "지상부 길이": "#e74c3c",
                        "지하부 길이": "#9b59b6",
                    }
                    n_cols = min(len(available), 2)
                    chart_idx = 0
                    for r in range((len(available) + n_cols - 1) // n_cols):
                        row_cols = st.columns(n_cols)
                        for ci in range(n_cols):
                            if chart_idx < len(available):
                                item = available[chart_idx]
                                vals = growth_df[item].dropna()
                                avg_v = vals.mean()
                                fig = px.bar(
                                    x=growth_df["개체번호"],
                                    y=growth_df[item],
                                    color=growth_df[item],
                                    color_continuous_scale=[
                                        (0, "#aed6f1"),
                                        (1, color_map.get(item, "#888")),
                                    ],
                                    labels={"x": "개체번호", "y": item, "color": item},
                                    title=f"개체별 {item}",
                                    height=300,
                                )
                                if item == "생중량":
                                    fig.add_hline(
                                        y=avg_v,
                                        line_dash="dash",
                                        line_color="orange",
                                        annotation_text=f"평균 {avg_v:.2f}",
                                        annotation_position="top right",
                                    )
                                plotly_font(fig)
                                row_cols[ci].plotly_chart(fig, width="stretch")
                                chart_idx += 1
                            else:
                                row_cols[ci].empty()
                else:
                    st.info("선택한 생육 측정 항목이 데이터에 없거나 데이터가 부족합니다.")
            else:
                st.info("사이드바에서 생육 측정 항목을 선택해주세요.")

            st.divider()

            # 생중량 분포 (박스플롯)
            st.subheader("📦 생육 측정 항목 분포 (박스플롯)")
            plot_cols = [c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"]
                         if c in growth_df.columns]
            if plot_cols:
                plot_col = st.selectbox(
                    "분포를 확인할 측정 항목 선택",
                    plot_cols,
                    index=plot_cols.index("생중량") if "생중량" in plot_cols else 0,
                )
                vals = growth_df[plot_col].dropna()
                if len(vals) >= 2:
                    fig_box = px.box(
                        y=vals,
                        labels={"y": plot_col},
                        title=f"{plot_col} 분포 (박스플롯)",
                        height=300,
                    )
                    fig_box.update_layout(yaxis_title=plot_col)
                    plotly_font(fig_box)
                    st.plotly_chart(fig_box, width="stretch")

                    stats_c1, stats_c2 = st.columns(2)
                    stats_c1.write(f"평균: {vals.mean():.2f}")
                    stats_c1.write(f"중앙값: {vals.median():.2f}")
                    stats_c1.write(f"표준편차: {vals.std():.2f}")
                    stats_c1.write(f"CV(%): {vals.std() / vals.mean() * 100:.1f}")
                    stats_c2.write(f"최소값: {vals.min():.2f}")
                    stats_c2.write(f"최대값: {vals.max():.2f}")
                    stats_c2.write(f"Q1: {vals.quantile(0.25):.2f}")
                    stats_c2.write(f"Q3: {vals.quantile(0.75):.2f}")
                else:
                    st.info(f"{plot_col} 데이터가 부족하여 분포를 표시할 수 없습니다.")
            else:
                st.info("분포를 표시할 수 있는 측정 항목이 없습니다.")

            with st.expander("📄 생육 결과 데이터 원본 테이블 및 CSV 다운로드"):
                st.dataframe(growth_df, width="stretch", use_container_width=True)
                safe_pdf_download(growth_df, "📥 생육 데이터 CSV 다운로드", "생육결과데이터.csv")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 탭 4 — 가설 검증
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    try:
        if growth_df.empty:
            st.info("사이드바에서 생육 결과 데이터 CSV를 업로드해주세요.")
        else:
            shoot = growth_df["지상부 길이"].dropna() if "지상부 길이" in growth_df.columns else pd.Series(dtype=float)
            root = growth_df["지하부 길이"].dropna() if "지하부 길이" in growth_df.columns else pd.Series(dtype=float)

            if len(shoot) >= 2 and len(root) >= 2:
                df_corr_base = pd.DataFrame({"지상부 길이": shoot, "지하부 길이": root}).dropna()
                if len(df_corr_base) >= 2:
                    corr_val = df_corr_base["지상부 길이"].corr(df_corr_base["지하부 길이"])

                    # 상관계수 강도 판정
                    abs_r = abs(corr_val)
                    if abs_r < 0.3:
                        strength = "매우 약한 상관"
                    elif abs_r < 0.5:
                        strength = "약한 상관"
                    elif abs_r < 0.7:
                        strength = "중간 상관"
                    else:
                        strength = "강한 상관"

                    # 핵심 결과 카드
                    st.subheader("🥇 가설 검증 핵심 결과")
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("지상부 길이 평균", f"{shoot.mean():.1f} mm")
                    k2.metric("지하부 길이 평균", f"{root.mean():.1f} mm")
                    k3.metric("피어슨 상관계수 (r)", f"{corr_val:.3f}")
                    k4.metric("상관 강도", strength)

                    st.caption(
                        "※ 피어슨 상관계수(r): -1 ~ +1 사이 값. 0에 가까우면 선형 상관관계가 거의 없음을 의미합니다."
                    )
                    st.caption(
                        "※ 상관계수가 0에 가깝다는 결과가 두 지표 생장이 완전히 무관하다는 뜻은 아니며, "
                        "선형 관계가 거의 없다는 의미로 해석하는 것이 적절합니다."
                    )

                    st.divider()

                    # 기초 통계분석표
                    st.subheader("📋 지상부·지하부 길이 기초 통계분석표")
                    stat_rows = []
                    for col_name, series in [("지상부 길이", shoot), ("지하부 길이", root)]:
                        if len(series) >= 2:
                            mean_v = series.mean()
                            stat_rows.append({
                                "측정 항목": col_name,
                                "개체 수": len(series),
                                "평균": round(mean_v, 2),
                                "표준편차": round(series.std(), 2),
                                "CV(%)": round(series.std() / mean_v * 100, 2) if mean_v != 0 else np.nan,
                                "최소값": round(series.min(), 2),
                                "중앙값": round(series.median(), 2),
                                "최대값": round(series.max(), 2),
                                "범위": round(series.max() - series.min(), 2),
                            })
                    st.dataframe(pd.DataFrame(stat_rows), width="stretch", use_container_width=True)

                    st.divider()

                    # 평균 비율 원그래프
                    st.subheader("🥧 지상부·지하부 길이 평균 비율")
                    avg_shoot = shoot.mean()
                    avg_root = root.mean()
                    avg_total = avg_shoot + avg_root
                    if avg_total > 0:
                        ratio_df = pd.DataFrame({
                            "구분": ["지상부 길이", "지하부 길이"],
                            "평균(mm)": [round(avg_shoot, 1), round(avg_root, 1)],
                            "비율(%)": [
                                round(avg_shoot / avg_total * 100, 1),
                                round(avg_root / avg_total * 100, 1),
                            ],
                        })
                        fig_pie = px.pie(
                            ratio_df, names="구분", values="비율(%)",
                            color="구분",
                            color_discrete_map={
                                "지상부 길이": "#e74c3c",
                                "지하부 길이": "#9b59b6",
                            },
                            title=f"지상부 : 지하부 = {avg_shoot:.1f}mm : {avg_root:.1f}mm",
                            height=300,
                        )
                        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                        plotly_font(fig_pie)
                        st.plotly_chart(fig_pie, width="stretch")
                        st.dataframe(ratio_df, width="stretch", use_container_width=True)
                    else:
                        st.info("평균 길이를 계산할 수 있는 데이터가 부족합니다.")

                    st.divider()

                    # 산점도 (추세선: np.polyfit)
                    st.subheader("🔥 지상부·지하부 길이 상관관계 산점도 (추세선 포함)")
                    fig_scatter = go.Figure()

                    # 개체번호 텍스트 준비
                    if "개체번호" in growth_df.columns:
                        id_text = growth_df.loc[df_corr_base.index, "개체번호"].values.astype(str)
                    else:
                        id_text = df_corr_base.index.astype(str).values

                    fig_scatter.add_trace(go.Scatter(
                        x=df_corr_base["지상부 길이"],
                        y=df_corr_base["지하부 길이"],
                        mode="markers",
                        name="개체별 데이터",
                        marker=dict(color="#3498db", size=8, opacity=0.75),
                        hovertemplate=(
                            "개체: %{text}<br>"
                            "지상부: %{x:.1f}mm<br>"
                            "지하부: %{y:.1f}mm<extra></extra>"
                        ),
                        text=id_text,
                    ))

                    x_vals = df_corr_base["지상부 길이"].values
                    y_vals = df_corr_base["지하부 길이"].values
                    if len(x_vals) >= 2:
                        slope, intercept = np.polyfit(x_vals, y_vals, 1)
                        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                        y_line = slope * x_line + intercept
                        fig_scatter.add_trace(go.Scatter(
                            x=x_line, y=y_line,
                            mode="lines",
                            name=f"추세선 (기울기={slope:.4f})",
                            line=dict(color="#e74c3c", width=2, dash="dash"),
                        ))

                    fig_scatter.update_layout(
                        title=f"지상부 길이 vs 지하부 길이 (r = {corr_val:.3f})",
                        xaxis_title="지상부 길이 (mm)",
                        yaxis_title="지하부 길이 (mm)",
                        height=420,
                        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
                    )
                    plotly_font(fig_scatter)
                    st.plotly_chart(fig_scatter, width="stretch")

                    st.caption(
                        f"※ 추세선 기울기 = {slope:.4f}. "
                        f"지상부 길이가 1mm 증가할 때 지하부 길이는 약 {slope:.3f}mm 변화하는 연관성을 보입니다."
                    )

                    st.divider()

                    # 가설 검증 결과 분석 설명
                    st.subheader("📝 가설 검증 결과 분석")
                    if abs_r < 0.3:
                        conclusion = (
                            f"지상부 길이와 지하부 길이 사이의 피어슨 상관계수는 **r = {corr_val:.3f}**로, "
                            "0에 매우 가까워 **뚜렷한 선형 상관관계가 거의 없습니다.** "
                            "이는 \"지상부 길이와 지하부 길이 사이에는 뚜렷한 상관관계가 없을 것\"이라는 "
                            "**연구 가설을 지지하는 결과**입니다.\n\n"
                            "식물의 줄기와 뿌리는 같은 개체 안에서 자라지만 서로 다른 환경 조건과 "
                            "생리적 요인의 영향을 받아 성장할 수 있어, 두 길이 사이에 반드시 같은 방향으로 "
                            "자라는 관계가 나타나지는 않습니다.\n\n"
                            "다만 상관계수가 0에 가깝다는 결과가 두 지표의 생장이 완전히 무관하다는 뜻은 아니며, "
                            "**선형 관계가 거의 없다**는 의미로 해석하는 것이 적절합니다."
                        )
                    elif abs_r < 0.5:
                        direction = "양의" if corr_val > 0 else "음의"
                        conclusion = (
                            f"지상부 길이와 지하부 길이 사이의 피어슨 상관계수는 **r = {corr_val:.3f}**로, "
                            f"**약한 {direction} 상관관계**가 관찰됩니다. "
                            "일부 선형 관계가 있음을 시사하지만, 강한 상관은 보이지 않아 "
                            "가설을 완전히 기각하기는 어렵습니다."
                        )
                    else:
                        conclusion = (
                            f"지상부 길이와 지하부 길이 사이의 피어슨 상관계수는 **r = {corr_val:.3f}**로, "
                            "**뚜렷한 상관관계**가 관찰됩니다. "
                            "이는 연구 가설과 다른 방향의 결과입니다."
                        )
                    st.markdown(conclusion)

                else:
                    st.info("지상부 길이와 지하부 길이 데이터를 충분히 확보할 수 없어 상관관계 분석을 건너뜁니다.")
            else:
                st.info("지상부 길이 또는 지하부 길이 열이 없거나 데이터가 부족하여 가설 검증을 표시할 수 없습니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 탭 5 — 한계 및 개선방향
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    try:
        st.subheader("⚠️ 연구 한계")

        limits = [
            ("📅 생육 측정일 미기록",
             "생육결과데이터에는 생육 측정일이 기록되어 있지 않아 "
             "환경데이터의 측정 시점과 생육 결과를 같은 시점에 연결하여 분석하기 어렵다."),
            ("📍 개체 위치·처리구 정보 부재",
             "개체의 위치, 화분 번호, 처리구 정보가 없어 "
             "개체 간 환경 차이를 반영한 분석이 불가능하다."),
            ("🔢 제한된 표본 수",
             "생육결과데이터의 표본은 81개체로 제한적이어서 "
             "상관관계를 일반화하거나 통계적으로 안정적인 결론을 내리기 어렵다."),
            ("⏱️ 환경데이터 측정 시각 불일정",
             "환경데이터의 측정 시각이 일정하지 않아 "
             "생육 측정 시점과 환경 조건을 정확히 일치시키기 어렵다."),
            ("📊 부족한 통합 변수",
             "환경데이터에는 온도, 습도, EC, pH, 수온, CO₂가 있으나 "
             "생육결과데이터에는 잎 수, 지상부 길이, 지하부 길이, 생중량 및 "
             "지상부·지하부 질량만 있어 환경적 요인과 생장 특성을 "
             "통합적으로 분석하기 위한 변수가 충분하지 않다."),
        ]
        for title, desc in limits:
            st.markdown(f"**{title}**")
            st.info(desc)

        st.divider()
        st.subheader("💡 개선 방향")

        improvements = [
            ("📅 생육 측정일 기록",
             "동일한 실험을 다시 한다면 생육 측정일인 날짜와 시각을 "
             "개체번호와 함께 기록한다."),
            ("📍 개체 위치 정보 기록",
             "개체의 위치를 화분 번호, 재배대 위치, 광 조건 등과 함께 기록하여 "
             "개체 간 환경 차이를 설명할 수 있게 한다."),
            ("⏱️ 환경데이터와 생육데이터 시점 연계",
             "측정 시각과 생육 측정 시점을 일치시키는 방식으로 "
             "환경데이터와 생육결과데이터를 연계하여 분석한다."),
            ("🔢 표본 수 확대 및 반복 실험",
             "표본 수를 늘려 여러 개체를 반복 측정하고, "
             "가능하면 실험을 2~3회 반복하여 결과의 신뢰성을 높인다."),
            ("🔬 환경 요인-생육 통합 분석",
             "지상부 길이와 지하부 길이뿐 아니라 환경 요인과 생육 결과를 "
             "함께 비교·분석하여 줄기와 뿌리 생장에 영향을 주는 요인을 탐색한다."),
            ("📝 이상치·급변동 구간 현장 기록",
             "환경데이터에서 확인된 EC·pH 급변동, 온도 급상승 구간이나 "
             "생육 수치가 매우 작은 개체들에 대해 "
             "당시 장비 상태·측정 방식·재배 상황을 메모로 남겨 판단 근거를 확보한다."),
        ]
        for title, desc in improvements:
            st.markdown(f"**{title}**")
            st.success(desc)

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
