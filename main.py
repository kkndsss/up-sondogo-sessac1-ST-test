import io, os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_absolute_error
import streamlit as st

# ── 한글 폰트 CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ── CSV 읽기 (utf-8-sig → cp949) ──────────────────────────────
@st.cache_data
def load_and_std(file, enc_list=("utf-8-sig", "cp949")):
    data = file.getvalue()
    df = pd.DataFrame()
    for enc in enc_list:
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc)
            if not df.empty:
                break
        except Exception:
            pass
    if df.empty:
        return pd.DataFrame()
    return std_cols(df)

# ── 열 이름 표준화 (KeyError 방지) ──────────────────────────────
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
                errors="coerce"
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

# ── 통계 helper ────────────────────────────────────────────────
def safe_stats(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n < 2:
        return dict(n=0, mean=None, std=None, min=None, max=None, cv=None)
    mean = s.mean()
    std = s.std()
    return dict(
        n=n,
        mean=mean,
        std=std,
        min=s.min(),
        max=s.max(),
        cv=(std / mean * 100) if mean != 0 else None,
    )

def fmt_stat(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if abs(v) >= 100:
        return f"{v:,.1f}"
    if abs(v) >= 10:
        return f"{v:,.2f}"
    return f"{v:,.3f}"

# ── CSV 다운로드 (한글 깨짐 방지) ──────────────────────────────
def download_csv(df, label):
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label, csv_data, file_name=f"{label}.csv", mime="text/csv")

# ── Plotly 한글 폰트 적용 (width 제외) ─────────────────────────
def fig_layout(fig):
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=40, b=40),
    )

# ── 평균 막대 그래프 공통 함수 ────────────────────────────────
def bar_avg(df, col, title, ylabel):
    if col not in df.columns or df[col].dropna().empty:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        return None
    fig = go.Figure()
    fig.add_bar(x=[col], y=[vals.mean()], name="평균")
    fig.update_layout(
        title=title,
        xaxis_title="측정 항목",
        yaxis_title=ylabel,
        showlegend=False,
    )
    return fig

def bar_individual(df, col, title, ylabel):
    if col not in df.columns or df[col].dropna().empty:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    # 원래 DataFrame에서 해당 열의 유효값만 정렬 (개체번호 정보 유지)
    tmp = df[[c for c in df.columns if c != col]].copy()
    tmp["_val"] = pd.to_numeric(df[col], errors="coerce")
    tmp = tmp.dropna(subset=["_val"]).sort_values("_val", ascending=False)
    labels = tmp["개체번호"].astype(str).tolist() if "개체번호" in tmp.columns else [str(i) for i in range(len(tmp))]
    fig = go.Figure()
    colors = ["#d64545" if i < 3 else "#4578d6" if i >= len(tmp) - 3 else "#9bb7e8"
              for i in range(len(tmp))]
    fig.add_bar(x=labels, y=tmp["_val"].tolist(), marker_color=colors, name="개체별 값")
    fig.add_hline(y=s.mean(), line_dash="dot", line_color="#ff7f0e",
                  annotation_text=f"평균 {s.mean():.2f}", annotation_position="top left")
    fig.update_layout(title=title, xaxis_title="개체", yaxis_title=ylabel,
                      xaxis=dict(tickangle=45))
    return fig

# ════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="송도 새싹 1팀", layout="wide")
    st.title("송도 새싹 1팀")

    # ── 학교명 추출 (파일명 앞부분) ────────────────────────────
    school_name = ""
    if "growth_file_name" in st.session_state:
        raw = os.path.splitext(st.session_state.growth_file_name)[0]
        school_name = raw.split("_")[0].strip() or "업로드된 학교"

    tabs = st.tabs([
        "📖 실험 개요",
        "🌡️ 환경 데이터",
        "📊 생육 결과",
        "🔗 상관관계 분석",
        "📈 상위·하위 그룹 비교",
        "🧮 잎 수 기반 생중량 예측",
    ])

    with tabs[0]:
        try:
            st.header("연구 배경 및 목적")
            st.markdown("""
**극지식물**은 극한의 추위·건조·자외선에서 살아남는 식물로, 기후변화 연구 지표이자 극한 환경 농업의 가능성을 열어줍니다.  
스마트팜의 온도·습도·EC·pH는 식물 생육의 핵심 변수이며, 특히 **지상부 길이(줄기)와 지하부 길이(뿌리)**는 서로 다른 환경·생리적 요인의 영향을 받아 성장할 수 있습니다.

이 대시보드는 우리 학교 극지식물 **개체별 생육 차이**를 시각화하고, 지상부 길이와 지하부 길이 사이에 뚜렷한 상관관계가 있는지 탐구합니다.
""")

            st.header("데이터 요약")
            rows = []
            if env_df is not None and not env_df.empty and "time" in env_df.columns:
                t = env_df["time"].dropna()
                rows.append(["환경 데이터", f"{len(env_df):,}개 행",
                             f"{t.min().strftime('%Y-%m-%d %H:%M') if not t.empty else '-'}"
                             f" ~ {t.max().strftime('%Y-%m-%d %H:%M') if not t.empty else '-'}",
                             ", ".join([c for c in env_df.columns if c != "time"])])
            else:
                rows.append(["환경 데이터", "업로드되지 않음", "-", "-"])

            if growth_df is not None and not growth_df.empty:
                n = len(growth_df)
                ind_col = "개체번호"
                rng = "-"
                if ind_col in growth_df.columns:
                    vals = growth_df[ind_col].dropna()
                    rng = f"{vals.min()} ~ {vals.max()}" if not vals.empty else "-"
                rows.append(["생육 결과 데이터", f"{n}개체",
                             "수확 시점 (개체별 측정일 없음)",
                             ", ".join([c for c in growth_df.columns if c != ind_col])])
            else:
                rows.append(["생육 결과 데이터", "업로드되지 않음", "-", "-"])

            st.dataframe(pd.DataFrame(rows, columns=["데이터", "규모", "측정 기간", "측정 항목"]),
                         width="stretch", hide_index=True)

            st.header("주요 지표 카드")
            c1, c2, c3, c4 = st.columns(4)
            # 총 개체수
            if growth_df is not None and not growth_df.empty:
                agg = safe_stats(growth_df["개체번호"].dropna())
                total = len(growth_df) if len(growth_df) else 0
                c1.metric("총 개체수", f"{total}개체")
            else:
                c1.metric("총 개체수", "데이터 없음")

            # 평균 온도
            if env_df is not None and not env_df.empty and "temperature" in env_df.columns:
                st_ = safe_stats(env_df["temperature"])
                c2.metric("평균 온도", f"{fmt_stat(st_['mean'])}℃",
                          delta=f"CV {fmt_stat(st_['cv'])}%" if st_['cv'] else None)
            else:
                c2.metric("평균 온도", "데이터 없음")

            # 평균 습도
            if env_df is not None and not env_df.empty and "humidity" in env_df.columns:
                st_ = safe_stats(env_df["humidity"])
                c3.metric("평균 습도", f"{fmt_stat(st_['mean'])}%",
                          delta=f"CV {fmt_stat(st_['cv'])}%" if st_['cv'] else None)
            else:
                c3.metric("평균 습도", "데이터 없음")

            # 평균 생중량
            if growth_df is not None and not growth_df.empty and "생중량" in growth_df.columns:
                st_ = safe_stats(growth_df["생중량"])
                c4.metric("평균 생중량", f"{fmt_stat(st_['mean'])}g",
                          delta=f"CV {fmt_stat(st_['cv'])}%" if st_['cv'] else None)
            else:
                c4.metric("평균 생중량", "데이터 없음")
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    with tabs[1]:
        try:
            st.header("환경 데이터 요약 (평균 · 표준편차 · 최소 · 최대 · CV)")
            if env_df is None or env_df.empty:
                st.info("☝️ 환경 CSV를 업로드하면 여기에 요약이 표시됩니다.")
            else:
                env_cols = [c for c in ["temperature", "humidity", "ph", "ec"]
                            if c in env_df.columns]
                if not env_cols:
                    st.info("☝️ 온도·습도·pH·EC 열이 없어 환경 요약을 표시할 수 없습니다.")
                else:
                    st_ = []
                    for c in env_cols:
                        s = safe_stats(env_df[c])
                        st_.append([c, fmt_stat(s["mean"]), fmt_stat(s["std"]),
                                    fmt_stat(s["min"]), fmt_stat(s["max"]),
                                    fmt_stat(s["cv"])])
                    st.dataframe(
                        pd.DataFrame(st_, columns=["항목", "평균", "표준편차", "최소", "최대", "CV(%)"]),
                        width="stretch", hide_index=True,
                    )
                    st.caption("※ 표준편차: 데이터가 평균에서 얼마나 퍼져 있는지 나타내는 값 (분산의 제곱근)")
                    st.caption("※ CV(%): 표준편차÷평균×100, 단위가 다른 항목끼리도 평균 대비 변동성을 비교 가능")

            st.header("날짜별 평균 시계열")
            if env_df is None or env_df.empty or "time" not in env_df.columns:
                st.info("☝️ 환경 CSV의 time 열이 필요해 시계열을 표시할 수 없습니다.")
            else:
                ed = env_df.dropna(subset=["time"]).copy()
                if ed.empty:
                    st.info("☝️ time 값이 없어 시계열을 표시할 수 없습니다.")
                else:
                    ed["day"] = ed["time"].dt.floor("D")

                    if "temperature" in ed.columns:
                        daily_t = ed.groupby("day")["temperature"].agg(["mean", "min", "max"])
                        daily_t = daily_t.dropna()
                        if len(daily_t) >= 1:
                            fig_t = go.Figure()
                            fig_t.add_trace(go.Scatter(x=daily_t.index, y=daily_t["mean"],
                                                        mode="lines", name="평균 온도",
                                                        line=dict(color="#e74c3c", width=2)))
                            fig_t.add_trace(go.Scatter(x=daily_t.index,
                                                        y=daily_t["min"], mode="lines",
                                                        name="최저", line=dict(color="#e74c3c", width=1,
                                                                              dash="dot"),
                                                        showlegend=True))
                            fig_t.add_trace(go.Scatter(x=daily_t.index,
                                                        y=daily_t["max"], mode="lines",
                                                        name="최고", line=dict(color="#e74c3c", width=1,
                                                                              dash="dot"),
                                                        showlegend=True,
                                                        fill=None))
                            # 범위 영역 (min~max 사이)
                            fig_t.add_trace(go.Scatter(x=daily_t.index,
                                                        y=daily_t["max"], mode="lines",
                                                        name="최고(범위)", line=dict(color="#e74c3c", width=0),
                                                        fill="tonexty", fillcolor="rgba(231,76,60,0.12)"))
                            fig_t.update_layout(
                                title="날짜별 평균 온도 변화",
                                xaxis_title="날짜",
                                yaxis_title="온도 (℃)",
                                showlegend=True,
                            )
                            fig_t.data[2].showlegend = False
                            fig_layout(fig_t)
                            st.plotly_chart(fig_t, width="stretch")

                    if "humidity" in ed.columns:
                        daily_h = ed.groupby("day")["humidity"].agg(["mean", "min", "max"])
                        daily_h = daily_h.dropna()
                        if len(daily_h) >= 1:
                            fig_h = go.Figure()
                            fig_h.add_trace(go.Scatter(x=daily_h.index, y=daily_h["mean"],
                                                        mode="lines", name="평균 습도",
                                                        line=dict(color="#3498db", width=2)))
                            fig_h.add_trace(go.Scatter(x=daily_h.index,
                                                        y=daily_h["max"], mode="lines",
                                                        name="최고(범위)", line=dict(color="#3498db", width=0),
                                                        fill="tonexty", fillcolor="rgba(52,152,219,0.15)"))
                            fig_h.update_layout(
                                title="날짜별 평균 습도 변화",
                                xaxis_title="날짜",
                                yaxis_title="습도 (%)",
                            )
                            fig_layout(fig_h)
                            st.plotly_chart(fig_h, width="stretch")

                    if "ph" in ed.columns:
                        daily_ph = ed.groupby("day")["ph"].agg(["mean", "min", "max"]).dropna()
                        if len(daily_ph) >= 1:
                            fig_ph = go.Figure()
                            fig_ph.add_trace(go.Scatter(x=daily_ph.index, y=daily_ph["mean"],
                                                        mode="lines", name="평균 pH",
                                                        line=dict(color="#8e44ad", width=2)))
                            fig_ph.add_trace(go.Scatter(x=daily_ph.index,
                                                        y=daily_ph["max"], mode="lines",
                                                        name="범위", line=dict(color="#8e44ad", width=0),
                                                        fill="tonexty", fillcolor="rgba(142,68,173,0.15)"))
                            fig_ph.update_layout(
                                title="날짜별 평균 pH 변화",
                                xaxis_title="날짜",
                                yaxis_title="pH",
                            )
                            fig_layout(fig_ph)
                            st.plotly_chart(fig_ph, width="stretch")
                        else:
                            st.info("☝️ pH 데이터가 충분하지 않아 그래프를 표시할 수 없습니다.")
                    else:
                        st.info("☝️ pH 열이 없어 pH 시계열을 건너뜁니다.")

                    if "ec" in ed.columns:
                        daily_ec = ed.groupby("day")["ec"].agg(["mean", "min", "max"]).dropna()
                        if len(daily_ec) >= 1:
                            fig_ec = go.Figure()
                            fig_ec.add_trace(go.Scatter(x=daily_ec.index, y=daily_ec["mean"],
                                                         mode="lines", name="평균 EC",
                                                         line=dict(color="#27ae60", width=2)))
                            fig_ec.add_trace(go.Scatter(x=daily_ec.index,
                                                         y=daily_ec["max"], mode="lines",
                                                         name="범위", line=dict(color="#27ae60", width=0),
                                                         fill="tonexty", fillcolor="rgba(39,174,96,0.15)"))
                            fig_ec.update_layout(
                                title="날짜별 평균 EC 변화",
                                xaxis_title="날짜",
                                yaxis_title="EC (mS/cm)",
                            )
                            fig_layout(fig_ec)
                            st.plotly_chart(fig_ec, width="stretch")
                        else:
                            st.info("☝️ EC 데이터가 충분하지 않아 그래프를 표시할 수 없습니다.")
                    else:
                        st.info("☝️ EC 열이 없어 EC 시계열을 건너뜁니다.")

            with st.expander("📋 환경 데이터 원본 테이블 보기 + CSV 다운로드"):
                if env_df is not None and not env_df.empty:
                    display_cols = [c for c in ["time", "temperature", "humidity", "ph", "ec"] if c in env_df.columns]
                    if len(display_cols) <= 1:
                        st.info("☝️ 표시할 환경 열이 충분하지 않습니다.")
                    else:
                        st.dataframe(env_df[display_cols], width="stretch")
                    download_csv(env_df, "환경데이터")
                else:
                    st.info("☝️ 환경 CSV를 업로드하면 여기에 원본 테이블이 표시됩니다.")
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    with tabs[2]:
        try:
            st.header("🥇 핵심 결과 카드")
            if growth_df is None or growth_df.empty:
                st.info("☝️ 생육 CSV를 업로드하면 여기에 핵심 결과가 표시됩니다.")
            else:
                col1, col2, col3 = st.columns(3)
                if "생중량" in growth_df.columns:
                    s = safe_stats(growth_df["생중량"])
                    col1.metric("평균 생중량", f"{fmt_stat(s['mean'])}g",
                                delta=f"CV {fmt_stat(s['cv'])}%" if s['cv'] else None)
                    # 최대·최소 개체 (개체번호 숫자 그대로 표시)
                    s_num = pd.to_numeric(growth_df["생중량"], errors="coerce")
                    if s_num.notna().sum() >= 1:
                        idx_max = s_num.idxmax()
                        idx_min = s_num.idxmin()
                        label_max = growth_df.loc[idx_max, "개체번호"] if "개체번호" in growth_df.columns else idx_max
                        label_min = growth_df.loc[idx_min, "개체번호"] if "개체번호" in growth_df.columns else idx_min
                        col2.metric("최대 생중량 개체",
                                    f"{fmt_stat(s['max'])}g",
                                    delta=f"개체 {label_max}")
                        col3.metric("최소 생중량 개체",
                                    f"{fmt_stat(s['min'])}g",
                                    delta=f"개체 {label_min}")
                    else:
                        col2.metric("최대 생중량 개체", "데이터 부족")
                        col3.metric("최소 생중량 개체", "데이터 부족")
                else:
                    col1.metric("평균 생중량", "생중량 열 없음")
                    col2.metric("최대 생중량 개체", "-")
                    col3.metric("최소 생중량 개체", "-")

            st.header("개체별 생육 비교 (막대 그래프)")
            # 평균 생중량 카드 그래프
            if "생중량" in growth_df.columns:
                fig_bw = bar_avg(growth_df, "생중량", "평균 생중량 (⭐ 핵심 지표)",
                                 "생중량 (g)")
                if fig_bw:
                    fig_layout(fig_bw)
                    st.plotly_chart(fig_bw, width="stretch")
                else:
                    st.info("☝️ 생중량 데이터가 부족해 그래프를 표시할 수 없습니다.")
            else:
                st.info("☝️ 생중량 열이 없어 평균 생중량 그래프를 건너뜁니다.")

            # 개체별 생중량 막대
            if "생중량" in growth_df.columns and not growth_df.empty:
                fig_ind = bar_individual(growth_df, "생중량", "개체별 생중량 비교 (상위 3·하위 3 강조)",
                                         "생중량 (g)")
                if fig_ind:
                    fig_layout(fig_ind)
                    st.plotly_chart(fig_ind, width="stretch")
            else:
                st.info("☝️ 생중량 데이터가 없어 개체별 비교를 표시할 수 없습니다.")

            # 평균 잎 수
            if "잎 수" in growth_df.columns:
                fig_lf = bar_avg(growth_df, "잎 수", "평균 잎 수",
                                 "잎 수 (장)")
                if fig_lf:
                    fig_layout(fig_lf)
                    st.plotly_chart(fig_lf, width="stretch")
            else:
                st.info("☝️ 잎 수 열이 없어 평균 잎 수 그래프를 건너뜁니다.")

            # 개체별 잎 수 막대
            if "잎 수" in growth_df.columns and not growth_df.empty:
                fig_ind_lf = bar_individual(growth_df, "잎 수", "개체별 잎 수 비교",
                                            "잎 수 (장)")
                if fig_ind_lf:
                    fig_layout(fig_ind_lf)
                    st.plotly_chart(fig_ind_lf, width="stretch")

            # 평균 지상부 길이
            if "지상부 길이" in growth_df.columns:
                fig_sl = bar_avg(growth_df, "지상부 길이", "평균 지상부 길이",
                                 "길이 (mm)")
                if fig_sl:
                    fig_layout(fig_sl)
                    st.plotly_chart(fig_sl, width="stretch")
            else:
                st.info("☝️ 지상부 길이 열이 없어 그래프를 건너뜁니다.")

            # 평균 지하부 길이
            if "지하부 길이" in growth_df.columns:
                fig_rl = bar_avg(growth_df, "지하부 길이", "평균 지하부 길이",
                                 "길이 (mm)")
                if fig_rl:
                    fig_layout(fig_rl)
                    st.plotly_chart(fig_rl, width="stretch")
            else:
                st.info("☝️ 지하부 길이 열이 없어 그래프를 건너뜁니다.")

            st.header("생중량 분포 (박스플롯)")
            if "생중량" in growth_df.columns and not growth_df.empty:
                s_num = pd.to_numeric(growth_df["생중량"], errors="coerce").dropna()
                if len(s_num) >= 2:
                    fig_box = go.Figure()
                    fig_box.add_box(y=s_num, name="생중량",
                                    fillcolor="#3498db",
                                    boxpoints="outliers",
                                    marker=dict(color="#e74c3c", size=6))
                    fig_box.update_layout(
                        title="개체별 생중량 분포",
                        yaxis_title="생중량 (g)",
                        showlegend=False,
                    )
                    fig_layout(fig_box)
                    st.plotly_chart(fig_box, width="stretch")
                else:
                    st.info("☝️ 생중량 값이 2개 이상 필요해 분포를 표시할 수 없습니다.")
            else:
                st.info("☝️ 생중량 열이 없어 박스플롯을 건너뜁니다.")

            with st.expander("📋 생육 데이터 원본 테이블 + CSV 다운로드"):
                if growth_df is not None and not growth_df.empty:
                    ind_col = "개체번호" if "개체번호" in growth_df.columns else None
                    display_cols = [c for c in ["개체번호", "잎 수", "지상부 길이", "지하부 길이",
                                                 "생중량", "지상부 질량", "지하부 질량"]
                                    if c in growth_df.columns]
                    if len(display_cols) <= 1:
                        st.info("☝️ 표시할 생육 열이 충분하지 않습니다.")
                    else:
                        st.dataframe(growth_df[display_cols], width="stretch")
                    download_csv(growth_df, "생육결과데이터")
                else:
                    st.info("☝️ 생육 CSV를 업로드하면 여기에 원본 테이블이 표시됩니다.")
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    with tabs[3]:
        try:
            st.header("상관관계는 두 변수가 함께 변하는 정도를 나타내는 값입니다.")
            st.caption("-1에 가까울수록 함께 감소(역상관), +1에 가까울수록 함께 증가(정상관), 0에 가까우면 선형 관계가 거의 없음")

            if growth_df is None or growth_df.empty:
                st.info("☝️ 생육 CSV를 업로드하면 상관관계 분석이 표시됩니다.")
            else:
                # ── 지상부 길이 vs 지하부 길이 (핵심 가설) ──
                if "지상부 길이" in growth_df.columns and "지하부 길이" in growth_df.columns:
                    valid = growth_df[["지상부 길이", "지하부 길이"]].dropna()
                    valid["지상부 길이"] = pd.to_numeric(valid["지상부 길이"], errors="coerce")
                    valid["지하부 길이"] = pd.to_numeric(valid["지하부 길이"], errors="coerce")
                    valid = valid.dropna()
                    if len(valid) >= 2:
                        fig_slrl = go.Figure()
                        fig_slrl.add_scatter(x=valid["지상부 길이"], y=valid["지하부 길이"],
                                             mode="markers", name="개체",
                                             marker=dict(color="#9bb7e8", size=8,
                                                         line=dict(color="#2c3e50", width=0.5)))
                        # 추세선 직접 계산
                        x = valid["지상부 길이"].values
                        y = valid["지하부 길이"].values
                        slope, intercept = np.polyfit(x, y, 1)
                        xline = np.linspace(x.min(), x.max(), 100)
                        fig_slrl.add_scatter(x=xline, y=slope * xline + intercept,
                                             mode="lines", name="추세선",
                                             line=dict(color="#e74c3c", width=2, dash="dash"))
                        fig_slrl.update_layout(
                            title="지상부 길이 vs 지하부 길이",
                            xaxis_title="지상부 길이 (mm)",
                            yaxis_title="지하부 길이 (mm)",
                        )
                        fig_layout(fig_slrl)
                        st.plotly_chart(fig_slrl, width="stretch")
                        r = np.corrcoef(x, y)[0, 1]
                        st.metric("지상부 ↔ 지하부 피어슨 상관계수",
                                  f"{r:.4f}",
                                  delta="0에 가까움 → 뚜렷한 선형 상관관계 거의 없음 (가설 지지)" if abs(r) < 0.2 else None)
                        st.caption("※ 지상부 길이와 지하부 길이 사이의 피어슨 상관계수입니다. 두 지표가 함께 증가/감소하는 선형 경향을 나타냅니다.")
                    else:
                        st.info("☝️ 지상부·지하부 길이 유효 데이터가 2개 이상 필요합니다.")
                else:
                    st.info("☝️ 지상부 길이 또는 지하부 길이 열이 없어 상관관계 그래프를 표시할 수 없습니다.")

                st.divider()

                # ── 잎 수 vs 생중량 ──
                if "잎 수" in growth_df.columns and "생중량" in growth_df.columns:
                    valid2 = growth_df[["잎 수", "생중량"]].dropna()
                    valid2["잎 수"] = pd.to_numeric(valid2["잎 수"], errors="coerce")
                    valid2["생중량"] = pd.to_numeric(valid2["생중량"], errors="coerce")
                    valid2 = valid2.dropna()
                    if len(valid2) >= 2:
                        fig_lf = go.Figure()
                        fig_lf.add_scatter(x=valid2["잎 수"], y=valid2["생중량"],
                                           mode="markers", name="개체",
                                           marker=dict(color="#27ae60", size=8))
                        x2 = valid2["잎 수"].values
                        y2 = valid2["생중량"].values
                        slope2, intercept2 = np.polyfit(x2, y2, 1)
                        xline2 = np.linspace(x2.min(), x2.max(), 100)
                        fig_lf.add_scatter(x=xline2, y=slope2 * xline2 + intercept2,
                                           mode="lines", name="추세선",
                                           line=dict(color="#e74c3c", width=2, dash="dash"))
                        fig_lf.update_layout(
                            title="잎 수 vs 생중량",
                            xaxis_title="잎 수 (장)",
                            yaxis_title="생중량 (g)",
                        )
                        fig_layout(fig_lf)
                        st.plotly_chart(fig_lf, width="stretch")
                        r2 = np.corrcoef(x2, y2)[0, 1]
                        st.metric("잎 수 ↔ 생중량 피어슨 상관계수", f"{r2:.4f}")
                    else:
                        st.info("☝️ 잎 수·생중량 유효 데이터가 2개 이상 필요합니다.")
                else:
                    st.info("☝️ 잎 수 또는 생중량 열이 없어 상관관계 그래프를 표시할 수 없습니다.")

                st.divider()

                # ── 지상부 길이 vs 생중량 ──
                if "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                    valid3 = growth_df[["지상부 길이", "생중량"]].dropna()
                    valid3["지상부 길이"] = pd.to_numeric(valid3["지상부 길이"], errors="coerce")
                    valid3["생중량"] = pd.to_numeric(valid3["생중량"], errors="coerce")
                    valid3 = valid3.dropna()
                    if len(valid3) >= 2:
                        fig_sl = go.Figure()
                        fig_sl.add_scatter(x=valid3["지상부 길이"], y=valid3["생중량"],
                                           mode="markers", name="개체",
                                           marker=dict(color="#8e44ad", size=8))
                        x3 = valid3["지상부 길이"].values
                        y3 = valid3["생중량"].values
                        slope3, intercept3 = np.polyfit(x3, y3, 1)
                        xline3 = np.linspace(x3.min(), x3.max(), 100)
                        fig_sl.add_scatter(x=xline3, y=slope3 * xline3 + intercept3,
                                           mode="lines", name="추세선",
                                           line=dict(color="#e74c3c", width=2, dash="dash"))
                        fig_sl.update_layout(
                            title="지상부 길이 vs 생중량",
                            xaxis_title="지상부 길이 (mm)",
                            yaxis_title="생중량 (g)",
                        )
                        fig_layout(fig_sl)
                        st.plotly_chart(fig_sl, width="stretch")
                        r3 = np.corrcoef(x3, y3)[0, 1]
                        st.metric("지상부 길이 ↔ 생중량 피어슨 상관계수", f"{r3:.4f}")
                    else:
                        st.info("☝️ 지상부 길이·생중량 유효 데이터가 2개 이상 필요합니다.")
                else:
                    st.info("☝️ 지상부 길이 또는 생중량 열이 없어 상관관계 그래프를 표시할 수 없습니다.")
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    with tabs[4]:
        try:
            st.header("생중량을 기준으로 상위 25%(Top)와 하위 25%(Bottom) 개체를 비교합니다.")
            st.caption("변동계수(CV)가 큰 항목일수록 두 그룹 간 차이가 두드러집니다.")

            if growth_df is None or growth_df.empty:
                st.info("☝️ 생육 CSV를 업로드하면 그룹 비교가 표시됩니다.")
            elif "생중량" not in growth_df.columns:
                st.info("☝️ 생중량 열이 없어 그룹 비교를 표시할 수 없습니다.")
            else:
                s_num = pd.to_numeric(growth_df["생중량"], errors="coerce").dropna()
                if len(s_num) < 2:
                    st.info("☝️ 생중량 유효 데이터가 부족합니다.")
                else:
                    q1 = s_num.quantile(0.25)
                    q3 = s_num.quantile(0.75)
                    top_idx = s_num[s_num >= q3].index
                    bot_idx = s_num[s_num <= q1].index
                    top_df = growth_df.loc[top_idx].copy() if len(top_idx) else pd.DataFrame()
                    bot_df = growth_df.loc[bot_idx].copy() if len(bot_idx) else pd.DataFrame()

                    if top_df.empty or bot_df.empty:
                        st.info("☝️ 상위 또는 하위 그룹이 비어 있어 비교할 수 없습니다.")
                    else:
                        metric_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                                       if c in growth_df.columns]
                        rows_g = []
                        for c in metric_cols:
                            tm = safe_stats(pd.to_numeric(top_df[c], errors="coerce"))
                            bm = safe_stats(pd.to_numeric(bot_df[c], errors="coerce"))
                            rows_g.append([
                                c,
                                fmt_stat(tm["mean"]),
                                fmt_stat(bm["mean"]),
                                fmt_stat(tm["mean"] - bm["mean"]) if tm["mean"] is not None and bm["mean"] is not None else "-",
                                f"{tm['n']}/{bm['n']}",
                            ])
                        st.dataframe(
                            pd.DataFrame(rows_g, columns=["지표", "Top 평균", "Bottom 평균", "차이", "개체 수(n)"]),
                            width="stretch", hide_index=True,
                        )
                        st.caption("※ Top 그룹: 생중량 상위 25% 개체 / Bottom 그룹: 생중량 하위 25% 개체")

                        # 그룹 막대 그래프 (Top vs Bottom)
                        chart_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                                      if c in growth_df.columns]
                        if chart_cols:
                            fig_gb = go.Figure()
                            x_pos = np.arange(len(chart_cols))
                            t_means = [safe_stats(pd.to_numeric(top_df[c], errors="coerce"))["mean"] for c in chart_cols]
                            b_means = [safe_stats(pd.to_numeric(bot_df[c], errors="coerce"))["mean"] for c in chart_cols]
                            t_means = [m if m is not None else 0 for m in t_means]
                            b_means = [m if m is not None else 0 for m in b_means]
                            fig_gb.add_bar(x=chart_cols, y=t_means, name="Top 25%",
                                           marker_color="#27ae60")
                            fig_gb.add_bar(x=chart_cols, y=b_means, name="Bottom 25%",
                                           marker_color="#e74c3c")
                            fig_gb.update_layout(
                                title="Top 25% vs Bottom 25% 평균 비교",
                                xaxis_title="생육 지표",
                                yaxis_title="평균값",
                                barmode="group",
                            )
                            fig_layout(fig_gb)
                            st.plotly_chart(fig_gb, width="stretch")

                        # 산점도 Top/Bottom
                        if {"잎 수", "생중량"} <= set(growth_df.columns):
                            fig_sg = go.Figure()
                            fig_sg.add_scatter(
                                x=pd.to_numeric(top_df["잎 수"], errors="coerce"),
                                y=pd.to_numeric(top_df["생중량"], errors="coerce"),
                                mode="markers", name="Top 25%",
                                marker=dict(color="#27ae60", size=9))
                            fig_sg.add_scatter(
                                x=pd.to_numeric(bot_df["잎 수"], errors="coerce"),
                                y=pd.to_numeric(bot_df["생중량"], errors="coerce"),
                                mode="markers", name="Bottom 25%",
                                marker=dict(color="#e74c3c", size=9))
                            fig_sg.update_layout(
                                title="잎 수 vs 생중량 — Top vs Bottom",
                                xaxis_title="잎 수 (장)",
                                yaxis_title="생중량 (g)",
                            )
                            fig_layout(fig_sg)
                            st.plotly_chart(fig_sg, width="stretch")
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    with tabs[5]:
        try:
            st.header("잎 수로 생중량을 예측하는 단순 선형회귀 모델입니다.")
            st.markdown("""
**선형회귀**: 두 변수 사이의 직선 관계(y = ax + b)를 찾는 방법입니다.  
- **R² (결정계수)**: 모델이 데이터 변동을 얼마나 설명하는지 나타내는 값 (0~1, 높을수록 좋음)  
- **MAE (평균 절대 오차)**: 예측값과 실제값의 평균 차이 (작을수록 정확)  
- **5겹 교차검증**: 데이터를 5등분해 번갈아 검증하여 표본 바깥에서도 잘 작동하는지 확인
""")

            if growth_df is None or growth_df.empty:
                st.info("☝️ 생육 CSV를 업로드하면 예측 모델이 표시됩니다.")
            elif "잎 수" not in growth_df.columns or "생중량" not in growth_df.columns:
                st.info("☝️ 잎 수와 생중량 열이 모두 필요해 예측 모델을 표시할 수 없습니다.")
            else:
                valid = growth_df[["잎 수", "생중량"]].dropna().copy()
                valid["잎 수"] = pd.to_numeric(valid["잎 수"], errors="coerce")
                valid["생중량"] = pd.to_numeric(valid["생중량"], errors="coerce")
                valid = valid.dropna()
                if len(valid) < 5:
                    st.info("☝️ 잎 수·생중량 유효 데이터가 5개 이상 필요합니다.")
                else:
                    X = valid[["잎 수"]].values
                    y = valid["생중량"].values
                    model = LinearRegression()
                    model.fit(X, y)
                    r2_train = model.score(X, y)
                    pred_train = model.predict(X)
                    mae_train = mean_absolute_error(y, pred_train)
                    slope = model.coef_[0]
                    intercept = model.intercept_

                    # 5겹 교차검증
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
                    mae_scorer = make_scorer(mean_absolute_error)
                    mae_scores = cross_val_score(model, X, y, cv=kf, scoring=mae_scorer)

                    col_r2, col_mae, col_cv = st.columns(3)
                    col_r2.metric("훈련 R²", f"{r2_train:.4f}")
                    col_mae.metric("훈련 MAE", f"{mae_train:.4f}g")
                    col_cv.metric("5겹 CV R² (평균)", f"{r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

                    st.markdown("**회귀식:**")
                    st.latex(f"\\text{{생중량}}(g) = {slope:.4f} \\times \\text{{잎 수}} + ({intercept:.4f})")
                    st.markdown(f"- **기울기**: 잎 수가 1장 증가할 때 생중량이 약 **{slope:.4f}g** 늘어나는 연관성")
                    st.markdown(f"- **절편**: {intercept:.4f}")

                    st.dataframe(
                        pd.DataFrame({
                            "지표": ["훈련 R²", "훈련 MAE (g)", "CV R² 평균 ± 표준편차",
                                     "CV MAE 평균 (g)", "CV MAE 표준편차 (g)"],
                            "값": [f"{r2_train:.4f}", f"{mae_train:.4f}",
                                   f"{r2_scores.mean():.4f} ± {r2_scores.std():.4f}",
                                   f"{mae_scores.mean():.4f}", f"{mae_scores.std():.4f}"],
                        }),
                        width="stretch", hide_index=True,
                    )

                    # 대표 개체 10개 예측값 비교
                    n_show = min(10, len(valid))
                    sample_idx = np.random.RandomState(42).choice(len(valid), size=n_show, replace=False)
                    sample = valid.iloc[sample_idx].copy()
                    pred_vals = model.predict(sample[["잎 수"]].values)
                    st.markdown("**대표 개체 10개 — 실제값과 예측값 비교**")
                    comp_df = pd.DataFrame({
                        "개체번호": sample["개체번호"].tolist() if "개체번호" in sample.columns else [f"개체{i+1}" for i in range(n_show)],
                        "잎 수 (장)": [f"{v:.0f}" for v in sample["잎 수"]],
                        "실제 생중량 (g)": [f"{v:.2f}" for v in sample["생중량"]],
                        "예측 생중량 (g)": [f"{v:.2f}" for v in pred_vals],
                    })
                    st.dataframe(comp_df, width="stretch", hide_index=True)
                    st.caption("※ 이 모델은 이 학교의 이 스마트팜에서 이 시기에 재배된 개체 데이터로 만들어졌으므로, 학습 데이터 범위를 벗어난 예측이나 다른 학교·환경에 적용은 적절하지 않습니다. 회귀계수를 인과관계로 해석해서는 안 됩니다.")
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ── 앱 진입점 ───────────────────────────────────────────────────
if __name__ == "__main__":
    env_file = st.sidebar.file_uploader("📁 환경 데이터 CSV", type="csv")
    growth_file = st.sidebar.file_uploader("📁 생육 결과 CSV", type="csv")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏫 학교명")
    st.sidebar.caption("(파일명에서 자동 추출)")
    st.sidebar.markdown("### 📏 생육 측정 항목 선택")
    st.sidebar.caption("모든 항목 자동 표시 (선택 제거 없음)")

    env_df = None
    growth_df = None

    if env_file is not None:
        with st.spinner("환경 데이터를 읽는 중..."):
            try:
                env_df = load_and_std(env_file)
                st.session_state.env_file_name = env_file.name
                if env_df.empty:
                    st.warning("⚠️ 환경 CSV를 읽지 못했습니다. 인코딩을 확인하세요.")
            except Exception as e:
                st.warning(f"⚠️ 환경 데이터 읽기 오류: {e}")

    if growth_file is not None:
        with st.spinner("생육 데이터를 읽는 중..."):
            try:
                growth_df = load_and_std(growth_file)
                st.session_state.growth_file_name = growth_file.name
                if growth_df.empty:
                    st.warning("⚠️ 생육 CSV를 읽지 못했습니다. 인코딩을 확인하세요.")
            except Exception as e:
                st.warning(f"⚠️ 생육 데이터 읽기 오류: {e}")

    if env_file is not None or growth_file is not None:
        raw = os.path.splitext(st.session_state.growth_file_name if growth_file is not None else
                                st.session_state.env_file_name)[0]
        school_name = raw.split("_")[0].strip() or "업로드된 학교"
        st.sidebar.success(f"✅ {school_name}")
    else:
        st.sidebar.info("CSV 파일을 업로드하면 대시보드가 시작됩니다.")

    main()
