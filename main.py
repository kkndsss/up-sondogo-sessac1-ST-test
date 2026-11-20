import io, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

# ── CSV 읽기 (utf-8-sig → cp949) + 열 이름 표준화 ─────────────
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


def safe_stats(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n < 2:
        return dict(n=0, mean=None, std=None, min=None, max=None, cv=None)
    mean = s.mean()
    std = s.std()
    return dict(
        n=n, mean=mean, std=std,
        min=s.min(), max=s.max(),
        cv=(std / mean * 100) if mean != 0 else None
    )


def fmt_stat(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if abs(v) >= 100:
        return f"{v:,.1f}"
    if abs(v) >= 10:
        return f"{v:,.2f}"
    return f"{v:,.3f}"


def download_csv(df, label):
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label, csv_data, file_name=f"{label}.csv", mime="text/csv")


def fig_layout(fig):
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=40, b=40),
    )


def bar_avg(df, col, title, ylabel):
    if col not in df.columns:
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
        showlegend=False
    )
    return fig


def bar_individual(df, col, title, ylabel):
    if col not in df.columns:
        return None
    tmp = df[["개체번호", col]].copy() if "개체번호" in df.columns else df[[col]].copy()
    tmp.columns = ["_id", "_val"] if "개체번호" in df.columns else [col]
    tmp["_val"] = pd.to_numeric(
        tmp["_val"] if "_val" in tmp.columns else tmp[col],
        errors="coerce"
    )
    tmp = (
        tmp.dropna(subset=["_val"])
        .sort_values("_val", ascending=False)
        .reset_index(drop=True)
    )
    if tmp.empty:
        return None
    if "개체번호" in df.columns:
        labels = df.loc[tmp.index, "개체번호"].astype(str).tolist()
    else:
        labels = [str(i) for i in range(len(tmp))]
    vals = tmp["_val"].tolist()
    n = len(vals)
    colors = []
    for i in range(n):
        if i < 3:
            colors.append("#d64545")
        elif i >= n - 3:
            colors.append("#4578d6")
        else:
            colors.append("#9bb7e8")
    fig = go.Figure()
    fig.add_bar(x=labels, y=vals, marker_color=colors, name="개체별 값")
    m = np.mean(vals)
    fig.add_hline(
        y=m,
        line_dash="dot",
        line_color="#ff7f0e",
        annotation_text=f"평균 {m:.2f}",
        annotation_position="top left"
    )
    fig.update_layout(
        title=title,
        xaxis_title="개체",
        yaxis_title=ylabel,
        xaxis=dict(tickangle=45),
        showlegend=False
    )
    return fig


def scatter_with_trend(df, x_col, y_col, x_label, y_label, title):
    if x_col not in df.columns or y_col not in df.columns:
        return None
    v = df[[x_col, y_col]].dropna().copy()
    v[x_col] = pd.to_numeric(v[x_col], errors="coerce")
    v[y_col] = pd.to_numeric(v[y_col], errors="coerce")
    v = v.dropna()
    if len(v) < 2:
        return None
    fig = go.Figure()
    fig.add_scatter(
        x=v[x_col],
        y=v[y_col],
        mode="markers",
        name="개체",
        marker=dict(
            color="#9bb7e8",
            size=8,
            line=dict(color="#2c3e50", width=0.5)
        )
    )
    x_vals = v[x_col].values
    y_vals = v[y_col].values
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    xline = np.linspace(x_vals.min(), x_vals.max(), 100)
    fig.add_scatter(
        x=xline,
        y=slope * xline + intercept,
        mode="lines",
        name="추세선",
        line=dict(color="#e74c3c", width=2, dash="dash")
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label
    )
    return fig, v


def main():
    st.set_page_config(page_title="송도 새싹 1팀", layout="wide")
    st.title("송도 새싹 1팀")

    school_name = ""
    if "growth_file_name" in st.session_state:
        raw = os.path.splitext(st.session_state.growth_file_name)[0]
        school_name = raw.split("_")[0].strip() or "업로드된 학교"

    tabs = st.tabs([
        "📖 실험 개요",
        "🌡️ 환경 데이터 분석",
        "📊 개체별 생육 차이",
        "🔗 가설 검증: 지상부 vs 지하부 길이",
    ])

    # ══════════════════════════════════════════════════════════
    # Tab 1 — 실험 개요
    # ══════════════════════════════════════════════════════════
    with tabs[0]:
        try:
            st.header("연구 배경 및 목적")
            st.markdown(
                """
**극지식물**은 극한의 추위·건조·자외선에서 살아남는 식물로,
기후변화 연구 지표이자 극한 환경 농업의 가능성을 열어줍니다.

스마트팜의 **온도·습도·EC·pH**는 식물 생육의 핵심 변수입니다.
온도는 효소 활성과 생장 속도를, 습도는 수분 균형을,
EC는 양액 양분 농도를, pH는 양분 흡수와 뿌리 건강을 좌우합니다.

식물의 줄기와 뿌리는 같은 개체 안에서 자라지만,
**서로 다른 환경 조건과 생리적 요인의 영향을 받아 성장**할 수 있습니다.
따라서 이 연구에서는 실제 극지식물 데이터를 이용해
**지상부 길이(줄기)와 지하부 길이(뿌리) 사이에
어떤 상관관계가 나타나는지** 분석합니다.
"""
            )

            st.header("연구 가설")
            st.info(
                """
**가설: 지상부 길이와 지하부 길이 사이에는
뚜렷한 상관관계가 없을 것이다.**

식물의 줄기와 뿌리는 서로 다른 환경 조건과 생리적 요인의
영향을 받아 성장할 수 있으므로,
지상부 길이만으로는 지하부 길이를 예측하기 어렵고
그 반대도 마찬가지일 것이라는 가설입니다.
"""
            )

            st.header("데이터 요약")
            rows = []

            if env_df is not None and not env_df.empty and "time" in env_df.columns:
                t = env_df["time"].dropna()
                if not t.empty:
                    time_range_str = (
                        f"{t.min().strftime('%Y-%m-%d %H:%M')}"
                        f" ~ {t.max().strftime('%Y-%m-%d %H:%M')}"
                    )
                else:
                    time_range_str = "-"
                env_cols_str = ", ".join(
                    [c for c in env_df.columns if c != "time"]
                )
                rows.append([
                    "환경 데이터",
                    f"{len(env_df):,}개 행",
                    time_range_str,
                    env_cols_str,
                ])
            else:
                rows.append(["환경 데이터", "업로드되지 않음", "-", "-"])

            if growth_df is not None and not growth_df.empty:
                n_g = len(growth_df)
                ind_col = "개체번호"
                if ind_col in growth_df.columns:
                    vals = growth_df[ind_col].dropna()
                    id_range = (
                        f"{vals.min()} ~ {vals.max()}"
                        if not vals.empty else "-"
                    )
                else:
                    id_range = "-"
                growth_cols_str = ", ".join(
                    [c for c in growth_df.columns if c != ind_col]
                )
                rows.append([
                    "생육 결과 데이터",
                    f"{n_g}개체",
                    "수확 시점 (개체별 측정일 없음)",
                    growth_cols_str,
                ])
            else:
                rows.append(["생육 결과 데이터", "업로드되지 않음", "-", "-"])

            st.dataframe(
                pd.DataFrame(
                    rows,
                    columns=["데이터", "규모", "측정 기간", "측정 항목"]
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "※ 생육 결과 데이터는 개체별 측정일이 없어 "
                "시계열 분석이 어렵습니다."
            )

            st.header("주요 지표 카드")
            c1, c2, c3, c4 = st.columns(4)

            if growth_df is not None and not growth_df.empty:
                c1.metric("총 개체수", f"{len(growth_df)}개체")
            else:
                c1.metric("총 개체수", "데이터 없음")

            if env_df is not None and not env_df.empty and "temperature" in env_df.columns:
                st_t = safe_stats(env_df["temperature"])
                cv_str = (
                    f"CV {fmt_stat(st_t['cv'])}%"
                    if st_t['cv'] is not None else None
                )
                c2.metric("평균 온도", f"{fmt_stat(st_t['mean'])}℃", delta=cv_str)
            else:
                c2.metric("평균 온도", "데이터 없음")

            if env_df is not None and not env_df.empty and "humidity" in env_df.columns:
                st_h = safe_stats(env_df["humidity"])
                cv_str = (
                    f"CV {fmt_stat(st_h['cv'])}%"
                    if st_h['cv'] is not None else None
                )
                c3.metric("평균 습도", f"{fmt_stat(st_h['mean'])}%", delta=cv_str)
            else:
                c3.metric("평균 습도", "데이터 없음")

            if growth_df is not None and not growth_df.empty and "생중량" in growth_df.columns:
                st_g = safe_stats(growth_df["생중량"])
                cv_str = (
                    f"CV {fmt_stat(st_g['cv'])}%"
                    if st_g['cv'] is not None else None
                )
                c4.metric("평균 생중량", f"{fmt_stat(st_g['mean'])}g", delta=cv_str)
            else:
                c4.metric("평균 생중량", "데이터 없음")

            st.caption(
                "※ CV(변동계수): 표준편차 ÷ 평균 × 100. "
                "단위가 다른 항목끼리도 평균 대비 들쭉날쭉함을 비교 가능."
            )
        except Exception as e:
            st.warning(
                f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}"
            )

    # ══════════════════════════════════════════════════════════
    # Tab 2 — 환경 데이터 분석
    # ══════════════════════════════════════════════════════════
    with tabs[1]:
        try:
            st.header("환경 요약 (평균 · 표준편차 · 최소 · 최대 · CV)")
            st.caption(
                "※ 표준편차: 데이터가 평균에서 얼마나 퍼져 있는지 "
                "나타내는 값 (분산의 제곱근)."
            )
            st.caption(
                "※ CV(%): 표준편차÷평균×100, "
                "단위가 다른 항목끼리도 평균 대비 변동성을 비교 가능."
            )

            if env_df is None or env_df.empty:
                st.info("☝️ 환경 CSV를 업로드하면 환경 요약이 표시됩니다.")
            else:
                env_cols = [
                    c for c in ["temperature", "humidity", "ph", "ec"]
                    if c in env_df.columns
                ]
                if not env_cols:
                    st.info(
                        "☝️ 온도·습도·pH·EC 열이 없어 "
                        "환경 요약을 표시할 수 없습니다."
                    )
                else:
                    srows = []
                    for c in env_cols:
                        s = safe_stats(env_df[c])
                        srows.append([
                            c,
                            fmt_stat(s["mean"]),
                            fmt_stat(s["std"]),
                            fmt_stat(s["min"]),
                            fmt_stat(s["max"]),
                            fmt_stat(s["cv"]),
                        ])
                    st.dataframe(
                        pd.DataFrame(
                            srows,
                            columns=["항목", "평균", "표준편차", "최소", "최대", "CV(%)"]
                        ),
                        width="stretch",
                        hide_index=True,
                    )

            st.header("날짜별 평균 시계열 (일별 평균 꺾은선)")

            if env_df is None or env_df.empty or "time" not in env_df.columns:
                st.info("☝️ 환경 CSV의 time 열이 필요해 시계열을 표시할 수 없습니다.")
            else:
                ed = env_df.dropna(subset=["time"]).copy()
                if ed.empty:
                    st.info("☝️ time 값이 없어 시계열을 표시할 수 없습니다.")
                else:
                    ed["day"] = ed["time"].dt.floor("D")
                    daily = ed.groupby("day")

                    if "temperature" in ed.columns:
                        dt = daily["temperature"].agg(["mean", "min", "max"]).dropna()
                        if len(dt) >= 1:
                            fig_t = go.Figure()
                            fig_t.add_scatter(
                                x=dt.index, y=dt["min"], mode="lines",
                                name="최저",
                                line=dict(color="#e74c3c", width=1, dash="dot")
                            )
                            fig_t.add_scatter(
                                x=dt.index, y=dt["max"], mode="lines",
                                name="최고(범위)",
                                line=dict(color="#e74c3c", width=0),
                                fill="tonexty",
                                fillcolor="rgba(231,76,60,0.12)"
                            )
                            fig_t.add_scatter(
                                x=dt.index, y=dt["mean"], mode="lines",
                                name="평균",
                                line=dict(color="#e74c3c", width=2)
                            )
                            fig_t.update_layout(
                                title="날짜별 평균 온도 변화",
                                xaxis_title="날짜",
                                yaxis_title="온도 (℃)",
                            )
                            fig_layout(fig_t)
                            st.plotly_chart(fig_t, width="stretch")

                    if "humidity" in ed.columns:
                        dh = daily["humidity"].agg(["mean", "min", "max"]).dropna()
                        if len(dh) >= 1:
                            fig_h = go.Figure()
                            fig_h.add_scatter(
                                x=dh.index, y=dh["min"], mode="lines",
                                name="최저",
                                line=dict(color="#3498db", width=1, dash="dot")
                            )
                            fig_h.add_scatter(
                                x=dh.index, y=dh["max"], mode="lines",
                                name="최고(범위)",
                                line=dict(color="#3498db", width=0),
                                fill="tonexty",
                                fillcolor="rgba(52,152,219,0.12)"
                            )
                            fig_h.add_scatter(
                                x=dh.index, y=dh["mean"], mode="lines",
                                name="평균",
                                line=dict(color="#3498db", width=2)
                            )
                            fig_h.update_layout(
                                title="날짜별 평균 습도 변화",
                                xaxis_title="날짜",
                                yaxis_title="습도 (%)",
                            )
                            fig_layout(fig_h)
                            st.plotly_chart(fig_h, width="stretch")

                    if "ph" in ed.columns:
                        dp = daily["ph"].agg(["mean", "min", "max"]).dropna()
                        if len(dp) >= 1:
                            fig_p = go.Figure()
                            fig_p.add_scatter(
                                x=dp.index, y=dp["min"], mode="lines",
                                name="최저",
                                line=dict(color="#8e44ad", width=1, dash="dot")
                            )
                            fig_p.add_scatter(
                                x=dp.index, y=dp["max"], mode="lines",
                                name="최고(범위)",
                                line=dict(color="#8e44ad", width=0),
                                fill="tonexty",
                                fillcolor="rgba(142,68,173,0.12)"
                            )
                            fig_p.add_scatter(
                                x=dp.index, y=dp["mean"], mode="lines",
                                name="평균",
                                line=dict(color="#8e44ad", width=2)
                            )
                            fig_p.update_layout(
                                title="날짜별 평균 pH 변화",
                                xaxis_title="날짜",
                                yaxis_title="pH",
                            )
                            fig_layout(fig_p)
                            st.plotly_chart(fig_p, width="stretch")
                        else:
                            st.info("☝️ pH 데이터가 충분하지 않아 그래프를 표시할 수 없습니다.")
                    else:
                        st.info("☝️ pH 열이 없어 pH 시계열을 건너뜁니다.")

                    if "ec" in ed.columns:
                        dec = daily["ec"].agg(["mean", "min", "max"]).dropna()
                        if len(dec) >= 1:
                            fig_e = go.Figure()
                            fig_e.add_scatter(
                                x=dec.index, y=dec["min"], mode="lines",
                                name="최저",
                                line=dict(color="#27ae60", width=1, dash="dot")
                            )
                            fig_e.add_scatter(
                                x=dec.index, y=dec["max"], mode="lines",
                                name="최고(범위)",
                                line=dict(color="#27ae60", width=0),
                                fill="tonexty",
                                fillcolor="rgba(39,174,96,0.12)"
                            )
                            fig_e.add_scatter(
                                x=dec.index, y=dec["mean"], mode="lines",
                                name="평균",
                                line=dict(color="#27ae60", width=2)
                            )
                            fig_e.update_layout(
                                title="날짜별 평균 EC 변화",
                                xaxis_title="날짜",
                                yaxis_title="EC (mS/cm)",
                            )
                            fig_layout(fig_e)
                            st.plotly_chart(fig_e, width="stretch")
                        else:
                            st.info("☝️ EC 데이터가 충분하지 않아 그래프를 표시할 수 없습니다.")
                    else:
                        st.info("☝️ EC 열이 없어 EC 시계열을 건너뜁니다.")

            st.header("급변동 구간 (연속 측정값 차이가 큰 시점)")
            st.caption("※ 연속 두 시간대 측정값의 차이가 큰 구간을 분석한 결과입니다.")

            if env_df is None or env_df.empty or "time" not in env_df.columns:
                st.info("☝️ 환경 CSV가 필요해 급변동 분석을 표시할 수 없습니다.")
            else:
                surge_cols = [
                    c for c in ["temperature", "humidity", "ph", "ec"]
                    if c in env_df.columns
                ]
                if not surge_cols:
                    st.info("☝️ 급변동 분석할 환경 열이 없습니다.")
                else:
                    surge_rows = []
                    for c in surge_cols:
                        s = pd.to_numeric(env_df[c], errors="coerce").dropna()
                        if len(s) < 2:
                            continue
                        diff = s.diff().abs().dropna()
                        if diff.empty:
                            continue
                        thr = diff.quantile(0.95)
                        big = diff[diff >= thr]
                        for idx in big.index:
                            if idx > 0 and idx < len(env_df):
                                row_cur = env_df.iloc[idx]
                                row_prv = env_df.iloc[idx - 1]
                                cur_t = row_cur.get("time", None)
                                prv_t = row_prv.get("time", None)
                                cur_v = row_cur.get(c, None)
                                prv_v = row_prv.get(c, None)
                                if (
                                    pd.notna(cur_t) and pd.notna(prv_t)
                                    and pd.notna(cur_v) and pd.notna(prv_v)
                                ):
                                    pt_str = (
                                        prv_t.strftime("%Y-%m-%d %H:%M")
                                        if hasattr(prv_t, "strftime") else str(prv_t)
                                    )
                                    ct_str = (
                                        cur_t.strftime("%Y-%m-%d %H:%M")
                                        if hasattr(cur_t, "strftime") else str(cur_t)
                                    )
                                    surge_rows.append([
                                        c, pt_str, ct_str,
                                        f"{prv_v:.2f} → {cur_v:.2f}",
                                        f"{cur_v - prv_v:+.2f}",
                                        f"임계: 상위 5% (최대 변화 {diff.max():.2f})",
                                    ])
                    if surge_rows:
                        st.dataframe(
                            pd.DataFrame(
                                surge_rows,
                                columns=["지표", "이전 시각", "이후 시각", "변화값", "변화량", "참고"]
                            ),
                            width="stretch",
                            hide_index=True,
                        )
                    else:
                        st.info("☝️ 급변동 구간이 발견되지 않았습니다.")

            with st.expander("📋 환경 데이터 원본 테이블 + CSV 다운로드"):
                if env_df is not None and not env_df.empty:
                    disp_cols = [
                        c for c in ["time", "temperature", "humidity", "ph", "ec"]
                        if c in env_df.columns
                    ]
                    if len(disp_cols) <= 1:
                        st.info("☝️ 표시할 환경 열이 충분하지 않습니다.")
                    else:
                        st.dataframe(env_df[disp_cols], width="stretch")
                    download_csv(env_df, "환경데이터")
                else:
                    st.info("☝️ 환경 CSV를 업로드하면 여기에 원본 테이블이 표시됩니다.")
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    # ══════════════════════════════════════════════════════════
    # Tab 3 — 개체별 생육 차이
    # ══════════════════════════════════════════════════════════
    with tabs[2]:
        try:
            st.header("🥇 핵심 결과 카드")

            if growth_df is None or growth_df.empty:
                st.info("☝️ 생육 CSV를 업로드하면 핵심 결과가 표시됩니다.")
            else:
                col1, col2, col3 = st.columns(3)

                if "생중량" in growth_df.columns:
                    s = safe_stats(growth_df["생중량"])
                    cv_str = (
                        f"CV {fmt_stat(s['cv'])}%"
                        if s['cv'] is not None else None
                    )
                    col1.metric("평균 생중량", f"{fmt_stat(s['mean'])}g", delta=cv_str)
                    s_num = pd.to_numeric(growth_df["생중량"], errors="coerce")
                    if s_num.notna().sum() >= 1:
                        idx_max = s_num.idxmax()
                        idx_min = s_num.idxmin()
                        lmax = (
                            growth_df.loc[idx_max, "개체번호"]
                            if "개체번호" in growth_df.columns else idx_max
                        )
                        lmin = (
                            growth_df.loc[idx_min, "개체번호"]
                            if "개체번호" in growth_df.columns else idx_min
                        )
                        col2.metric("최대 생중량 개체", f"{fmt_stat(s['max'])}g", delta=f"개체 {lmax}")
                        col3.metric("최소 생중량 개체", f"{fmt_stat(s['min'])}g", delta=f"개체 {lmin}")
                    else:
                        col2.metric("최대 생중량 개체", "데이터 부족")
                        col3.metric("최소 생중량 개체", "데이터 부족")
                else:
                    col1.metric("평균 생중량", "생중량 열 없음")
                    col2.metric("최대 생중량 개체", "-")
                    col3.metric("최소 생중량 개체", "-")

            st.header("개체별 생육 비교 (막대 그래프)")
            st.caption("※ 빨간 막대: 상위 3개체 / 파란 막대: 하위 3개체 / 점선: 평균")

            if "생중량" in growth_df.columns:
                f = bar_avg(growth_df, "생중량", "평균 생중량 (⭐ 핵심 지표)", "생중량 (g)")
                if f:
                    fig_layout(f)
                    st.plotly_chart(f, width="stretch")
            else:
                st.info("☝️ 생중량 열이 없어 평균 생중량 그래프를 건너뜁니다.")

            if "생중량" in growth_df.columns and not growth_df.empty:
                f = bar_individual(
                    growth_df, "생중량",
                    "개체별 생중량 비교 (상위·하위 3개체 강조)", "생중량 (g)"
                )
                if f:
                    fig_layout(f)
                    st.plotly_chart(f, width="stretch")
            else:
                st.info("☝️ 생중량 데이터가 부족해 개체별 비교를 표시할 수 없습니다.")

            if "잎 수" in growth_df.columns:
                f = bar_avg(growth_df, "잎 수", "평균 잎 수", "잎 수 (장)")
                if f:
                    fig_layout(f)
                    st.plotly_chart(f, width="stretch")
            else:
                st.info("☝️ 잎 수 열이 없어 평균 잎 수 그래프를 건너뜁니다.")

            if "잎 수" in growth_df.columns and not growth_df.empty:
                f = bar_individual(growth_df, "잎 수", "개체별 잎 수 비교", "잎 수 (장)")
                if f:
                    fig_layout(f)
                    st.plotly_chart(f, width="stretch")

            if "지상부 길이" in growth_df.columns:
                f = bar_avg(growth_df, "지상부 길이", "평균 지상부 길이", "길이 (mm)")
                if f:
                    fig_layout(f)
                    st.plotly_chart(f, width="stretch")
            else:
                st.info("☝️ 지상부 길이 열이 없어 그래프를 건너뜁니다.")

            if "지하부 길이" in growth_df.columns:
                f = bar_avg(growth_df, "지하부 길이", "평균 지하부 길이", "길이 (mm)")
                if f:
                    fig_layout(f)
                    st.plotly_chart(f, width="stretch")
            else:
                st.info("☝️ 지하부 길이 열이 없어 그래프를 건너뜁니다.")

            st.header("생중량 분포 (박스플롯)")
            if "생중량" in growth_df.columns and not growth_df.empty:
                s_num = pd.to_numeric(growth_df["생중량"], errors="coerce").dropna()
                if len(s_num) >= 2:
                    fig_box = go.Figure()
                    fig_box.add_box(
                        y=s_num, name="생중량", fillcolor="#3498db",
                        boxpoints="outliers", marker=dict(color="#e74c3c", size=6)
                    )
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

            st.header("개체별 생육 지표 요약 (평균·표준편차·최소·최대·CV)")
            summary_rows = []
            grow_cols = [
                c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                if c in growth_df.columns
            ]
            if grow_cols:
                for c in grow_cols:
                    s = safe_stats(growth_df[c])
                    if s["n"] >= 2:
                        summary_rows.append([
                            c, fmt_stat(s["mean"]), fmt_stat(s["std"]),
                            fmt_stat(s["min"]), fmt_stat(s["max"]), fmt_stat(s["cv"]),
                        ])
                if summary_rows:
                    st.dataframe(
                        pd.DataFrame(
                            summary_rows,
                            columns=["지표", "평균", "표준편차", "최소", "최대", "CV(%)"]
                        ),
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(
                        "※ CV가 클수록 개체 간 차이가 큽니다. "
                        "생중량 CV 89.42%로 4개 지표 중 가장 큽니다."
                    )
            else:
                st.info("☝️ 표시할 생육 지표가 없습니다.")

            st.header("잘 자란 개체 vs 덜 자란 개체 (생중량 기준 Top/Bottom 25%)")
            st.caption("※ 생중량으로 상위 25%와 하위 25%를 나누어 생육 지표를 비교합니다.")

            if "생중량" in growth_df.columns and not growth_df.empty:
                s_num = pd.to_numeric(growth_df["생중량"], errors="coerce").dropna()
                if len(s_num) >= 2:
                    q1 = s_num.quantile(0.25)
                    q3 = s_num.quantile(0.75)
                    top_idx = list(s_num[s_num >= q3].index)
                    bot_idx = list(s_num[s_num <= q1].index)
                    top_df = growth_df.loc[top_idx].copy() if top_idx else pd.DataFrame()
                    bot_df = growth_df.loc[bot_idx].copy() if bot_idx else pd.DataFrame()
                    if not top_df.empty and not bot_df.empty:
                        grp_cols = [
                            c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                            if c in growth_df.columns
                        ]
                        grp_rows = []
                        for c in grp_cols:
                            tm = safe_stats(pd.to_numeric(top_df[c], errors="coerce"))
                            bm = safe_stats(pd.to_numeric(bot_df[c], errors="coerce"))
                            if tm["n"] >= 1 and bm["n"] >= 1:
                                diff = (
                                    tm["mean"] - bm["mean"]
                                    if tm["mean"] is not None and bm["mean"] is not None
                                    else None
                                )
                                grp_rows.append([
                                    c, fmt_stat(tm["mean"]), fmt_stat(bm["mean"]),
                                    fmt_stat(diff) if diff is not None else "-",
                                    f"{tm['n']}/{bm['n']}",
                                ])
                        if grp_rows:
                            st.dataframe(
                                pd.DataFrame(
                                    grp_rows,
                                    columns=["지표", "Top 25% 평균", "Bottom 25% 평균", "차이", "n"]
                                ),
                                width="stretch",
                                hide_index=True,
                            )
                            t_means = []
                            b_means = []
                            for c in grp_cols:
                                tm2 = safe_stats(pd.to_numeric(top_df[c], errors="coerce"))
                                bm2 = safe_stats(pd.to_numeric(bot_df[c], errors="coerce"))
                                t_means.append(tm2["mean"] if tm2["mean"] is not None else 0)
                                b_means.append(bm2["mean"] if bm2["mean"] is not None else 0)
                            fig_gb = go.Figure()
                            fig_gb.add_bar(
                                x=grp_cols, y=t_means, name="Top 25%", marker_color="#27ae60"
                            )
                            fig_gb.add_bar(
                                x=grp_cols, y=b_means, name="Bottom 25%", marker_color="#e74c3c"
                            )
                            fig_gb.update_layout(
                                title="Top 25% vs Bottom 25% 평균 비교",
                                xaxis_title="생육 지표",
                                yaxis_title="평균값",
                                barmode="group",
                            )
                            fig_layout(fig_gb)
                            st.plotly_chart(fig_gb, width="stretch")
                        else:
                            st.info("☝️ 비교할 지표가 충분하지 않습니다.")
                    else:
                        st.info("☝️ 상위 또는 하위 그룹이 비어 있어 비교할 수 없습니다.")
                else:
                    st.info("☝️ 생중량 유효 데이터가 부족합니다.")
            else:
                st.info("☝️ 생중량 열이 없어 그룹 비교를 건너뜁니다.")

            with st.expander("📋 생육 데이터 원본 테이블 + CSV 다운로드"):
                if growth_df is not None and not growth_df.empty:
                    disp = [
                        c for c in [
                            "개체번호", "잎 수", "지상부 길이", "지하부 길이",
                            "생중량", "지상부 질량", "지하부 질량"
                        ]
                        if c in growth_df.columns
                    ]
                    if len(disp) <= 1:
                        st.info("☝️ 표시할 생육 열이 충분하지 않습니다.")
                    else:
                        st.dataframe(growth_df[disp], width="stretch")
                    download_csv(growth_df, "생육결과데이터")
                else:
                    st.info("☝️ 생육 CSV를 업로드하면 여기에 원본 테이블이 표시됩니다.")
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    # ══════════════════════════════════════════════════════════
    # Tab 4 — 가설 검증: 지상부 vs 지하부 길이
    # ══════════════════════════════════════════════════════════
    with tabs[3]:
        try:
            st.header("연구 가설 검증: 지상부 길이 ↔ 지하부 길이 상관관계")
            st.markdown(
                """
**가설: 지상부 길이와 지하부 길이 사이에는
뚜렷한 상관관계가 없을 것이다.**

**상관계수**는 두 변수가 함께 변하는 정도를 나타내는 값으로,
-1에 가까우면 함께 감소(역상관),
+1에 가까우면 함께 증가(정상관),
0에 가까우면 선형 관계가 거의 없음을 뜻합니다.
"""
            )

            if growth_df is None or growth_df.empty:
                st.info("☝️ 생육 CSV를 업로드하면 가설 검증이 표시됩니다.")
            elif not (
                "지상부 길이" in growth_df.columns
                and "지하부 길이" in growth_df.columns
            ):
                st.info(
                    "☝️ 지상부 길이와 지하부 길이 열이 모두 필요해 "
                    "가설 검증을 표시할 수 없습니다."
                )
            else:
                res = scatter_with_trend(
                    growth_df, "지상부 길이", "지하부 길이",
                    "지상부 길이 (mm)", "지하부 길이 (mm)",
                    "지상부 길이 vs 지하부 길이",
                )
                if res is not None:
                    fig, v = res
                    fig_layout(fig)
                    st.plotly_chart(fig, width="stretch")
                    r = np.corrcoef(
                        v["지상부 길이"].values,
                        v["지하부 길이"].values,
                    )[0, 1]
                    st.metric("지상부 ↔ 지하부 피어슨 상관계수", f"{r:.4f}")
                    if abs(r) < 0.2:
                        st.success(
                            f"✅ 지상부 길이와 지하부 길이 사이의 "
                            f"상관계수는 **{r:.4f}**로 0에 가깝습니다."
                        )
                        st.success(
                            "연구 가설 **'지상부 길이와 지하부 길이 사이에는 "
                            "뚜렷한 상관관계가 없을 것이다'** 를 지지합니다."
                        )
                    else:
                        st.info(
                            f"상관계수 {r:.4f} — "
                            "0에서 멀어 가설 검증 결과를 신중히 해석해야 합니다."
                        )
                    st.caption(
                        "※ 지상부 길이와 지하부 길이 사이의 피어슨 상관계수입니다. "
                        "두 지표가 함께 증가/감소하는 선형 경향을 나타냅니다."
                    )

                st.divider()

                st.header("참고: 잎 수 ↔ 생중량 상관관계")
                if "잎 수" in growth_df.columns and "생중량" in growth_df.columns:
                    res2 = scatter_with_trend(
                        growth_df, "잎 수", "생중량",
                        "잎 수 (장)", "생중량 (g)",
                        "잎 수 vs 생중량",
                    )
                    if res2 is not None:
                        fig2, v2 = res2
                        fig_layout(fig2)
                        st.plotly_chart(fig2, width="stretch")
                        r2 = np.corrcoef(
                            v2["잎 수"].values,
                            v2["생중량"].values,
                        )[0, 1]
                        st.metric("잎 수 ↔ 생중량 피어슨 상관계수", f"{r2:.4f}")
                        st.caption(
                            "※ 잎 수와 생중량 사이에는 비교적 강한 정상관이 나타납니다. "
                            "잎 수가 많을수록 생중량이 큰 경향이 있습니다."
                        )
                else:
                    st.info("☝️ 잎 수 또는 생중량 열이 없어 참고 그래프를 표시할 수 없습니다.")

                st.divider()

                st.header("참고: 지상부 길이 ↔ 생중량 상관관계")
                if "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                    res3 = scatter_with_trend(
                        growth_df, "지상부 길이", "생중량",
                        "지상부 길이 (mm)", "생중량 (g)",
                        "지상부 길이 vs 생중량",
                    )
                    if res3 is not None:
                        fig3, v3 = res3
                        fig_layout(fig3)
                        st.plotly_chart(fig3, width="stretch")
                        r3 = np.corrcoef(
                            v3["지상부 길이"].values,
                            v3["생중량"].values,
                        )[0, 1]
                        st.metric("지상부 길이 ↔ 생중량 피어슨 상관계수", f"{r3:.4f}")
                        st.caption(
                            "※ 지상부 길이와 생중량 사이에도 정상관이 나타납니다. "
                            "그러나 잎 수만큼 강한 상관은 아닙니다."
                        )
                else:
                    st.info("☝️ 지상부 길이 또는 생중량 열이 없어 참고 그래프를 표시할 수 없습니다.")

            st.divider()
            st.header("결론")
            st.markdown(
                """
본 탐구의 분석 결과,
**지상부 길이와 지하부 길이 사이의 피어슨 상관계수는
약 +0.01**로 나타나
두 변수 사이에 뚜렷한 선형 상관관계가 거의 없음을 확인했습니다.

이는 "지상부 길이와 지하부 길이 사이에는
뚜렷한 상관관계가 없을 것이다"라는
**연구 가설을 지지**하는 결과입니다.
식물의 줄기와 뿌리는 같은 개체 안에서 자라지만
서로 다른 환경·생리적 요인의 영향을 받아 성장할 수 있음을 시사합니다.

다만 상관계수가 0에 가깝다는 결과가
두 지표의 생장이 완전히 무관하다는 뜻은 아니며,
**비선형 관계나 다른 요인이 존재할 가능성**은 남아 있습니다.
"""
            )
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ════════════════════════════════════════════════════════════════
# 앱 진입점
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    env_file = st.sidebar.file_uploader("📁 환경 데이터 CSV", type="csv")
    growth_file = st.sidebar.file_uploader("📁 생육 결과 CSV", type="csv")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏫 학교명")
    st.sidebar.caption("(파일명에서 자동 추출)")
    st.sidebar.markdown("### 📏 생육 측정 항목")
    st.sidebar.caption("모든 항목 자동 표시")

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
        raw = os.path.splitext(
            st.session_state.growth_file_name
            if growth_file is not None
            else st.session_state.env_file_name
        )[0]
        school_name = raw.split("_")[0].strip() or "업로드된 학교"
        st.sidebar.success(f"✅ {school_name}")
    else:
        st.sidebar.info("CSV 파일을 업로드하면 대시보드가 시작됩니다.")

    main()
