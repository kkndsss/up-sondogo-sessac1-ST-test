import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 한글 폰트 CSS ──────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>""", unsafe_allow_html=True)

# ── CSV 읽기 (인코딩 이중 시도) ─────────────────────────────────────────────────
@st.cache_data
def read_csv(file):
    if file is None:
        return pd.DataFrame()
    raw = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ── 열 이름 표준화 ──────────────────────────────────────────────────────────────
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
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + \
                pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df

# ── 폰트 세팅 헬퍼 ─────────────────────────────────────────────────────────────
def set_font(fig):
    fig.update_layout(font=dict(
        family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"))
    return fig

# ── 앱 설정 ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

# ── 사이드바: 파일 업로드 ──────────────────────────────────────────────────────
env_file = st.sidebar.file_uploader("환경 CSV 업로드", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV 업로드", type="csv")

selected_items = st.sidebar.multiselect(
    "생육 측정 항목 (그래프에 표시할 항목)",
    ["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량"],
    default=["잎 수", "지상부 길이", "지하부 길이"],
)

# ── 학교명 추출 ────────────────────────────────────────────────────────────────
school_name = "우리 학교"
if growth_file is not None:
    bn = os.path.splitext(os.path.basename(growth_file.name))[0]
    school_name = bn.replace("_", " ").replace("-", " ")

# ── 데이터 로딩 (캐시 + 스피너) ────────────────────────────────────────────────
with st.spinner("데이터를 불러오는 중..."):
    env_df = read_csv(env_file)
    growth_df = read_csv(growth_file)
    if not env_df.empty:
        env_df = std_cols(env_df)
    if not growth_df.empty:
        growth_df = std_cols(growth_df)

# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 1: 실험 개요
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

with tab1:
    try:
        st.header("연구 배경 및 목적")
        st.markdown(
            f"""**{school_name}**에서 재배 중인 극지식물의 **개체별 생육 차이**를 분석합니다.
            같은 환경에서 자라도 유전자 차이, 미세 환경 차이 등으로 개체마다 자라는 정도가 다를 수 있습니다.
            이 대시보드는 각 개체의 잎 수·길이·질량 데이터를 비교해 **어떤 개체가 잘 자랐는지**,
            **생육 지표끼리 어떤 관계**가 있는지 살펴봅니다."""
        )

        if growth_df is not None and not growth_df.empty:
            n_total = len(growth_df)
            gcol = "생중량"
            gw = growth_df[gcol].dropna() if gcol in growth_df.columns else pd.Series(dtype=float)
            mean_bw = gw.mean() if len(gw) >= 1 else np.nan
            std_bw = gw.std(ddof=1) if len(gw) >= 2 else np.nan
            cv_bw = (std_bw / mean_bw * 100) if (
                not np.isnan(mean_bw) and len(gw) >= 2 and mean_bw != 0) else np.nan

            st.subheader("데이터 요약")
            rows = []
            rows.append({"항목": "총 개체 수", "값": f"{n_total}개체"})
            if "time" in env_df.columns and not env_df.empty:
                t = env_df["time"].dropna()
                if len(t) >= 2:
                    rows.append({
                        "항목": "환경 측정 기간",
                        "값": f"{t.min().strftime('%Y-%m-%d')} ~ "
                              f"{t.max().strftime('%Y-%m-%d')}"
                    })
            item_cols_summary = [c for c in growth_df.columns
                                 if c not in ("개체번호",)]
            rows.append({
                "항목": "생육 측정 항목",
                "값": ", ".join(item_cols_summary) if item_cols_summary else "–"
            })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            st.subheader("주요 지표")
            m1, m2, m3, m4 = st.columns(4)

            m1.metric("총 개체 수", f"{n_total}개체")

            if "temperature" in env_df.columns and not env_df.empty:
                t_d = env_df["temperature"].dropna()
                if len(t_d) >= 1 and not t_d.empty:
                    m2.metric("평균 온도", f"{t_d.mean():.1f} °C")
                else:
                    m2.metric("평균 온도", "–")
            else:
                m2.metric("평균 온도", "데이터 없음")

            if "humidity" in env_df.columns and not env_df.empty:
                h_d = env_df["humidity"].dropna()
                if len(h_d) >= 1 and not h_d.empty:
                    m3.metric("평균 습도", f"{h_d.mean():.1f} %")
                else:
                    m3.metric("평균 습도", "–")
            else:
                m3.metric("평균 습도", "데이터 없음")

            if not np.isnan(mean_bw):
                m4.metric("평균 생중량", f"{mean_bw:.2f} g",
                          delta=f"CV {cv_bw:.1f}%" if not np.isnan(cv_bw) else None)
            else:
                m4.metric("평균 생중량", "데이터 없음")

            st.info(
                "💡 **통계 용어 간단 설명**  \n"
                "- **평균**: 모든 값을 더해 개수로 나눈 대표값  \n"
                "- **표준편차**: 값들이 평균에서 얼마나 퍼져 있는지 나타내는 정도  \n"
                "- **CV(변동계수)**: 표준편차 ÷ 평균 × 100. 클수록 개체 간 차이 큼"
            )
        else:
            st.info("📥 왼쪽 사이드바에서 생육 결과 CSV를 업로드해 주세요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 2: 환경 데이터
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    try:
        if env_df is None or env_df.empty:
            st.info("📥 왼쪽 사이드바에서 환경 CSV를 업로드해 주세요.")
        else:
            env_num_cols = [c for c in ["temperature", "humidity", "ph", "ec"]
                            if c in env_df.columns]
            if env_num_cols:
                st.subheader("🌡️ 환경 요약 (평균 · 최소 · 최대)")
                rows = []
                for col in env_num_cols:
                    s = env_df[col].dropna()
                    if len(s) >= 1:
                        rows.append({
                            "항목": col,
                            "평균": round(float(s.mean()), 2),
                            "최소": round(float(s.min()), 2),
                            "최대": round(float(s.max()), 2),
                        })
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            if "time" in env_df.columns and not env_df.empty:
                st.subheader("📅 날짜별 환경 변화 (일별 평균)")
                d_cols = [c for c in ["temperature", "humidity", "ph", "ec"]
                          if c in env_df.columns]
                if d_cols:
                    daily = env_df.copy()
                    daily["date"] = daily["time"].dt.date
                    grp = daily.groupby("date")[d_cols].mean()
                    tab_objects = st.tabs([f"🌡️ {c}" for c in d_cols])
                    for t_obj, col in zip(tab_objects, d_cols):
                        with t_obj:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=list(grp.index), y=list(grp[col]),
                                mode="lines+markers",
                                line=dict(color="#2E86AB", width=2),
                                marker=dict(size=4),
                                name=col,
                            ))
                            fig.update_xaxes(title_text="날짜")
                            fig.update_yaxes(title_text=col)
                            set_font(fig)
                            st.plotly_chart(fig, width="stretch")
                else:
                    st.info("표시할 환경 지표(온도·습도·pH·EC) 열이 없습니다.")

            with st.expander("📋 환경 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(env_df, width="stretch")
                csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school_name}_환경데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 3: 생육 결과
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    try:
        if growth_df is None or growth_df.empty:
            st.info("📥 왼쪽 사이드바에서 생육 결과 CSV를 업로드해 주세요.")
        else:
            gcol = "생중량"
            xcol = "잎 수"
            scol = "지상부 길이"

            # ── 핵심 결과 카드 ──────────────────────────────────────────
            gw = growth_df[gcol].dropna() if gcol in growth_df.columns else pd.Series(dtype=float)
            if len(gw) >= 1:
                mean_bw = gw.mean()
                max_idx = gw.idxmax()
                min_idx = gw.idxmin()
                max_val = float(gw.max())
                min_val = float(gw.min())
                max_id = growth_df.iloc[max_idx]["개체번호"]
                min_id = growth_df.iloc[min_idx]["개체번호"]

                st.subheader("🥇 핵심 결과")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("평균 생중량", f"{mean_bw:.2f} g")
                k2.metric("최대 생중량", f"{max_val:.2f} g",
                          delta=f"개체 {max_id}")
                k3.metric("최소 생중량", f"{min_val:.2f} g",
                          delta=f"개체 {min_id}")
                if len(gw) >= 2:
                    std_bw = float(gw.std(ddof=1))
                    cv_bw = std_bw / mean_bw * 100
                    k4.metric("CV (변동계수)", f"{cv_bw:.1f} %",
                              help="표준편차 ÷ 평균 × 100. 클수록 개체 간 차이 큼")
                else:
                    k4.metric("CV (변동계수)", "–")

            # ── 개체별 생육 비교 (막대 그래프) ───────────────────────────
            st.subheader("📊 개체별 생육 비교")
            bw_present = gcol in growth_df.columns
            item_cols = [c for c in selected_items if c in growth_df.columns]
            compare_cols = [c for c in ([gcol] if bw_present else []) + item_cols
                           if c in growth_df.columns]

            if compare_cols:
                bw_vals = growth_df[gcol].dropna() if gcol in growth_df.columns else None
                bw_q75 = float(bw_vals.quantile(0.75)) if (bw_vals is not None and len(bw_vals) >= 4) else None
                bw_q25 = float(bw_vals.quantile(0.25)) if (bw_vals is not None and len(bw_vals) >= 4) else None

                fig_bar = make_subplots(
                    rows=len(compare_cols), cols=1,
                    subplot_titles=list(compare_cols),
                    vertical_spacing=0.08,
                )
                for i, col in enumerate(compare_cols):
                    mask = growth_df[col].notna()
                    x_ids = list(growth_df.loc[mask, "개체번호"].values)
                    y_vals = list(growth_df.loc[mask, col].values)
                    color = []
                    for v in y_vals:
                        if bw_q75 is not None and v >= bw_q75:
                            color.append("#E84855")
                        elif bw_q25 is not None and v <= bw_q25:
                            color.append("#F0A500")
                        else:
                            color.append("#4CAF50")
                    fig_bar.add_trace(go.Bar(
                        x=x_ids, y=y_vals,
                        marker=dict(color=color),
                        name=col,
                        showlegend=(i == 0),
                    ), row=i+1, col=1)
                    fig_bar.update_yaxes(title_text=col, row=i+1, col=1)
                fig_bar.update_xaxes(title_text="개체번호")
                set_font(fig_bar)
                st.plotly_chart(fig_bar, width="stretch")
                st.caption("🟩 평균 수준  ·  🟨 하위 25%  ·  🟥 상위 25% (생중량 기준)")

            # ── 생중량 분포 (박스플롯) ────────────────────────────────────
            if bw_present and len(gw) >= 2:
                st.subheader("📦 생중량 분포 (박스플롯)")
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(
                    y=list(gw.values),
                    name="생중량",
                    fillcolor="#4CAF50",
                    line=dict(color="#2E7D32"),
                    boxpoints="outliers",
                    marker=dict(color="#2E7D32", size=6),
                ))
                fig_box.update_yaxes(title_text="생중량 (g)")
                set_font(fig_box)
                st.plotly_chart(fig_box, width="stretch")
                st.caption("상자 가운데 선 = 중앙값, 상자 위아래 = 25%·75% 구간, 점 = 이상치")

            # ── 상관관계 분석 (산점도 + 추세선) ──────────────────────────
            st.subheader("🔗 상관관계 분석")
            corr_pairs = []
            if xcol in growth_df.columns and bw_present:
                corr_pairs.append((xcol, "잎 수 vs 생중량"))
            if scol in growth_df.columns and bw_present:
                corr_pairs.append((scol, "지상부 길이 vs 생중량"))

            if corr_pairs:
                fig_corr = make_subplots(
                    rows=len(corr_pairs), cols=1,
                    subplot_titles=[t for _, t in corr_pairs],
                    vertical_spacing=0.1,
                )
                for i, (col, title) in enumerate(corr_pairs):
                    valid_idx = growth_df[col].notna() & growth_df[gcol].notna()
                    x_s = growth_df.loc[valid_idx, col]
                    y_s = growth_df.loc[valid_idx, gcol]
                    if len(x_s) >= 2:
                        x_v = x_s.values.astype(float)
                        y_v = y_s.values.astype(float)
                        r = float(np.corrcoef(x_v, y_v)[0, 1])
                        fig_corr.add_trace(go.Scatter(
                            x=list(x_v), y=list(y_v),
                            mode="markers",
                            marker=dict(size=8, color="#2E86AB",
                                        line=dict(width=1, color="white")),
                            name=title,
                            hovertemplate=(f"{title}<br>"
                                           f"x: %{{x:.1f}}<br>"
                                           f"y: %{{y:.2f}}<extra></extra>"),
                        ), row=i+1, col=1)
                        # np.polyfit으로 추세선 직접 추가
                        coef = np.polyfit(x_v, y_v, 1)
                        xs_line = np.array([float(x_v.min()), float(x_v.max())])
                        ys_line = coef[0] * xs_line + coef[1]
                        fig_corr.add_trace(go.Scatter(
                            x=list(xs_line), y=list(ys_line),
                            mode="lines",
                            line=dict(color="#E84855", width=2, dash="dash"),
                            name=f"추세선",
                            showlegend=False,
                        ), row=i+1, col=1)
                        fig_corr.update_xaxes(title_text=col, row=i+1, col=1)
                        fig_corr.update_yaxes(title_text="생중량 (g)", row=i+1, col=1)
                        st.markdown(
                            f"**{title}**  \n"
                            f"- 상관계수 r = **{r:.2f}**  "
                            f"(−1 ~ 1, 1에 가까울수록 함께 증가, −1에 가까울수록 함께 감소)"
                        )
                set_font(fig_corr)
                st.plotly_chart(fig_corr, width="stretch")
            else:
                st.info("상관계수를 계산할 측정 항목(잎 수·지상부 길이)과 생중량 열이 필요합니다.")

            # ── 원본 테이블 + 다운로드 ────────────────────────────────────
            with st.expander("📋 생육 데이터 원본 + CSV 다운로드"):
                st.dataframe(growth_df, width="stretch")
                csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school_name}_생육결과데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

st.caption(f"🏫 학교명: {school_name}  ·  ⚡ Streamlit Cloud 호환")
