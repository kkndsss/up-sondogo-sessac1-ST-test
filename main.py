import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# 한글 폰트 (Streamlit CSS)
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

st.set_page_config(page_title="송도 새싹 1팀", layout="wide")

# ─────────────────────────────────────────────
# CSV 읽기 (인코딩 폴백)
@st.cache_data
def read_csv_file(file):
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ─────────────────────────────────────────────
# 열 이름 표준화
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

# ─────────────────────────────────────────────
# 학교명 추출 (파일명에서)
def school_name(filename):
    base = os.path.basename(filename) if filename else ""
    # 앞부분 학교명만 추출 (언더스코어 이전)
    return base.split("_")[0] if "_" in base else base.split(".")[0]

# ─────────────────────────────────────────────
# CSV 다운로드 헬퍼
def csv_download_button(label, df, filename):
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv")

# ─────────────────────────────────────────────
# 공통: 숫자 열만 골라 통계 요약
def num_summary(df):
    cols = [c for c in df.columns if c not in ("time", "개체번호")]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for c in cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        rows.append({
            "측정 항목": c,
            "평균": round(s.mean(), 2),
            "최소": round(s.min(), 2),
            "최대": round(s.max(), 2),
            "표준편차": round(s.std(), 2),
            "데이터 수": len(s),
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# 사이드바: 파일 업로드
st.sidebar.title("📁 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

env_df = std_cols(read_csv_file(env_file))
growth_df = std_cols(read_csv_file(growth_file))

sn_env = school_name(env_file.name) if env_file else ""
sn_growth = school_name(growth_file.name) if growth_file else ""
school_display = sn_env or sn_growth or "우리 학교"

# 생육 측정 항목 선택 (사이드바)
growth_items = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns]
selected_items = st.sidebar.multiselect(
    "생육 측정 항목 선택",
    growth_items,
    default=growth_items[:2] if len(growth_items) >= 2 else growth_items,
)

# ─────────────────────────────────────────────
# 탭 구성
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 실험 개요",
    "🌡️ 환경 데이터",
    "📊 생육 결과",
    "🔍 가설 검증",
    "📝 한계 및 개선방향",
])

# ═════════════════════════════════════════════
# Tab 1: 실험 개요
# ═════════════════════════════════════════════
with tab1:
    try:
        st.title(f"📖 실험 개요 — {school_display}")

        st.markdown("""
**연구 배경 및 목적**
극지식물은 낮은 온도와 건조한 환경에서도 생존할 수 있는 특수한 식물입니다.
이 실험은 우리 학교 극지식물 개체의 생육 차이를 분석하여,
어떤 개체가 더 잘 자랐는지, 지상부(줄기)와 지하부(뿌리) 길이 사이에
어떤 관계가 있는지 알아보는 데 목적이 있습니다.

- **상관분석**: 지상부 길이와 지하부 길이가 함께 변하는 경향이 있는지 확인합니다.
  상관계수(r)는 -1에서 1 사이 값으로, 1에 가까울수록 같이 증가하고 -1에 가까울수록 반대로 변합니다.
- **평균·표준편차**: 평균은 '대표값', 표준편차는 '값들이 평균에서 얼마나 퍼져있는지'를 나타냅니다.
- **CV(변동계수)**: 표준편차를 평균으로 나눈 값으로, 단위가 다른 항목끼리 안정성을 비교할 때 사용합니다.
        """)

        # 데이터 요약 표
        has_env = not env_df.empty
        has_growth = not growth_df.empty

        if has_growth:
            total = len(growth_df)
            growth_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"] if c in growth_df.columns]
            st.subheader("📋 생육 데이터 요약")
            st.dataframe(
                pd.DataFrame({
                    "항목": ["총 개체 수", "측정 항목"],
                    "값": [total, ", ".join(growth_cols)],
                }),
                width="stretch",
            )
        else:
            st.info("📋 생육 결과 CSV를 업로드하면 요약 정보가 표시됩니다.")

        if has_env:
            times = env_df["time"].dropna()
            if len(times) > 0:
                period = f"{times.min().strftime('%Y-%m-%d')} ~ {times.max().strftime('%Y-%m-%d')}"
            else:
                period = "알 수 없음"
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec", "water_temp", "co2"] if c in env_df.columns]
            st.subheader("🌡️ 환경 데이터 요약")
            st.dataframe(
                pd.DataFrame({
                    "항목": ["측정 기간", "측정 항목"],
                    "값": [period, ", ".join(env_cols)],
                }),
                width="stretch",
            )
        else:
            st.info("🌡️ 환경 데이터 CSV를 업로드하면 요약 정보가 표시됩니다.")

        # 주요 지표 카드 4개
        col1, col2, col3, col4 = st.columns(4)

        if has_growth and "생중량" in growth_df.columns:
            bw = growth_df["생중량"].dropna()
            avg_bw = round(bw.mean(), 2) if len(bw) > 0 else 0
        else:
            avg_bw = 0
        col1.metric("총 개체 수", len(growth_df) if has_growth else 0)

        if has_env and "temperature" in env_df.columns:
            tt = env_df["temperature"].dropna()
            avg_temp = round(tt.mean(), 1) if len(tt) > 0 else 0
        else:
            avg_temp = 0
        col2.metric("평균 온도 (°C)", avg_temp)

        if has_env and "humidity" in env_df.columns:
            hh = env_df["humidity"].dropna()
            avg_hum = round(hh.mean(), 1) if len(hh) > 0 else 0
        else:
            avg_hum = 0
        col3.metric("평균 습도 (%)", avg_hum)

        col4.metric("평균 생중량 (g)", avg_bw)

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═════════════════════════════════════════════
# Tab 2: 환경 데이터
# ═════════════════════════════════════════════
with tab2:
    try:
        st.title(f"🌡️ 환경 데이터 — {school_display}")

        if env_df.empty:
            st.info("환경 데이터 CSV를 업로드하면 여기에 환경 분석이 표시됩니다.")
        else:
            # 환경 요약 표
            st.subheader("📊 환경 요약 (평균·최소·최대)")
            summ = num_summary(env_df)
            if not summ.empty:
                st.dataframe(summ, width="stretch")
            else:
                st.info("수치 데이터가 없어 요약표를 표시할 수 없습니다.")

            # 날짜별 시계열
            st.subheader("📈 날짜별 시계열 (일별 평균)")

            if "time" in env_df.columns:
                env_df_sorted = env_df.dropna(subset=["time"]).copy()
                env_df_sorted["date"] = env_df_sorted["time"].dt.date

                env_plot_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df_sorted.columns]

                if env_plot_cols:
                    daily = env_df_sorted.groupby("date")[env_plot_cols].mean().reset_index()
                    daily["date"] = pd.to_datetime(daily["date"])

                    n_p = min(2, len(env_plot_cols))
                    n_q = (len(env_plot_cols) + n_p - 1) // n_p
                    fig_ts = make_subplots(rows=n_q, cols=n_p, subplot_titles=env_plot_cols)

                    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
                    for i, col in enumerate(env_plot_cols):
                        row = i // n_p + 1
                        col_idx = i % n_p + 1
                        fig_ts.add_trace(
                            go.Scatter(
                                x=daily["date"],
                                y=daily[col].dropna(),
                                mode="lines",
                                name=col,
                                line=dict(color=colors[i % len(colors)], width=2),
                                hovertemplate=f"{col}<br>x=%{{x|%Y-%m-%d}}<br>y=%{{y:.2f}}<extra></extra>",
                            ),
                            row=row,
                            col=col_idx,
                        )

                    fig_ts.update_layout(
                        height=300 * n_q,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        showlegend=True,
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_ts, width="stretch")
                else:
                    st.info("표시할 환경 수치 열(온도·습도·pH·EC)이 없습니다.")
            else:
                st.info("시간(time) 열이 없어 시계열 그래프를 그릴 수 없습니다.")

            # 이상치 확인 (박스플롯)
            st.subheader("🔍 이상치 확인 (박스플롯)")
            box_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if box_cols:
                fig_box = px.box(
                    env_df.dropna(subset=box_cols),
                    y=box_cols,
                    title="환경 데이터 이상치 (박스플롯)",
                    color_discrete_sequence=["#3498db"],
                )
                fig_box.update_layout(
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    yaxis_title="값",
                    showlegend=False,
                )
                st.plotly_chart(fig_box, width="stretch")
            else:
                st.info("박스플롯을 그릴 수치 열이 없습니다.")

            # CV 비교 막대그래프
            st.subheader("📊 환경 데이터별 CV(변동계수) 비교")
            cv_rows = []
            for c in ["temperature", "humidity", "ph", "ec", "water_temp", "co2"]:
                if c in env_df.columns:
                    s = env_df[c].dropna()
                    if len(s) >= 2 and s.mean() != 0:
                        cv_rows.append({"항목": c, "CV": round(s.std() / s.mean(), 3)})
            if cv_rows:
                cv_df = pd.DataFrame(cv_rows).sort_values("CV", ascending=True)
                fig_cv = px.bar(
                    cv_df,
                    x="CV",
                    y="항목",
                    orientation="h",
                    title="CV(변동계수) = 표준편차 ÷ 평균 — 값이 클수록 변동이 큼",
                    color_discrete_sequence=["#e74c3c"],
                )
                fig_cv.update_layout(
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    xaxis_title="CV",
                    showlegend=False,
                )
                st.plotly_chart(fig_cv, width="stretch")
            else:
                st.info("CV를 계산할 수치 열이 없습니다.")

            # expander: 원본 테이블 + 다운로드
            with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(env_df, width="stretch")
                csv_download_button(
                    "📥 환경 데이터 CSV 다운로드",
                    env_df,
                    f"{school_display}_환경데이터.csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═════════════════════════════════════════════
# Tab 3: 생육 결과
# ═════════════════════════════════════════════
with tab3:
    try:
        st.title(f"📊 생육 결과 — {school_display}")

        if growth_df.empty:
            st.info("생육 결과 CSV를 업로드하면 여기에 분석이 표시됩니다.")
        else:
            # 핵심 결과 카드
            st.subheader("🥇 핵심 결과")
            bw_s = growth_df["생중량"].dropna() if "생중량" in growth_df.columns else pd.Series(dtype=float)

            if len(bw_s) >= 2:
                avg_bw_v = round(bw_s.mean(), 2)
                max_idx = bw_s.idxmax()
                min_idx = bw_s.idxmin()
                max_id = growth_df.loc[max_idx, "개체번호"] if "개체번호" in growth_df.columns else "?"
                min_id = growth_df.loc[min_idx, "개체번호"] if "개체번호" in growth_df.columns else "?"
                max_v = round(bw_s.max(), 2)
                min_v = round(bw_s.min(), 2)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("평균 생중량 (g)", avg_bw_v)
                c2.metric("최대 개체", f"{max_id} ({max_v} g)")
                c3.metric("최소 개체", f"{min_id} ({min_v} g)")
                c4.metric("표준편차 (g)", round(bw_s.std(), 2))
            else:
                st.info("생중량 데이터가 충분하지 않아 핵심 카드를 표시할 수 없습니다.")

            # 개체별 생육 비교 막대그래프 (선택 항목)
            if selected_items:
                st.subheader("📊 개체별 생육 비교 (막대그래프)")
                for item in selected_items:
                    if item in growth_df.columns:
                        s = growth_df[[ "개체번호", item ]].dropna()
                        if len(s) >= 2:
                            fig_bar = px.bar(
                                s,
                                x="개체번호",
                                y=item,
                                title=f"개체별 {item}",
                                color_discrete_sequence=["#2ecc71"],
                            )
                            fig_bar.update_layout(
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                xaxis_title="개체번호",
                                yaxis_title=item,
                                showlegend=False,
                            )
                            st.plotly_chart(fig_bar, width="stretch")
                        else:
                            st.info(f"{item} 데이터가 부족해 그래프를 표시할 수 없습니다.")
                    else:
                        st.info(f"{item} 열이 데이터에 없습니다.")
            else:
                st.info("측정 항목을 선택해주세요.")

            # 생중량 분포 (박스플롯)
            st.subheader("📦 생중량 분포")
            if "생중량" in growth_df.columns:
                bw_df = growth_df[["생중량"]].dropna()
                if len(bw_df) >= 2:
                    fig_distr = px.box(
                        bw_df,
                        y="생중량",
                        title="생중량 분포 (박스플롯)",
                        color_discrete_sequence=["#e74c3c"],
                    )
                    fig_distr.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        yaxis_title="생중량 (g)",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_distr, width="stretch")

                    fig_hist = px.histogram(
                        bw_df,
                        x="생중량",
                        title="생중량 히스토그램",
                        nbins=15,
                        color_discrete_sequence=["#3498db"],
                    )
                    fig_hist.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        xaxis_title="생중량 (g)",
                        yaxis_title="빈도",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_hist, width="stretch")
                else:
                    st.info("생중량 데이터가 부족해 분포를 표시할 수 없습니다.")
            else:
                st.info("생중량 열이 없습니다.")

            # expander
            with st.expander("📄 생육 데이터 원본 + CSV 다운로드"):
                st.dataframe(growth_df, width="stretch")
                csv_download_button(
                    "📥 생육 결과 CSV 다운로드",
                    growth_df,
                    f"{school_display}_생육결과데이터.csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═════════════════════════════════════════════
# Tab 4: 가설 검증
# ═════════════════════════════════════════════
with tab4:
    try:
        st.title(f"🔍 가설 검증 — {school_display}")

        if not ("지상부 길이" in growth_df.columns and "지하부 길이" in growth_df.columns):
            st.info("지상부 길이와 지하부 길이 열이 모두 있어야 가설 검증을 표시할 수 있습니다.")
        else:
            above_s = growth_df["지상부 길이"].dropna()
            below_s = growth_df["지하부 길이"].dropna()
            paired = pd.concat([above_s, below_s], axis=1).dropna()

            if len(paired) < 2:
                st.info("지상부·지하부 길이 데이터가 2개 미만이라 상관관계를 계산할 수 없습니다.")
            else:
                a = paired.iloc[:, 0]
                b = paired.iloc[:, 1]
                corr_val = round(a.corr(b), 4)
                slope_v, intercept_v = np.polyfit(a, b, 1)

                # 핵심 결과 카드
                st.subheader("🥇 지상부 vs 지하부 길이 상관관계")
                cc1, cc2 = st.columns(2)
                cc1.metric("상관계수 r", corr_val)
                cc2.metric("추세선 기울기", round(slope_v, 4))

                st.info(
                    f"상관계수 r = {corr_val} 입니다. "
                    f"|r|이 0.7 이상이면 강한 관계, 0.4~0.7은 중간, 0.4 미만은 약한 관계로 해석합니다. "
                    f"현재 r = {corr_val} 으로 {'강한' if abs(corr_val) >= 0.7 else '중간' if abs(corr_val) >= 0.4 else '약한'} 관계입니다."
                )

                # 기초 통계분석표
                st.subheader("📋 지상부·지하부 기초 통계분석표")
                stat_rows = []
                for label, s in [("지상부 길이 (mm)", a), ("지하부 길이 (mm)", b)]:
                    stat_rows.append({
                        "항목": label,
                        "평균": round(s.mean(), 2),
                        "표준편차": round(s.std(), 2),
                        "최소": round(s.min(), 2),
                        "최대": round(s.max(), 2),
                        "데이터 수": len(s),
                    })
                st.dataframe(pd.DataFrame(stat_rows), width="stretch")

                # 평균 비율 원그래프
                st.subheader("🥧 지상부·지하부 길이 평균 비율")
                a_mean = a.mean()
                b_mean = b.mean()
                total_mean = a_mean + b_mean
                if total_mean > 0:
                    fig_pie = go.FigureData = None
                    fig_pie = go.Figure()
                    fig_pie.add_trace(go.Pie(
                        labels=["지상부 길이", "지하부 길이"],
                        values=[a_mean, b_mean],
                        hole=0.4,
                        marker_colors=["#e74c3c", "#3498db"],
                    ))
                    fig_pie.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        title=f"평균 비율 (지상부 {round(a_mean,1)} mm : 지하부 {round(b_mean,1)} mm)",
                    )
                    st.plotly_chart(fig_pie, width="stretch")
                else:
                    st.info("평균이 0이라 원그래프를 그릴 수 없습니다.")

                # 상관관계 히트맵
                st.subheader("🔥 지하부 길이 상관관계 히트맵")
                corr_df = paired.copy()
                corr_df.columns = ["지상부 길이", "지하부 길이"]
                fig_heat = px.imshow(
                    corr_df.corr(),
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title="지상부·지하부 길이 상관계수 행렬",
                    aspect="auto",
                )
                fig_heat.update_layout(
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                )
                st.plotly_chart(fig_heat, width="stretch")

                # 산점도 + 추세선
                st.subheader("📈 지상부·지하부 길이 상관관계 산점도")
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=a,
                    y=b,
                    mode="markers",
                    marker=dict(size=8, color="#e74c3c", opacity=0.7, line=dict(width=1, color="white")),
                    hovertemplate="지상부=%{x:.1f} mm<br>지하부=%{y:.1f} mm<extra></extra>",
                    name="개체",
                ))
                x_range = np.linspace(a.min(), a.max(), 100)
                y_fit = slope_v * x_range + intercept_v
                fig_scatter.add_trace(go.Scatter(
                    x=x_range,
                    y=y_fit,
                    mode="lines",
                    line=dict(color="#2c3e50", width=2, dash="dash"),
                    name=f"추세선 (r={corr_val})",
                ))
                fig_scatter.update_layout(
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    xaxis_title="지상부 길이 (mm)",
                    yaxis_title="지하부 길이 (mm)",
                    showlegend=True,
                    hovermode="closest",
                )
                st.plotly_chart(fig_scatter, width="stretch")

                # 가설 검증 결과 분석 설명
                st.subheader("📝 가설 검증 결과 분석")
                if abs(corr_val) >= 0.7:
                    interp = "강한"
                elif abs(corr_val) >= 0.4:
                    interp = "중간 정도의"
                else:
                    interp = "약한"
                direction = "양의" if corr_val > 0 else "음의"
                st.markdown(f"""
**해석**
- 지상부 길이와 지하부 길이 사이의 상관계수 r = **{corr_val}** 으로 **{direction} {interp} 상관관계**가 나타났습니다.
- r이 0에 가까우면 두 변수 사이에 선형 관계가 거의 없다는 뜻입니다.
- 추세선의 기울기는 **{round(slope_v, 3)}** 로, 지상부가 1 mm 길어질 때 지하부는 약 {round(slope_v, 3)} mm 변화하는 경향이 있습니다.

**참고**
- 상관계수는 두 변수가 '직선적으로' 함께 변하는 정도만을 나타내며, 인과관계를 뜻하지는 않습니다.
- 표본 수가 적거나 특정 개체에 의해 값이 치우쳐 있을 수 있으므로, 결과를 일반화하기 전에 주의가 필요합니다.
                """)
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═════════════════════════════════════════════
# Tab 5: 한계 및 개선방향
# ═════════════════════════════════════════════
with tab5:
    try:
        st.title(f"📝 한계 및 개선방향 — {school_display}")

        st.subheader("🔸 연구 한계")
        st.markdown("""
1. **측정 시점 불일치**: 생육결과데이터에는 생육 측정일이 기록되어 있지 않아 환경데이터의 측정 시점과 생육 결과를 같은 시점에 연결하여 분석하기 어렵습니다.
2. **개체 위치 정보 누락**: 개체의 위치, 화분 번호, 처리구 정보가 없어 개체 간 환경 차이를 반영한 분석이 불가능합니다.
3. **표본 수 제한**: 생육결과데이터의 표본은 81개체로 제한적이어서 상관관계를 일반화하거나 통계적으로 안정적인 결론을 내리기 어렵습니다.
4. **측정 시각 불일치**: 환경데이터의 측정 시각이 일정하지 않아 생육 측정 시점과 환경 조건을 정확히 대응시키기 어렵습니다.
        """)

        st.subheader("🔸 개선 방향")
        st.markdown("""
1. **측정일 기록**: 동일한 실험을 다시 한다면 생육 측정일인 날짜와 시각을 개체번호와 함께 기록합니다.
2. **위치 정보 기록**: 개체의 위치를 화분 번호, 재배대 위치, 광 조건 등과 함께 기록하여 개체 간 환경 차이를 설명할 수 있게 합니다.
3. **측정 시점 일치**: 측정 시각과 생육 측정 시점을 일치시키는 방식으로 환경데이터와 생육결과데이터를 연계하여 분석합니다.
4. **표본 수 확대**: 표본 수를 늘려 여러 개체를 반복 측정하고, 가능하면 실험을 2~3회 반복하여 결과의 신뢰성을 높입니다.
5. **통합 분석**: 지상부 길이와 지하부 길이뿐 아니라 환경 요인과 생육 결과를 함께 비교·분석하여 줄기와 뿌리 생장에 영향을 주는 요인을 탐색합니다.
        """)
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
