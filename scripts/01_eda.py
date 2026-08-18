"""
1단계 EDA — 센서 데이터에 고장 징후가 실제로 남아 있는가
========================================================

실행:  python scripts/01_eda.py

이 스크립트가 답하려는 질문:
    Q1. 같은 기종 설비의 수명은 얼마나 제각각인가?   (예방보전의 한계 근거)
    Q2. 센서 21개 중 실제로 열화 신호를 담고 있는 것은 몇 개인가?
    Q3. 그 신호는 고장 몇 사이클 전부터 눈에 보이는가?
    Q4. 센서 하나만으로 정상/고장을 가를 수 있는가?
    Q5. "얼마나 일찍" 잡으려 하면 얼마나 어려워지는가?   <- 이 프로젝트의 진짜 문제

산출물: reports/figures/*.png, reports/metrics/eda_sensor_stats.csv
"""

from __future__ import annotations

# --- import 경로 설정 -------------------------------------------------------
# 파이썬은 "실행한 파일이 있는 폴더"를 기준으로 모듈을 찾습니다.
# 이 파일은 scripts/ 안에 있으므로, 그냥 실행하면 scripts/ 를 기준으로 찾다가
# 프로젝트 루트에 있는 src/ 를 못 찾아 ModuleNotFoundError 가 납니다.
# 그래서 프로젝트 루트를 검색 경로 맨 앞에 직접 추가해 줍니다.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import ROOT
from src.data.load import ID_COL, SENSOR_COLS, SENSOR_MEANING, TIME_COL, load_cmapss
from src.features.select import usable_sensors
from src.viz import style
from src.viz.style import CAT, INK_SUB, MUTED, save

SUBSET = "FD001"

HEALTHY_RUL = 125   # RUL 이 이보다 크면 '아직 멀쩡' (기준 구간)
CRITICAL_RUL = 30   # RUL 이 이보다 작으면 '고장 임박'
TREND_THRESHOLD = 0.2

# 조기 탐지 난이도를 볼 RUL 구간 (왼쪽이 과거, 오른쪽이 고장 직전)
RUL_BANDS = [(100, 125), (75, 100), (50, 75), (30, 50), (0, 30)]


# =============================================================================
# 분석
# =============================================================================
def sensor_trend_table(train: pd.DataFrame) -> pd.DataFrame:
    """
    센서마다 '열화 추세가 얼마나 뚜렷한가'를 두 가지로 측정한다.

    1) 추세 상관(rho): 설비 개체마다 (운전시간 vs 센서값) 스피어만 상관을 구해 평균.
       +1 이면 시간이 갈수록 꾸준히 상승, -1 이면 꾸준히 하락, 0 이면 무관.
       직선 관계가 아니어도 '한 방향으로 꾸준한가'만 보면 되므로 스피어만을 씁니다.

    2) 효과크기(d): (고장 직전 20사이클 평균 - 초기 20사이클 평균) / 전체 표준편차.
       '노이즈 대비 얼마나 크게 움직였나'. 2 를 넘으면 매우 큰 변화입니다.
    """
    rows = []
    for c in SENSOR_COLS:
        s = train[c]
        rhos = [
            spearmanr(g[TIME_COL], g[c]).statistic
            for _, g in train.groupby(ID_COL)
            if g[c].nunique() > 1
        ]
        rho = float(np.nanmean(rhos)) if rhos else np.nan

        early = train.loc[train[TIME_COL] <= 20, c].mean()
        late = train.loc[train["RUL"] <= 20, c].mean()
        d = (late - early) / s.std() if s.std() > 0 else 0.0

        rows.append((c, SENSOR_MEANING[c], rho, d, s.nunique(), s.std()))

    df = pd.DataFrame(
        rows, columns=["sensor", "meaning", "trend_rho", "effect_size", "n_unique", "std"]
    )
    df["abs_rho"] = df["trend_rho"].abs()
    return df.sort_values("abs_rho", ascending=False, na_position="last").reset_index(drop=True)


def auc(reference: np.ndarray, target: np.ndarray) -> float:
    """
    두 집단을 얼마나 잘 구분하는지 (AUC).

    0.5 = 동전던지기(전혀 구분 못함), 1.0 = 완벽히 구분.
    순위합으로 직접 계산합니다. (무엇을 재는 값인지 코드에 드러나도록)
    방향(오르는 센서/내리는 센서)과 무관하게 '구분력' 자체만 봅니다.
    """
    both = np.concatenate([reference, target])
    ranks = pd.Series(both).rank().to_numpy()
    n1, n2 = len(reference), len(target)
    v = (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)
    return max(v, 1 - v)


def early_detection_curve(train: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    '정상 구간' 과 '각 RUL 구간' 을 센서 하나로 얼마나 구분할 수 있는지.

    고장 직전만 보면 쉽습니다. 진짜 문제는 얼마나 일찍 알아채느냐입니다.
    이 표가 그 난이도를 숫자로 보여줍니다.
    """
    base = {c: train.loc[train["RUL"] > HEALTHY_RUL, c].to_numpy() for c in cols}
    rows = []
    for lo, hi in RUL_BANDS:
        seg = train[(train["RUL"] > lo) & (train["RUL"] <= hi)]
        row = {"band": f"{lo}~{hi}", "lo": lo, "hi": hi}
        for c in cols:
            row[c] = auc(base[c], seg[c].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 그림
# =============================================================================
def fig1_lifetime_distribution(train: pd.DataFrame) -> None:
    """Q1. 같은 기종인데 수명이 얼마나 다른가."""
    life = train.groupby(ID_COL)[TIME_COL].max()

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(life, bins=24, color=CAT[0], edgecolor=style.SURFACE, linewidth=1.2)

    ax.axvline(life.mean(), color=INK_SUB, linewidth=1.5)
    ax.annotate(f"평균 {life.mean():.0f}", xy=(life.mean(), ax.get_ylim()[1] * 0.95),
                xytext=(7, 0), textcoords="offset points",
                color=INK_SUB, fontsize=9, va="top")
    ymax = ax.get_ylim()[1]
    for val, label, dx in [(life.min(), f"최단 {life.min()}", 14),
                           (life.max(), f"최장 {life.max()}", -14)]:
        ax.annotate(
            label, xy=(val, 1.2), xytext=(val + dx, ymax * 0.62),
            ha="left" if dx > 0 else "right", va="center",
            color=CAT[1], fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=CAT[1], linewidth=1.0,
                            shrinkA=2, shrinkB=2),
        )

    ax.set_title(f"같은 기종 엔진 {len(life)}대의 수명 분포 — 최대 {life.max()/life.min():.1f}배 차이")
    ax.set_xlabel("고장까지의 운전 사이클")
    ax.set_ylabel("엔진 수")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    save(fig, "01_lifetime_distribution.png")


def fig2_sensor_trend_ranking(stats: pd.DataFrame, keep: list[str]) -> None:
    """Q2. 어떤 센서가 열화 신호를 담고 있는가."""
    d = stats.copy()
    d["plot_rho"] = d["trend_rho"].fillna(0.0)
    d["useful"] = d["sensor"].isin(keep)
    d = d.sort_values("plot_rho")

    labels = [f"{r.sensor.replace('sensor_', 'S')}  {r.meaning.split(' (')[0]}"
              for r in d.itertuples()]
    colors = [CAT[0] if u else MUTED for u in d["useful"]]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(d)), d["plot_rho"], color=colors, height=0.72)

    # 값이 0 인 센서는 막대 길이가 0 이라 화면에서 사라집니다.
    # 그대로 두면 "데이터가 빠졌나?" 로 오해되므로, 측정은 했으나 신호가 없다는
    # 뜻으로 0 위치에 점을 찍어 줍니다.
    zero_idx = [i for i, v in enumerate(d["plot_rho"]) if abs(v) < 1e-6]
    ax.scatter([0] * len(zero_idx), zero_idx, s=26, color=MUTED, zorder=5)

    ax.set_yticks(range(len(d)), labels)
    ax.axvline(0, color=INK_SUB, linewidth=1.0)

    ax.set_title(f"센서별 열화 추세 강도 — 21개 중 {len(keep)}개만 신호를 담고 있다")
    ax.set_xlabel("추세 상관계수 ρ   (양수 = 시간에 따라 상승, 음수 = 하락, 0 = 무관)")
    ax.set_xlim(-1, 1)
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT[0]),
               plt.Rectangle((0, 0), 1, 1, color=MUTED)]
    ax.legend(handles, ["활용", "제외 (상수이거나 고유값 2개 이하)"], loc="lower right")
    save(fig, "02_sensor_trend_ranking.png")


def fig3_degradation_curves(train: pd.DataFrame, top: list[str]) -> None:
    """Q3. 열화 신호는 고장 몇 사이클 전부터 보이는가."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    rng = np.random.default_rng(42)
    sample_units = rng.choice(train[ID_COL].unique(), size=40, replace=False)

    for ax, col in zip(axes.ravel(), top):
        for u in sample_units:
            g = train[train[ID_COL] == u]
            ax.plot(g["RUL"], g[col], color=MUTED, linewidth=0.6, alpha=0.45)

        med = train.groupby("RUL")[col].median()
        med = med[med.index <= 250]
        ax.plot(med.index, med.values, color=CAT[0], linewidth=2.4, zorder=5)
        # x축이 반전(250 -> 0)되어 있으므로 med.index[-1] 이 화면 왼쪽 끝입니다.
        # ha="right" 로 두면 축 바깥으로 나가 잘리므로 오른쪽으로 밀어냅니다.
        ax.annotate("전체 중앙값", xy=(med.index[-1], med.iloc[-1]),
                    xytext=(8, 12), textcoords="offset points",
                    color=CAT[0], fontsize=9, fontweight="bold", ha="left")

        ax.axvspan(0, CRITICAL_RUL, color=CAT[1], alpha=0.10, zorder=0)
        ax.set_xlim(250, 0)                       # 왼쪽 = 과거, 오른쪽 = 고장 시점
        ax.set_title(f"{col.replace('sensor_', 'S')}  {SENSOR_MEANING[col]}", fontsize=10)
        ax.set_xlabel("잔여수명 RUL (사이클)")
        ax.set_axisbelow(True)

    fig.suptitle("열화 곡선 — 엔진 40대를 고장 시점에 맞춰 정렬 (오른쪽 끝이 고장)",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005, f"주황 음영 = 고장 임박 구간 (RUL ≤ {CRITICAL_RUL})",
             ha="center", color=INK_SUB, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    save(fig, "03_degradation_curves.png")


def fig4_healthy_vs_critical(train: pd.DataFrame, top: list[str], aucs: dict[str, float]) -> None:
    """Q4. 고장 임박 구간은 센서 하나로도 갈린다 (= 쉬운 문제)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))

    for ax, col in zip(axes.ravel(), top):
        healthy = train.loc[train["RUL"] > HEALTHY_RUL, col]
        critical = train.loc[train["RUL"] <= CRITICAL_RUL, col]
        lo, hi = train[col].quantile([0.001, 0.999])
        bins = np.linspace(lo, hi, 45)

        ax.hist(healthy, bins=bins, color=CAT[0], alpha=0.62, density=True,
                label=f"정상 (RUL > {HEALTHY_RUL})")
        ax.hist(critical, bins=bins, color=CAT[1], alpha=0.62, density=True,
                label=f"고장 임박 (RUL ≤ {CRITICAL_RUL})")

        ax.set_title(f"{col.replace('sensor_', 'S')} "
                     f"{SENSOR_MEANING[col].split(' (')[0]}   —   AUC {aucs[col]:.3f}",
                     fontsize=10)
        ax.set_yticks([])
        ax.set_axisbelow(True)
        ax.grid(axis="y", visible=False)

    axes[0, 0].legend(loc="upper left")
    fig.suptitle("고장 임박 구간은 센서 하나로도 거의 완벽히 갈린다 — 그러나 이건 쉬운 문제다",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "04_healthy_vs_critical.png")


def fig5_early_detection(curve: pd.DataFrame, top: list[str]) -> None:
    """Q5. 일찍 잡으려 할수록 얼마나 어려워지는가 — 이 프로젝트의 진짜 문제."""
    best = top[0]
    x = range(len(curve))

    fig, ax = plt.subplots(figsize=(9, 5))

    # 나머지 센서는 배경으로 물러나게 (강조 패턴)
    for c in top[1:]:
        ax.plot(x, curve[c], color=MUTED, linewidth=1.2, alpha=0.8)
    mid = len(curve) // 2
    ax.annotate("다른 센서들", xy=(mid, curve[top[1:]].iloc[mid].min()),
                xytext=(0, -20), textcoords="offset points",
                color=INK_SUB, fontsize=9, ha="center")

    ax.plot(x, curve[best], color=CAT[0], linewidth=2.6, marker="o", markersize=7,
            markeredgecolor=style.SURFACE, markeredgewidth=2, zorder=5)
    ax.annotate(f"{best.replace('sensor_', 'S')} (최고 성능 센서)",
                xy=(0, curve[best].iloc[0]), xytext=(10, 48), textcoords="offset points",
                color=CAT[0], fontsize=10, fontweight="bold")

    for i, v in [(0, curve[best].iloc[0]), (len(curve) - 1, curve[best].iloc[-1])]:
        ax.annotate(f"{v:.2f}", xy=(i, v), xytext=(0, -18), textcoords="offset points",
                    ha="center", color=CAT[0], fontsize=10, fontweight="bold")

    ax.axhline(0.5, color=INK_SUB, linewidth=1.0)
    ax.annotate("0.5 = 동전던지기 (전혀 구분 못함)", xy=(0, 0.5), xytext=(4, 5),
                textcoords="offset points", color=INK_SUB, fontsize=9)

    ax.set_xticks(list(x), curve["band"])
    ax.set_ylim(0.45, 1.02)
    ax.set_title("조기 탐지는 급격히 어려워진다 — 고장 100사이클 전에는 사실상 신호가 묻힌다")
    ax.set_xlabel("잔여수명 RUL 구간   (왼쪽 = 이른 시점,  오른쪽 = 고장 직전)")
    ax.set_ylabel("정상 구간과의 구분력 (AUC)")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    save(fig, "05_early_detection_difficulty.png")


# =============================================================================
# main
# =============================================================================
def main() -> None:
    style.setup()
    train, _ = load_cmapss(SUBSET, rul_cap=None)

    print(f"=== {SUBSET} EDA ===")
    print(f"행 {len(train):,} / 엔진 {train[ID_COL].nunique()}대 / 컬럼 {train.shape[1]}개\n")

    # --- Q1 -----------------------------------------------------------------
    life = train.groupby(ID_COL)[TIME_COL].max()
    print("[Q1] 설비 수명은 얼마나 제각각인가")
    print(f"  최단 {life.min()} / 중앙 {life.median():.0f} / 최장 {life.max()} 사이클 "
          f"→ {life.max()/life.min():.1f}배 차이")
    print(f"  최단 수명에 맞춰 일괄 교체하면 평균 {(1 - life.min()/life.mean())*100:.0f}% 의 "
          f"수명을 버리게 됨\n")

    # --- Q2 -----------------------------------------------------------------
    keep, drop = usable_sensors(train, SENSOR_COLS)
    stats = sensor_trend_table(train)
    strong = stats[stats["abs_rho"] >= TREND_THRESHOLD]["sensor"].tolist()

    print("[Q2] 센서 21개 중 쓸 만한 것은")
    print(f"  제외 {len(drop)}개 (상수 또는 고유값 2개 이하): {drop}")
    print(f"  사용 {len(keep)}개")
    print(f"  그중 추세가 뚜렷한 것 (|ρ| ≥ {TREND_THRESHOLD}): {len(strong)}개\n")
    print(stats[["sensor", "meaning", "trend_rho", "effect_size"]].head(6).to_string(index=False))

    # --- Q4 -----------------------------------------------------------------
    aucs = {
        c: auc(train.loc[train["RUL"] > HEALTHY_RUL, c].to_numpy(),
               train.loc[train["RUL"] <= CRITICAL_RUL, c].to_numpy())
        for c in keep
    }
    stats["auc_healthy_vs_critical"] = stats["sensor"].map(aucs)
    top4 = [c for c in stats["sensor"] if c in keep][:4]
    best = max(aucs, key=aucs.get)

    print(f"\n[Q4] 센서 하나만으로 정상 vs 고장임박을 구분할 수 있는가")
    print(f"  최고 단일 센서: {best}  AUC {aucs[best]:.3f}")
    print(f"  → 구분된다. 다만 이건 RUL>125 와 RUL<=30 이라는 '멀리 떨어진 두 구간'을")
    print(f"    비교한 것이라 쉬운 문제였다. 진짜 문제는 Q5.\n")

    # --- Q5 -----------------------------------------------------------------
    curve = early_detection_curve(train, top4)
    print("[Q5] 얼마나 일찍 잡을 수 있는가 (정상 구간 대비 AUC)")
    print(f"  {'RUL 구간':<10}" + "".join(f"{c.replace('sensor_','S'):>8}" for c in top4))
    for _, r in curve.iterrows():
        print(f"  {r['band']:<10}" + "".join(f"{r[c]:>8.3f}" for c in top4))
    first, last = curve[best].iloc[0], curve[best].iloc[-1]
    print(f"\n  고장 직전(RUL 0~30):  AUC {last:.3f}  → 거의 완벽")
    print(f"  100사이클 전:         AUC {first:.3f}  → 동전던지기에 가까움")
    print(f"  ** 이 격차가 이 프로젝트가 풀어야 할 문제다. **")
    print(f"  ** 고장 임박을 맞히는 건 쉽다. 일찍 알아채는 것이 어렵다. **\n")

    # --- 저장 ---------------------------------------------------------------
    metrics_dir = ROOT / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(metrics_dir / "eda_sensor_stats.csv", index=False, encoding="utf-8-sig")
    curve.to_csv(metrics_dir / "eda_early_detection.csv", index=False, encoding="utf-8-sig")
    print(f"  저장: reports/metrics/eda_sensor_stats.csv")
    print(f"  저장: reports/metrics/eda_early_detection.csv")

    print("\n=== 그림 생성 ===")
    fig1_lifetime_distribution(train)
    fig2_sensor_trend_ranking(stats, keep)
    fig3_degradation_curves(train, top4)
    fig4_healthy_vs_critical(train, top4, aucs)
    fig5_early_detection(curve, top4)
    print("\n완료.")


if __name__ == "__main__":
    main()
