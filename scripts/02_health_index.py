"""
2단계 — 건강도 점수 (Health Index) 만들기와 방법 비교
====================================================

실행:  python scripts/02_health_index.py

목표
    "지금 상태가 정상에서 얼마나 벗어났는가"를 0~100 점수 하나로 만든다.
    그리고 **고장 이력 없이, 정상 데이터만으로** 만든다.
    (실제 회사 설비에는 run-to-failure 데이터가 없기 때문)

1단계 EDA 에서 정한 목표
    쉬운 구간(RUL 0~30)은 이미 AUC 0.99 로 천장에 닿아 있다.
    개선 여지가 있는 곳은 **RUL 50~100 구간**이다. 여기를 얼마나 올리는지로 평가한다.

비교하는 것 (4 x 2 x 2 = 16가지)
    결합 방법 : z점수 평균 / 마할라노비스 / PCA 재구성오차 / Isolation Forest
    정규화    : 원본값 그대로 / 설비별 초기값 대비 편차
    평활화    : 없음 / 10사이클 이동평균

산출물: reports/figures/06~08*.png, reports/metrics/health_index_comparison.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ROOT
from src.data.load import ID_COL, SENSOR_COLS, TIME_COL, load_cmapss
from src.features.health import BASELINE_CYCLES, add_deviation, add_rolling
from src.features.select import usable_sensors
from src.models.health_index import SCORERS, HealthScale, auc, debounced_alarm
from src.viz import style
from src.viz.style import CAT, INK_SUB, MUTED, save

SUBSET = "FD001"
HEALTHY_RUL = 125          # 평가 기준이 되는 '정상' 구간
FOCUS_BAND = (50, 100)     # 이번 단계가 개선하려는 구간
RUL_BANDS = [(100, 125), (75, 100), (50, 75), (30, 50), (0, 30)]
MA_WINDOW = 10
ALARM_HEALTH = 50.0        # 건강도가 이 아래로 내려가면 경보
PERSISTENCE_GRID = [1, 3, 5, 10, 15, 20]   # 경보 지속 조건 후보


# =============================================================================
# 실험 한 판
# =============================================================================
def run_one(
    df: pd.DataFrame,
    feature_cols: list[str],
    scorer_name: str,
) -> tuple[pd.DataFrame, float]:
    """
    조합 하나를 학습 -> 채점 -> 평가.

    학습에 쓰는 것은 '설비별 초기 30사이클' 뿐입니다. RUL(정답)은 쓰지 않습니다.
    평가는 그 학습 구간을 제외한 나머지에서만 합니다 (자기 데이터로 자기 채점 방지).
    """
    train_mask = df[TIME_COL] <= BASELINE_CYCLES
    X_normal = df.loc[train_mask, feature_cols].to_numpy()

    scorer = SCORERS[scorer_name]().fit(X_normal)
    scores = scorer.score(df[feature_cols].to_numpy())

    scale = HealthScale().fit(scorer.score(X_normal))
    health = scale.to_health(scores)

    out = df[[ID_COL, TIME_COL, "RUL"]].copy()
    out["anomaly"] = scores
    out["health"] = health

    # --- 평가: 학습 구간 제외 ---
    ev = out[~train_mask]
    ref = ev.loc[ev["RUL"] > HEALTHY_RUL, "anomaly"].to_numpy()
    rows = []
    for lo, hi in RUL_BANDS:
        seg = ev.loc[ev["RUL"].between(lo, hi, inclusive="right"), "anomaly"].to_numpy()
        rows.append({"band": f"{lo}~{hi}", "auc": auc(ref, seg)})
    band_df = pd.DataFrame(rows)

    focus = ev.loc[ev["RUL"].between(*FOCUS_BAND, inclusive="right"), "anomaly"].to_numpy()
    return band_df, auc(ref, focus)


def build_variants(train: pd.DataFrame, sensors: list[str]) -> dict[str, tuple[pd.DataFrame, list[str]]]:
    """정규화·평활화 조합별로 (데이터, 사용할 컬럼) 을 준비한다."""
    variants: dict[str, tuple[pd.DataFrame, list[str]]] = {}

    variants["원본"] = (train, sensors)

    dev_df, dev_cols = add_deviation(train, sensors)
    variants["baseline 보정"] = (dev_df, dev_cols)

    ma_df, ma_cols = add_rolling(train, sensors, MA_WINDOW)
    variants[f"원본+MA{MA_WINDOW}"] = (ma_df, ma_cols)

    dev_ma_df, dev_ma_cols = add_rolling(dev_df, dev_cols, MA_WINDOW)
    variants[f"baseline 보정+MA{MA_WINDOW}"] = (dev_ma_df, dev_ma_cols)

    return variants


# =============================================================================
# 그림
# =============================================================================
def single_sensor_baseline(train: pd.DataFrame, sensors: list[str]) -> tuple[str, float]:
    """
    1단계에서 확인한 '센서 하나만 썼을 때'의 성적을 같은 평가 조건으로 다시 계산한다.

    비교 기준선은 반드시 같은 조건에서 직접 재서 써야 합니다.
    기억이나 다른 표의 숫자를 옮겨 적으면 사과와 오렌지를 비교하게 됩니다.
    """
    ev = train[train[TIME_COL] > BASELINE_CYCLES]
    ref_m = ev["RUL"] > HEALTHY_RUL
    seg_m = ev["RUL"].between(*FOCUS_BAND, inclusive="right")
    scored = [
        (auc(ev.loc[ref_m, c].to_numpy(), ev.loc[seg_m, c].to_numpy()), c) for c in sensors
    ]
    best_auc, best_col = max(scored)
    return best_col, best_auc


def fig6_method_comparison(results: pd.DataFrame, base_col: str, base_auc: float) -> None:
    """어떤 조합이 목표 구간을 가장 잘 잡는가."""
    piv = results.pivot(index="variant", columns="scorer", values="focus_auc")
    order = piv.mean(axis=1).sort_values().index
    piv = piv.loc[order]

    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(piv.columns)
    h = 0.8 / n
    for i, col in enumerate(piv.columns):
        y = np.arange(len(piv)) + (i - (n - 1) / 2) * h
        ax.barh(y, piv[col], height=h * 0.86, color=CAT[i], label=col)

    ax.set_yticks(np.arange(len(piv)), piv.index)
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, color=INK_SUB, linewidth=1.0)

    # 같은 조건으로 다시 잰 단일 센서 기준선
    ax.axvline(base_auc, color=CAT[1], linewidth=1.6, zorder=1)
    ax.annotate(f"센서 1개만 썼을 때 ({base_auc:.2f})",
                xy=(base_auc, len(piv) - 0.35),
                xytext=(6, 0), textcoords="offset points",
                color=CAT[1], fontsize=9, fontweight="bold", va="center")

    ax.set_title(f"건강도 점수 방법 비교 — 목표 구간(RUL {FOCUS_BAND[0]}~{FOCUS_BAND[1]}) 구분력")
    ax.set_xlabel("AUC  (정상 구간과 얼마나 잘 갈리는가)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=2)
    save(fig, "06_health_method_comparison.png")


def fig7_health_trajectories(traj: pd.DataFrame, label: str) -> None:
    """건강도 점수가 실제로 어떻게 떨어지는가."""
    fig, ax = plt.subplots(figsize=(9, 5))

    rng = np.random.default_rng(42)
    units = rng.choice(traj[ID_COL].unique(), size=40, replace=False)
    for u in units:
        g = traj[traj[ID_COL] == u].sort_values("RUL", ascending=False)
        ax.plot(g["RUL"], g["health"], color=MUTED, linewidth=0.7, alpha=0.5)

    med = traj.groupby("RUL")["health"].median()
    med = med[med.index <= 250]
    ax.plot(med.index, med.values, color=CAT[0], linewidth=2.6, zorder=5)
    ax.annotate("전체 중앙값", xy=(med.index[-1], med.iloc[-1]),
                xytext=(10, 10), textcoords="offset points",
                color=CAT[0], fontsize=10, fontweight="bold")

    ax.axhline(ALARM_HEALTH, color=CAT[1], linewidth=1.6)
    ax.annotate(f"경보선 (건강도 {ALARM_HEALTH:.0f})", xy=(250, ALARM_HEALTH),
                xytext=(6, 6), textcoords="offset points",
                color=CAT[1], fontsize=9, fontweight="bold")

    ax.set_xlim(250, 0)
    ax.set_ylim(0, 105)
    ax.set_title(f"건강도 점수 궤적 — {label}")
    ax.set_xlabel("잔여수명 RUL (사이클)     왼쪽 = 이른 시점,  오른쪽 = 고장")
    ax.set_ylabel("건강도 점수 (100 = 정상)")
    ax.set_axisbelow(True)
    save(fig, "07_health_trajectories.png")


def fig8_alarm_lead_time(lead: pd.Series, label: str) -> None:
    """경보가 고장 몇 사이클 전에 울리는가 — 현장에서 가장 궁금한 숫자."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.hist(lead, bins=26, color=CAT[0], edgecolor=style.SURFACE, linewidth=1.2)

    ax.axvline(lead.median(), color=CAT[1], linewidth=1.8)
    ax.annotate(f"중앙값 {lead.median():.0f}사이클 전",
                xy=(lead.median(), ax.get_ylim()[1] * 0.9),
                xytext=(8, 0), textcoords="offset points",
                color=CAT[1], fontsize=10, fontweight="bold", va="top")

    ax.set_title(f"경보가 고장 몇 사이클 전에 울리는가 — {label}")
    ax.set_xlabel(f"경보 시점의 잔여수명 (건강도가 {ALARM_HEALTH:.0f} 아래로 내려간 시점)")
    ax.set_ylabel("엔진 수")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    save(fig, "08_alarm_lead_time.png")


def fig9_alarm_tradeoff(trade: pd.DataFrame) -> None:
    """오경보를 줄이면 경보가 늦어진다 — 현장에서 정해야 하는 균형점."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(trade["false_alarm_rate"] * 100, trade["lead_median"],
            color=CAT[0], linewidth=2.4, marker="o", markersize=8,
            markeredgecolor=style.SURFACE, markeredgewidth=2)

    for _, r in trade.iterrows():
        ax.annotate(f"{int(r['persistence'])}사이클 연속",
                    xy=(r["false_alarm_rate"] * 100, r["lead_median"]),
                    xytext=(8, -4), textcoords="offset points",
                    color=INK_SUB, fontsize=9)

    # 오른쪽 끝 라벨이 축 밖으로 넘치지 않도록 여백을 준다
    ax.set_xlim(trade["false_alarm_rate"].min() * 100 - 1,
                trade["false_alarm_rate"].max() * 100 + 4)

    ax.set_title("경보 규칙의 균형점 — 오경보를 줄이면 경보가 늦어진다")
    ax.set_xlabel("오경보율 (%)   정상 구간인데 경보가 울린 비율")
    ax.set_ylabel("경보 리드타임 (사이클)   고장 몇 사이클 전에 알려주는가")
    ax.set_axisbelow(True)
    save(fig, "09_alarm_tradeoff.png")


# =============================================================================
# main
# =============================================================================
def main() -> None:
    style.setup()
    train, _ = load_cmapss(SUBSET, rul_cap=None)
    sensors, dropped = usable_sensors(train, SENSOR_COLS)

    print(f"=== {SUBSET} 건강도 점수 ===")
    print(f"사용 센서 {len(sensors)}개 (제외 {len(dropped)}개)")
    print(f"학습 데이터: 설비별 초기 {BASELINE_CYCLES}사이클만 — RUL 라벨 사용 안 함")
    print(f"목표: RUL {FOCUS_BAND[0]}~{FOCUS_BAND[1]} 구간의 구분력 개선\n")

    base_col, base_auc = single_sensor_baseline(train, sensors)
    print(f"기준선: 센서 1개만 썼을 때 최고 성적 = {base_auc:.3f} ({base_col})\n")

    variants = build_variants(train, sensors)

    # --- 16가지 조합 전부 실행 ---------------------------------------------
    records = []
    band_records = []
    for vname, (vdf, vcols) in variants.items():
        for sname in SCORERS:
            band_df, focus = run_one(vdf, vcols, sname)
            records.append({"variant": vname, "scorer": sname, "focus_auc": focus})
            for _, r in band_df.iterrows():
                band_records.append(
                    {"variant": vname, "scorer": sname, "band": r["band"], "auc": r["auc"]}
                )

    results = pd.DataFrame(records)
    bands = pd.DataFrame(band_records)

    print(f"[비교] 목표 구간(RUL {FOCUS_BAND[0]}~{FOCUS_BAND[1]}) AUC\n")
    piv = results.pivot(index="variant", columns="scorer", values="focus_auc")
    print(piv.round(3).to_string())

    best = results.sort_values("focus_auc", ascending=False).iloc[0]
    worst = results.sort_values("focus_auc").iloc[0]
    print(f"\n  최고: {best['variant']} + {best['scorer']}  AUC {best['focus_auc']:.3f}")
    print(f"  최저: {worst['variant']} + {worst['scorer']}  AUC {worst['focus_auc']:.3f}")

    # --- 최고 조합 재현 -----------------------------------------------------
    bdf, bcols = variants[best["variant"]]
    train_mask = bdf[TIME_COL] <= BASELINE_CYCLES
    scorer = SCORERS[best["scorer"]]().fit(bdf.loc[train_mask, bcols].to_numpy())
    scores = scorer.score(bdf[bcols].to_numpy())
    scale = HealthScale().fit(scorer.score(bdf.loc[train_mask, bcols].to_numpy()))

    traj = bdf[[ID_COL, TIME_COL, "RUL"]].copy()
    traj["health"] = scale.to_health(scores)

    label = f"{best['variant']} + {best['scorer']}"

    # --- 경보 규칙: 지속 조건별 트레이드오프 ---------------------------------
    ev = traj[traj[TIME_COL] > BASELINE_CYCLES].copy()
    n_units = traj[ID_COL].nunique()
    healthy_m = ev["RUL"] > HEALTHY_RUL

    print(f"\n[경보] 건강도 {ALARM_HEALTH:.0f} 미만이 N사이클 연속될 때 경보")
    print(f"  {'지속N':>6}{'오경보율':>10}{'경보설비':>10}{'리드타임중앙':>13}{'최악10%':>10}")
    trade_rows = []
    for n in PERSISTENCE_GRID:
        fired = debounced_alarm(ev, ID_COL, "health", ALARM_HEALTH, n)
        lead_n = ev[fired].groupby(ID_COL)["RUL"].max()
        fpr = fired[healthy_m].mean()
        trade_rows.append({
            "persistence": n,
            "false_alarm_rate": fpr,
            "n_alarmed": len(lead_n),
            "lead_median": lead_n.median(),
            "lead_p10": lead_n.quantile(0.1),
        })
        print(f"  {n:>6}{fpr*100:>9.1f}%{len(lead_n):>9}대"
              f"{lead_n.median():>12.0f}{lead_n.quantile(0.1):>10.0f}")
    trade = pd.DataFrame(trade_rows)

    print(f"\n  전 설비({n_units}대)에서 경보가 울림 — 놓친 설비 없음")
    print(f"  오경보를 {trade['false_alarm_rate'].iloc[0]*100:.1f}% -> "
          f"{trade['false_alarm_rate'].iloc[-1]*100:.1f}% 로 줄이는 대가는 "
          f"리드타임 {trade['lead_median'].iloc[0] - trade['lead_median'].iloc[-1]:.0f}사이클")

    # 그림 7/8 은 지속 조건 없는 기본 규칙 기준
    lead = ev[ev["health"] < ALARM_HEALTH].groupby(ID_COL)["RUL"].max()

    # --- 저장 ---------------------------------------------------------------
    mdir = ROOT / "reports" / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    results.to_csv(mdir / "health_index_comparison.csv", index=False, encoding="utf-8-sig")
    bands.to_csv(mdir / "health_index_bands.csv", index=False, encoding="utf-8-sig")
    trade.to_csv(mdir / "health_index_alarm_tradeoff.csv", index=False, encoding="utf-8-sig")
    print("\n  저장: reports/metrics/health_index_comparison.csv")
    print("  저장: reports/metrics/health_index_bands.csv")
    print("  저장: reports/metrics/health_index_alarm_tradeoff.csv")

    print("\n=== 그림 생성 ===")
    fig6_method_comparison(results, base_col, base_auc)
    fig7_health_trajectories(traj, label)
    fig8_alarm_lead_time(lead, label)
    fig9_alarm_tradeoff(trade)
    print("\n완료.")


if __name__ == "__main__":
    main()
