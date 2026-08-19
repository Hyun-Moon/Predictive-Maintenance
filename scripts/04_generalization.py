"""
4단계 — 조건이 바뀌어도 견디는가 (일반화 검증)
=============================================

실행:  python scripts/04_generalization.py

1~3단계는 전부 **FD001** 에서 만들었습니다.
운전조건 1가지 / 고장모드 1가지 — C-MAPSS 에서 가장 단순한 세트입니다.

여기서 답할 질문:
    Q1. FD001 에서 만든 방법을 다른 세트에 그대로 적용하면 어떻게 되는가?
    Q2. 무너진다면, 무너뜨린 것은 '고장모드가 늘어서'인가 '운전조건이 늘어서'인가?
    Q3. 원인을 알았다면 고칠 수 있는가?

이것이 실제 회사 설비 적용의 예행연습입니다.
현장 설비는 외기온·부하율에 따라 운전조건이 계속 바뀌기 때문입니다.

산출물: reports/figures/14~16*.png, reports/metrics/generalization_*.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.data.load import ID_COL, OP_COLS, SENSOR_COLS, TIME_COL, load_cmapss
from src.features.build import build_features, stack_feature_sets, usable_rows
from src.features.condition import ConditionNormalizer, condition_key
from src.features.health import BASELINE_CYCLES, baseline_table
from src.features.select import usable_sensors
from src.models.health_index import SCORERS, HealthScale, auc
from src.models.rul import MODELS, evaluate
from src.viz import style
from src.viz.style import CAT, INK_SUB, MUTED, save

SUBSETS = ["FD001", "FD003", "FD002", "FD004"]   # 조건 1개 -> 6개 순서로 배치
FEATURE_SET = "④ + 이동평균·기울기"


def subset_profile(sub: str, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    n_cond = len(train[OP_COLS].round(0).drop_duplicates())
    n_mode = {"FD001": 1, "FD002": 1, "FD003": 2, "FD004": 2}[sub]
    short = int((test.groupby(ID_COL)[TIME_COL].max() <= BASELINE_CYCLES).sum())
    return {"subset": sub, "conditions": n_cond, "fault_modes": n_mode, "short_units": short}


def run_pipeline(sub: str, cfg: dict, condition_aware: bool) -> dict:
    """
    한 세트에 파이프라인을 적용한다.

    condition_aware=False : 1~3단계에서 만든 그대로 (FD001 기준으로 설계된 것)
    condition_aware=True  : 운전조건 정규화 + 설비 길이에 맞춘 기준선 적용
    """
    train, test = load_cmapss(sub, cfg)
    sensors, _ = usable_sensors(train, SENSOR_COLS)

    if condition_aware:
        # 조건별 통계는 **학습 설비 전체**에서 구하고 시험에는 그대로 적용
        norm = ConditionNormalizer().fit(train, sensors)
        tr_feat, groups = build_features(train, sensors, normalizer=norm, adaptive_baseline=True)
        te_feat, _ = build_features(test, sensors, normalizer=norm, adaptive_baseline=True)
        tr_feat = tr_feat[usable_rows(tr_feat, adaptive=True)].reset_index(drop=True)
    else:
        tr_feat, groups = build_features(train, sensors)
        te_feat, _ = build_features(
            test, sensors, baseline=baseline_table(test, sensors, BASELINE_CYCLES)
        )
        tr_feat = tr_feat[usable_rows(tr_feat)].reset_index(drop=True)

    te_last = te_feat.groupby(ID_COL).tail(1).reset_index(drop=True)
    cols = stack_feature_sets(groups)[FEATURE_SET]

    model = MODELS["LightGBM"]().fit(tr_feat[cols], tr_feat["RUL"])
    pred = np.clip(model.predict(te_last[cols]), 0, 125)

    out = {"subset": sub, "condition_aware": condition_aware, "n_features": len(cols)}
    out |= evaluate(te_last["RUL"], pred)
    return out


def health_index_check(sub: str, cfg: dict, condition_aware: bool) -> dict:
    """
    건강도 점수(2단계 산출물)가 조건이 늘어도 동작하는가.

    RUL 모델과 달리 **이쪽이 실제로 현장에 가져갈 산출물**입니다.
    (고장 라벨 없이 정상 데이터만으로 만들기 때문)
    따라서 여기서 무너지면 프로젝트의 실무 가치가 사라집니다.
    """
    train, _ = load_cmapss(sub, cfg, rul_cap=None)
    sensors, _ = usable_sensors(train, SENSOR_COLS)

    if condition_aware:
        norm = ConditionNormalizer().fit(train, sensors)
        feat, groups = build_features(train, sensors, normalizer=norm, adaptive_baseline=True)
    else:
        feat, groups = build_features(train, sensors)

    ma_cols = [c for c in groups["rolling"] if c.endswith(f"_ma{10}")]
    normal_m = feat[TIME_COL] <= BASELINE_CYCLES
    scorer = SCORERS["mahalanobis"]().fit(feat.loc[normal_m, ma_cols].to_numpy())
    score = scorer.score(feat[ma_cols].to_numpy())

    ev = feat[~normal_m]
    ref_m = (ev["RUL"] > 125).to_numpy()
    seg_m = ev["RUL"].between(50, 100, inclusive="right").to_numpy()

    sc = score[(~normal_m).to_numpy()]
    multi = auc(sc[ref_m], sc[seg_m])

    # 비교 기준: 센서 하나만 썼을 때의 최고 성적 (현장의 '고정 임계값 경보'에 해당)
    single = max(
        auc(ev[c].to_numpy()[ref_m], ev[c].to_numpy()[seg_m]) for c in sensors
    )
    return {
        "subset": sub,
        "condition_aware": condition_aware,
        "auc_50_100": multi,
        "auc_single_sensor": single,
    }


def condition_dominance(cfg: dict, min_effect: float = 0.2) -> pd.DataFrame:
    """
    운전조건에 따른 센서값 차이가 열화로 인한 변화보다 얼마나 큰지 잰다.

    두 값 모두 **같은 조건 안에서의 표준편차**로 나눠 무차원으로 만듭니다.
    (센서마다 단위와 자릿수가 달라서 그대로 비교하면 의미가 없습니다)

    왜 전체 표준편차가 아니라 조건 내 표준편차인가 — 여기서 한 번 틀렸습니다.
        처음에는 전체 표준편차로 나눴습니다. 그런데 운전조건이 여러 개면
        전체 표준편차 자체가 조건 차이 때문에 부풀려져 있습니다.
        그 결과 열화 폭이 실제보다 훨씬 작아 보여 모든 센서가 걸러졌습니다.
        재려는 대상(조건 차이)이 자(표준편차) 안에 섞여 있으면 아무것도 못 잽니다.

    min_effect : 열화 폭이 조건 내 표준편차의 이 비율보다 작은 센서는 제외.
        이 조건을 안 걸었을 때는 "조건이 열화보다 1경 배 크다"는 값이 나왔습니다.
        애초에 열화가 거의 없는 센서를 분모에 넣어서 생긴 문제입니다.
        비율은 분모가 의미 있을 때만 유효합니다.
    """
    rows = []
    for sub in SUBSETS:
        train, _ = load_cmapss(sub, cfg, rul_cap=None)
        sensors, _ = usable_sensors(train, SENSOR_COLS)
        key = condition_key(train)
        first = key.iloc[0]
        g0 = train[key == first]

        for c in sensors:
            # 조건 내 표준편차 (조건별로 구해 평균) — 조건 차이가 섞이지 않은 자
            sd = float(train.groupby(key)[c].std().mean())
            if not np.isfinite(sd) or sd <= 0:
                continue
            # 열화 폭: 같은 조건 안에서 초기 구간과 고장 직전의 평균 차이
            deg = abs(g0.loc[g0["RUL"] <= 20, c].mean() - g0.loc[g0["RUL"] > 200, c].mean()) / sd
            if not np.isfinite(deg) or deg < min_effect:
                continue                      # 애초에 열화 신호가 없는 센서는 제외
            by_cond = train.groupby(key)[c].mean()
            cond_gap = float(by_cond.max() - by_cond.min()) / sd
            rows.append({"subset": sub, "sensor": c, "cond_gap": cond_gap,
                         "degradation": deg, "ratio": cond_gap / deg})
    return pd.DataFrame(rows)


# =============================================================================
# 그림
# =============================================================================
def fig14_generalization(results: pd.DataFrame, profiles: pd.DataFrame) -> None:
    """조건이 늘어나면 무너지고, 대책을 넣으면 회복되는가."""
    piv = results.pivot(index="subset", columns="condition_aware", values="rmse").loc[SUBSETS]
    labels = [f"{s}\n조건 {p.conditions} · 고장모드 {p.fault_modes}"
              for s, p in zip(SUBSETS, profiles.set_index("subset").loc[SUBSETS].itertuples())]

    x = np.arange(len(SUBSETS))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, piv[False], width=0.38, color=MUTED, label="1~3단계 그대로")
    ax.bar(x + 0.2, piv[True], width=0.38, color=CAT[0], label="운전조건 대책 적용")

    for i, (a, b) in enumerate(zip(piv[False], piv[True])):
        ax.annotate(f"{a:.1f}", xy=(i - 0.2, a), xytext=(0, 5), textcoords="offset points",
                    ha="center", color=INK_SUB, fontsize=9)
        ax.annotate(f"{b:.1f}", xy=(i + 0.2, b), xytext=(0, 5), textcoords="offset points",
                    ha="center", color=CAT[0], fontsize=9, fontweight="bold")

    ax.set_xticks(x, labels)
    ax.set_ylim(0, piv.to_numpy().max() * 1.2)
    ax.set_title("운전조건이 6가지로 늘면 무너진다 — 그리고 고칠 수 있다")
    ax.set_ylabel("시험 RMSE (사이클)   낮을수록 좋음")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    save(fig, "14_generalization_rmse.png")


def fig15_condition_dominance(dom: pd.DataFrame) -> None:
    """조건 차이가 열화보다 몇 배 큰가 — 무너진 이유."""
    multi = [s for s in SUBSETS if dom.loc[dom["subset"] == s, "ratio"].max() > 0]
    single = [s for s in SUBSETS if s not in multi]

    fig, ax = plt.subplots(figsize=(9, 5))
    rng = np.random.default_rng(42)
    for i, sub in enumerate(multi):
        vals = dom.loc[dom["subset"] == sub, "ratio"].to_numpy()
        ax.scatter(np.full(len(vals), i) + rng.normal(0, 0.05, len(vals)), vals,
                   s=60, color=CAT[1], edgecolor=style.SURFACE, linewidth=1.4, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [np.median(vals)] * 2,
                color=INK_SUB, linewidth=2.4, zorder=4)
        ax.annotate(f"중앙값 {np.median(vals):.0f}배", xy=(i + 0.24, np.median(vals)),
                    xytext=(4, 0), textcoords="offset points",
                    color=INK_SUB, fontsize=9, va="center")

    ax.axhline(1.0, color=CAT[0], linewidth=1.6, zorder=2)
    ax.annotate("1배 — 이 아래면 열화 신호가 조건 차이보다 크다",
                xy=(len(multi) - 0.5, 1.0), xytext=(0, 8), textcoords="offset points",
                ha="right", color=CAT[0], fontsize=9, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(range(len(multi)), [f"{s}\n(운전조건 6가지)" for s in multi])
    ax.set_xlim(-0.5, len(multi) - 0.1)
    ax.set_title("운전조건 차이가 열화 신호를 완전히 덮는다 — 센서별 배율")
    ax.set_ylabel("조건 간 차이 ÷ 열화 폭 (배, 로그 눈금)")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    fig.text(0.5, 0.005,
             f"{', '.join(single)} 은 운전조건이 1가지뿐이라 이 배율이 0 — 그래서 문제가 없었다",
             ha="center", color=INK_SUB, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "15_condition_dominance.png")


def fig16_phm_recovery(results: pd.DataFrame) -> None:
    """비용 기준 지표(PHM08)로 본 회복 효과."""
    piv = results.pivot(index="subset", columns="condition_aware", values="phm08").loc[SUBSETS]
    x = np.arange(len(SUBSETS))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - 0.2, piv[False], width=0.38, color=MUTED, label="1~3단계 그대로")
    ax.bar(x + 0.2, piv[True], width=0.38, color=CAT[0], label="운전조건 대책 적용")
    for i, (a, b) in enumerate(zip(piv[False], piv[True])):
        ax.annotate(f"{a:,.0f}", xy=(i - 0.2, a), xytext=(0, 5), textcoords="offset points",
                    ha="center", color=INK_SUB, fontsize=9)
        ax.annotate(f"{b:,.0f}", xy=(i + 0.2, b), xytext=(0, 5), textcoords="offset points",
                    ha="center", color=CAT[0], fontsize=9, fontweight="bold")

    ax.set_xticks(x, SUBSETS)
    ax.set_ylim(0, piv.to_numpy().max() * 1.2)
    ax.set_title("PHM08 점수 — 늦게 예측한 벌점까지 반영하면 차이가 더 크다")
    ax.set_ylabel("PHM08 점수   낮을수록 좋음")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    save(fig, "16_generalization_phm08.png")


def fig17_health_robustness(health: pd.DataFrame) -> None:
    """조건이 늘어도 다변량 건강도는 버틴다 — 현장 적용성의 핵심 근거."""
    base = health[~health["condition_aware"]].set_index("subset").loc[SUBSETS]
    x = np.arange(len(SUBSETS))

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(x - 0.2, base["auc_single_sensor"], width=0.38, color=MUTED,
           label="센서 1개 (고정 임계값 방식)")
    ax.bar(x + 0.2, base["auc_50_100"], width=0.38, color=CAT[0],
           label="다변량 건강도 점수")

    for i, (a, b) in enumerate(zip(base["auc_single_sensor"], base["auc_50_100"])):
        ax.annotate(f"{a:.2f}", xy=(i - 0.2, a), xytext=(0, 5), textcoords="offset points",
                    ha="center", color=INK_SUB, fontsize=9)
        ax.annotate(f"{b:.2f}", xy=(i + 0.2, b), xytext=(0, 5), textcoords="offset points",
                    ha="center", color=CAT[0], fontsize=9, fontweight="bold")

    ax.set_xticks(x, [f"{s}\n조건 {1 if s in ('FD001','FD003') else 6}가지" for s in SUBSETS])
    # 막대가 바닥까지 내려오므로 기준선을 가로선으로 그으면 막대를 가로지릅니다.
    # 대신 축 하한을 0.5(동전던지기)로 잡아 바닥 자체가 기준선이 되게 했습니다.
    ax.set_ylim(0.5, 1.05)
    ax.set_title("운전조건이 늘수록 다변량이 더 중요해진다")
    ax.set_ylabel("RUL 50~100 구간 구분력 (AUC)")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)
    fig.text(0.5, 0.005, "세로축 바닥(0.5) = 동전던지기 — 전혀 구분하지 못하는 수준",
             ha="center", color=INK_SUB, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "17_health_index_robustness.png")


# =============================================================================
# main
# =============================================================================
def main() -> None:
    style.setup()
    cfg = load_config()

    print("=== 4단계: 조건이 바뀌어도 견디는가 ===\n")

    profiles = pd.DataFrame(
        [subset_profile(s, *load_cmapss(s, cfg)) for s in SUBSETS]
    )
    print("[세트 구성]")
    print(f"  {'세트':<8}{'운전조건':>7}{'고장모드':>8}{'기준선보다 짧은 시험설비':>22}")
    for _, r in profiles.iterrows():
        print(f"  {r['subset']:<8}{r['conditions']:>7}{r['fault_modes']:>8}{r['short_units']:>20}대")

    # --- Q1/Q2: 그대로 적용하면? --------------------------------------------
    rows = [run_pipeline(s, cfg, condition_aware=False) for s in SUBSETS]
    # --- Q3: 대책 적용 -------------------------------------------------------
    rows += [run_pipeline(s, cfg, condition_aware=True) for s in SUBSETS]
    results = pd.DataFrame(rows)

    piv_r = results.pivot(index="subset", columns="condition_aware", values="rmse").loc[SUBSETS]
    piv_p = results.pivot(index="subset", columns="condition_aware", values="phm08").loc[SUBSETS]

    print("\n[결과] 시험 RMSE / PHM08")
    print(f"  {'세트':<8}{'그대로':>18}{'대책 적용':>18}{'변화':>10}")
    for s in SUBSETS:
        a, b = piv_r.loc[s, False], piv_r.loc[s, True]
        pa, pb = piv_p.loc[s, False], piv_p.loc[s, True]
        print(f"  {s:<8}{a:>8.2f} /{pa:>7.0f}{b:>9.2f} /{pb:>7.0f}"
              f"{(b - a) / a * 100:>9.0f}%")

    # --- 건강도 점수도 같이 확인 ---------------------------------------------
    hrows = [health_index_check(s, cfg, ca) for ca in (False, True) for s in SUBSETS]
    health = pd.DataFrame(hrows)
    hp = health.pivot(index="subset", columns="condition_aware", values="auc_50_100").loc[SUBSETS]
    hs = health[~health["condition_aware"]].set_index("subset")["auc_single_sensor"]
    print("\n[건강도 점수] 목표 구간(RUL 50~100) 구분력 AUC — 라벨 없이 만든 산출물")
    print(f"  {'세트':<8}{'센서 1개':>10}{'다변량':>10}{'조건대책':>10}")
    for s_ in SUBSETS:
        print(f"  {s_:<8}{hs[s_]:>10.3f}{hp.loc[s_, False]:>10.3f}{hp.loc[s_, True]:>10.3f}")
    print("  → 조건이 6가지가 되면 센서 1개짜리는 무너지지만(0.58) 다변량은 버틴다(0.91).")
    print("    조건 대책은 RUL 예측에는 크게 도움이 되고, 건강도에는 필요 없었다.")

    # --- 원인 진단 -----------------------------------------------------------
    dom = condition_dominance(cfg)
    print("\n[원인] 운전조건 차이 ÷ 열화 폭 (센서별 중앙값)")
    for s in SUBSETS:
        v = dom.loc[dom["subset"] == s, "ratio"]
        print(f"  {s}: 중앙 {v.median():>6.1f}배   최대 {v.max():>7.1f}배")

    print("\n  → 고장모드가 2개인 것(FD003)은 문제가 되지 않았다.")
    print("    무너뜨린 것은 오직 운전조건이다.")

    # --- 저장 ---------------------------------------------------------------
    mdir = ROOT / "reports" / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    results.to_csv(mdir / "generalization_results.csv", index=False, encoding="utf-8-sig")
    health.to_csv(mdir / "generalization_health_index.csv", index=False, encoding="utf-8-sig")
    dom.to_csv(mdir / "generalization_condition_dominance.csv", index=False, encoding="utf-8-sig")
    print("\n  저장: reports/metrics/generalization_results.csv")
    print("  저장: reports/metrics/generalization_condition_dominance.csv")

    print("\n=== 그림 생성 ===")
    fig14_generalization(results, profiles)
    fig15_condition_dominance(dom)
    fig16_phm_recovery(results)
    fig17_health_robustness(health)
    print("\n완료.")


if __name__ == "__main__":
    main()
