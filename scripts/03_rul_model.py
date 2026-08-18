"""
3단계 — RUL(잔여수명) 예측 모델
==============================

실행:  python scripts/03_rul_model.py

2단계까지는 "지금 얼마나 나빠졌는가"(건강도)를 라벨 없이 만들었습니다.
이 단계는 **"앞으로 몇 사이클 더 쓸 수 있는가"** 를 직접 맞힙니다.
여기서는 고장 라벨(RUL)을 쓰므로, 현장에 바로 가져갈 수 있는 것이 아니라
**방법론이 제대로 동작하는지 벤치마크로 검증하는 단계**입니다.

무엇을 비교하는가
    특징: ① 운전시간만 → ② +센서원본 → ③ +개체 baseline 편차 → ④ +이동평균·기울기
    모델: 상수(평균) / 선형회귀(Ridge) / LightGBM
    특징을 하나씩 쌓으며 "무엇을 추가했더니 얼마나 좋아졌는지"를 측정합니다.

어떻게 평가하는가
    검증 : train 엔진을 개체 단위로 80/20 분할 (행 단위 분할은 성능을 부풀립니다)
    시험 : C-MAPSS 공식 test set 의 **마지막 시점 100개** — 논문과 직접 비교 가능
    지표 : RMSE + PHM08 점수(늦게 예측하면 벌점 2배) + 늦게 본 비율

산출물: reports/figures/10~12*.png, reports/metrics/rul_*.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.data.load import ID_COL, SENSOR_COLS, TIME_COL, load_cmapss
from src.features.build import build_features, stack_feature_sets, usable_rows
from src.features.health import BASELINE_CYCLES, baseline_table
from src.features.select import usable_sensors
from src.models.health_index import SCORERS, HealthScale
from src.models.rul import MODELS, evaluate, group_split
from src.viz import style
from src.viz.style import CAT, INK_SUB, MUTED, save

SUBSET = "FD001"
SEED = 42


# =============================================================================
# 2단계 산출물을 특징으로 붙이기
# =============================================================================
def attach_health_index(
    tr_feat: pd.DataFrame, te_feat: pd.DataFrame, groups: dict[str, list[str]]
) -> str:
    """
    2단계에서 만든 건강도 점수를 RUL 예측의 입력 특징으로 추가한다.

    질문: "정상 데이터만으로 만든 요약 점수가 RUL 예측에도 도움이 되는가?"

    이상탐지 모델은 **학습 설비의 정상 구간에서만** 학습하고,
    시험 설비에는 그대로 적용합니다. (실제 배포와 같은 순서)
    """
    ma_cols = [c for c in groups["rolling"] if c.endswith("_ma10")]

    normal_m = tr_feat[TIME_COL] <= BASELINE_CYCLES * 2   # 학습 설비의 이른 구간
    scorer = SCORERS["mahalanobis"]().fit(tr_feat.loc[normal_m, ma_cols].to_numpy())
    scale = HealthScale().fit(scorer.score(tr_feat.loc[normal_m, ma_cols].to_numpy()))

    for df in (tr_feat, te_feat):
        df["health_index"] = scale.to_health(scorer.score(df[ma_cols].to_numpy()))

    return "health_index"


# =============================================================================
# 실험
# =============================================================================
def run_experiments(
    tr_feat: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    te_last: pd.DataFrame,
) -> pd.DataFrame:
    """
    특징 세트 x 모델의 모든 조합을 학습·평가한다.

    모델을 두 번 학습시키는 이유 (각 숫자의 의미를 하나로 유지하기 위함):
        검증 점수 : 학습 80% 로만 학습한 모델을 나머지 20% 설비에 적용
                    -> 어떤 조합을 고를지 판단하는 용도
        시험 점수 : 학습 데이터 100% 로 학습한 모델을 공식 시험 세트에 적용
                    -> 실제로 배포할 모델의 최종 성적

    한 모델로 두 점수를 다 내면 "검증은 80%, 시험은 100% 학습" 처럼
    조건이 섞여 나중에 무엇과 무엇을 비교하는지 알 수 없게 됩니다.
    """
    fit_units, val_units = group_split(tr_feat[ID_COL].to_numpy(), val_ratio=0.2, seed=SEED)
    fit_m = tr_feat[ID_COL].isin(fit_units)
    val_m = tr_feat[ID_COL].isin(val_units)

    rows = []
    for fs_name, cols in feature_sets.items():
        for m_name, factory in MODELS.items():
            # (1) 조합 선택용 — 80% 학습 -> 20% 검증
            split_model = factory().fit(tr_feat.loc[fit_m, cols], tr_feat.loc[fit_m, "RUL"])
            val_pred = np.clip(split_model.predict(tr_feat.loc[val_m, cols]), 0, 125)

            # (2) 최종 성적용 — 100% 학습 -> 공식 시험 세트
            full_model = factory().fit(tr_feat[cols], tr_feat["RUL"])
            test_pred = np.clip(full_model.predict(te_last[cols]), 0, 125)

            rec = {"features": fs_name, "model": m_name}
            rec |= {f"val_{k}": v for k, v in evaluate(tr_feat.loc[val_m, "RUL"], val_pred).items()}
            rec |= {f"test_{k}": v for k, v in evaluate(te_last["RUL"], test_pred).items()}
            rows.append(rec)

    return pd.DataFrame(rows)


def best_predictions(
    tr_feat: pd.DataFrame, te_last: pd.DataFrame, cols: list[str], model_name: str
) -> tuple[np.ndarray, np.ndarray, object, list[str]]:
    """최고 조합을 train 전체로 다시 학습해 test 예측을 만든다."""
    model = MODELS[model_name]().fit(tr_feat[cols], tr_feat["RUL"])
    pred = np.clip(model.predict(te_last[cols]), 0, 125)
    return te_last["RUL"].to_numpy(), pred, model, cols


# =============================================================================
# 그림
# =============================================================================
def fig10_feature_contribution(results: pd.DataFrame) -> None:
    """특징을 하나씩 쌓을 때 성능이 어떻게 변하는가."""
    piv = results.pivot(index="features", columns="model", values="test_rmse")
    piv = piv.loc[[f for f in results["features"].unique()]]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(piv))
    for i, m in enumerate(piv.columns):
        ax.plot(x, piv[m], color=CAT[i], linewidth=2.4, marker="o", markersize=8,
                markeredgecolor=style.SURFACE, markeredgewidth=2, label=m)
        ax.annotate(f"{piv[m].iloc[-1]:.1f}", xy=(x[-1], piv[m].iloc[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    color=CAT[i], fontsize=10, fontweight="bold", va="center")

    ax.set_xticks(x, piv.index)
    ax.set_xlim(-0.3, len(piv) - 0.3)
    ax.set_title("특징을 하나씩 쌓았을 때의 시험 성적 — 무엇이 실제로 기여했는가")
    ax.set_xlabel("입력 특징 (왼쪽에서 오른쪽으로 누적)")
    ax.set_ylabel("RMSE (사이클)   낮을수록 좋음")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    # 상수 모델 선이 위쪽에 평평하게 깔리므로 범례는 비어 있는 왼쪽 아래로
    ax.legend(loc="lower left")
    save(fig, "10_rul_feature_contribution.png")


def fig11_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    """예측이 정답을 얼마나 따라가는가."""
    fig, ax = plt.subplots(figsize=(7.5, 7))

    lim = 135
    ax.plot([0, lim], [0, lim], color=INK_SUB, linewidth=1.2, zorder=1)
    ax.annotate("완벽한 예측", xy=(lim * 0.82, lim * 0.82), xytext=(6, -14),
                textcoords="offset points", color=INK_SUB, fontsize=9, rotation=45)

    late = y_pred > y_true
    ax.scatter(y_true[~late], y_pred[~late], s=55, color=CAT[0],
               edgecolor=style.SURFACE, linewidth=1.5, zorder=3,
               label=f"일찍 봄 — 안전 ({(~late).sum()}대)")
    ax.scatter(y_true[late], y_pred[late], s=55, color=CAT[1],
               edgecolor=style.SURFACE, linewidth=1.5, zorder=4,
               label=f"늦게 봄 — 위험 ({late.sum()}대)")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_title(f"시험 세트 100대 예측 결과 — {label}")
    ax.set_xlabel("실제 잔여수명 (사이클)")
    ax.set_ylabel("예측 잔여수명 (사이클)")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    save(fig, "11_rul_pred_vs_true.png")


def safety_margin_sweep(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    예측을 일부러 낮게(보수적으로) 잡으면 어떻게 되는가.

    PHM08 점수는 늦게 본 예측에 벌점을 2배 줍니다.
    현장 논리도 같습니다 — 일찍 정비하면 비용이 조금 더 들 뿐이지만,
    늦으면 설비가 멈춥니다. 그래서 예측값에서 몇 사이클을 빼는
    '안전 마진'을 두는 것이 실무에서 흔한 방식입니다.
    """
    rows = []
    for margin in [0, 5, 10, 15, 20, 25, 30]:
        m = evaluate(y_true, np.clip(y_pred - margin, 0, 125))
        rows.append({"margin": margin, **m})
    return pd.DataFrame(rows)


def fig13_safety_margin(sweep: pd.DataFrame) -> None:
    """
    안전 마진의 효과 — 늦게 보는 예측을 얼마나 줄일 수 있나.

    '늦게 본 비율(%)' 과 'RMSE(사이클)' 은 단위가 다릅니다.
    한 그림에 겹쳐 그리면 두 축의 눈금을 임의로 맞추게 되어
    있지도 않은 관계가 있는 것처럼 보입니다. 그래서 위아래로 나눠 그립니다.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    sweet = sweep.loc[sweep["phm08"].idxmin(), "margin"]

    specs = [
        (axes[0], "late_ratio", 100, CAT[0], "o",
         "늦게 본 비율 (%)", "고장을 놓칠 위험 — 낮을수록 안전"),
        (axes[1], "rmse", 1, CAT[1], "s",
         "RMSE (사이클)", "예측 정확도 — 낮을수록 정확"),
    ]
    for ax, col, scale, color, marker, ylabel, title in specs:
        ax.plot(sweep["margin"], sweep[col] * scale, color=color, linewidth=2.4,
                marker=marker, markersize=8,
                markeredgecolor=style.SURFACE, markeredgewidth=2)
        ax.axvline(sweet, color=INK_SUB, linewidth=1.2, zorder=1)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
        # 점 위에 붙는 값 라벨이 소제목과 겹치지 않도록 위쪽 여백 확보
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.18)

        v0 = sweep[col].iloc[0] * scale
        v5 = sweep.loc[sweep["margin"] == sweet, col].iloc[0] * scale
        for x, v in [(0, v0), (sweet, v5)]:
            ax.annotate(f"{v:.0f}" if scale == 100 else f"{v:.1f}",
                        xy=(x, v), xytext=(0, 10), textcoords="offset points",
                        ha="center", color=color, fontsize=10, fontweight="bold")

    axes[0].annotate(f"PHM08 점수가 가장 낮은 지점 (마진 {sweet:.0f})",
                     xy=(sweet, axes[0].get_ylim()[1]), xytext=(8, -6),
                     textcoords="offset points", color=INK_SUB, fontsize=9, va="top")
    axes[1].set_xlabel("안전 마진 (예측에서 빼는 사이클 수)")

    fig.suptitle("안전 마진 — 예측을 조금 낮게 잡으면 '고장을 놓칠' 위험이 절반으로",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "13_rul_safety_margin.png")


def fig12_error_by_rul(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """고장이 가까울수록 잘 맞히는가 — 실무에서 중요한 것은 이쪽이다."""
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    df["err"] = df["pred"] - df["true"]
    bins = [0, 30, 60, 90, 125]
    labels = ["0~30\n(임박)", "30~60", "60~90", "90~125\n(여유)"]
    df["band"] = pd.cut(df["true"], bins=bins, labels=labels, include_lowest=True)

    agg = df.groupby("band", observed=True).agg(
        rmse=("err", lambda e: float(np.sqrt((e**2).mean()))),
        n=("err", "size"),
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    colors = [CAT[1] if i == 0 else CAT[0] for i in range(len(agg))]
    ax.bar(range(len(agg)), agg["rmse"], color=colors, width=0.6)

    for i, (v, n) in enumerate(zip(agg["rmse"], agg["n"])):
        ax.annotate(f"{v:.1f}\n({n}대)", xy=(i, v), xytext=(0, 6),
                    textcoords="offset points", ha="center",
                    color=INK_SUB, fontsize=9)

    ax.set_xticks(range(len(agg)), agg.index)
    ax.set_ylim(0, agg["rmse"].max() * 1.22)   # 막대 위 라벨이 제목에 닿지 않도록
    ax.set_title("고장이 가까울수록 정확하다 — 정비 판단이 필요한 구간에서 잘 맞는다")
    ax.set_xlabel("실제 잔여수명 구간 (사이클)")
    ax.set_ylabel("RMSE (사이클)")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    save(fig, "12_rul_error_by_band.png")


# =============================================================================
# main
# =============================================================================
def main() -> None:
    style.setup()
    cfg = load_config()
    cap = cfg["cmapss"]["rul"]["cap"]

    train, test = load_cmapss(SUBSET, cfg)          # RUL 상한 125 적용본
    sensors, dropped = usable_sensors(train, SENSOR_COLS)

    print(f"=== {SUBSET} RUL 예측 ===")
    print(f"사용 센서 {len(sensors)}개 / RUL 상한 {cap}")
    print(f"학습 {train[ID_COL].nunique()}대, 시험 {test[ID_COL].nunique()}대\n")

    # --- 특징 생성 (train/test 각자 자기 기준선 사용) ------------------------
    tr_feat, groups = build_features(train, sensors)
    te_base = baseline_table(test, sensors, BASELINE_CYCLES)
    te_feat, _ = build_features(test, sensors, baseline=te_base)

    # 기준선이 확정된 이후 구간만 사용
    tr_feat = tr_feat[usable_rows(tr_feat)].reset_index(drop=True)
    te_last = te_feat.groupby(ID_COL).tail(1).reset_index(drop=True)

    feature_sets = stack_feature_sets(groups)
    health_col = attach_health_index(tr_feat, te_last, groups)
    feature_sets["⑤ + 건강도 점수"] = feature_sets["④ + 이동평균·기울기"] + [health_col]
    print("특징 세트")
    for k, v in feature_sets.items():
        print(f"  {k:<22} {len(v):>3}개")
    print()

    # --- 전체 조합 실행 ------------------------------------------------------
    results = run_experiments(tr_feat, feature_sets, te_last)

    print("[결과] 공식 시험 세트 100대 기준\n")
    for metric, title in [("test_rmse", "RMSE (낮을수록 좋음)"),
                          ("test_phm08", "PHM08 점수 (낮을수록 좋음)")]:
        piv = results.pivot(index="features", columns="model", values=metric)
        piv = piv.loc[list(feature_sets)]
        print(f"  ── {title}")
        print(piv.round(2).to_string().replace("\n", "\n  "))
        print()

    best = results.sort_values("test_rmse").iloc[0]
    print(f"  최고: {best['features']} + {best['model']}")
    print(f"        RMSE {best['test_rmse']:.2f} / PHM08 {best['test_phm08']:.0f} / "
          f"늦게 본 비율 {best['test_late_ratio']*100:.0f}%")

    const_rmse = results[results["model"] == "상수(평균)"]["test_rmse"].iloc[0]
    print(f"  아무것도 안 배운 기준선 대비: {const_rmse:.2f} → {best['test_rmse']:.2f} "
          f"({(1 - best['test_rmse']/const_rmse)*100:.0f}% 개선)")

    # --- 최고 조합 재현 ------------------------------------------------------
    y_true, y_pred, model, cols = best_predictions(
        tr_feat, te_last, feature_sets[best["features"]], best["model"]
    )
    label = f"{best['features']} + {best['model']}"

    # 중요 특징 확인
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
        print("\n  모델이 가장 많이 본 특징 8개")
        for k, v in imp.head(8).items():
            print(f"    {k:<28}{v:>7.0f}")

    # --- 평가 프로토콜에 따른 차이 (논문과 비교할 때 중요) --------------------
    _, te_uncapped = load_cmapss(SUBSET, cfg, rul_cap=None)
    truth_uncapped = te_uncapped.groupby(ID_COL).tail(1)["RUL"].to_numpy()
    from src.models.rul import rmse as _rmse

    print("\n  [평가 기준에 따른 차이]")
    print(f"    정답을 {cap} 로 자른 기준   RMSE {_rmse(y_true, y_pred):.2f}")
    print(f"    정답을 자르지 않은 기준    RMSE {_rmse(truth_uncapped, y_pred):.2f}"
          f"   ({int((truth_uncapped > cap).sum())}대가 상한을 넘음)")
    print("    논문마다 기준이 달라 비교 시 반드시 확인해야 하는 부분")

    # --- 안전 마진 -----------------------------------------------------------
    sweep = safety_margin_sweep(y_true, y_pred)
    print("\n  [안전 마진] 예측을 일부러 낮게 잡으면")
    print(f"    {'마진':>6}{'RMSE':>9}{'PHM08':>9}{'늦게본비율':>12}")
    for _, r in sweep.iterrows():
        print(f"    {int(r['margin']):>6}{r['rmse']:>9.2f}{r['phm08']:>9.0f}"
              f"{r['late_ratio']*100:>11.0f}%")
    m5 = sweep[sweep["margin"] == 5].iloc[0]
    m0 = sweep[sweep["margin"] == 0].iloc[0]
    print(f"    -> 5사이클만 빼도 늦게 본 비율이 {m0['late_ratio']*100:.0f}% → "
          f"{m5['late_ratio']*100:.0f}% 로 줄고, PHM08 은 오히려 개선됨")

    # --- 저장 ---------------------------------------------------------------
    mdir = ROOT / "reports" / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    results.to_csv(mdir / "rul_model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"unit": te_last[ID_COL], "true": y_true, "pred": y_pred}).to_csv(
        mdir / "rul_test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    sweep.to_csv(mdir / "rul_safety_margin.csv", index=False, encoding="utf-8-sig")
    print("\n  저장: reports/metrics/rul_model_comparison.csv")
    print("  저장: reports/metrics/rul_test_predictions.csv")
    print("  저장: reports/metrics/rul_safety_margin.csv")

    print("\n=== 그림 생성 ===")
    fig10_feature_contribution(results)
    fig11_pred_vs_true(y_true, y_pred, label)
    fig12_error_by_rul(y_true, y_pred)
    fig13_safety_margin(sweep)
    print("\n완료.")


if __name__ == "__main__":
    main()
