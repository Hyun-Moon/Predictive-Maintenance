"""
5단계 — 실제 설비에 적용하기
===========================

실행:  python scripts/05_field_application.py

앞 4단계는 전부 C-MAPSS(항공기 엔진)였습니다.
이 단계에서 확인하는 것은 하나입니다.

    **코드를 한 줄도 고치지 않고, 완전히 다른 설비 데이터에서 돌아가는가?**

넣는 데이터는 공조기(AHU)입니다. 항공기 엔진과 공통점이 없습니다.
    - 컬럼 이름이 한글이고 다릅니다 (설비번호 / 일시 / 필터차압 / 팬전류 ...)
    - 센서 종류가 다릅니다
    - 시간 단위가 다릅니다 (운전 사이클 -> 1시간 간격)
    - 운전조건이 연속값입니다 (외기온·부하율)

**중요 — 이 데모의 데이터는 제가 만든 모의 데이터입니다.**
    성능에 대한 근거가 아닙니다. 제가 넣은 규칙을 제가 찾는 것이므로 잘 맞는 게 당연합니다.
    성능 근거는 오직 1~4단계의 C-MAPSS 실험입니다.
    이 단계가 증명하는 것은 **이식 가능성**뿐입니다.

산출물: reports/figures/18~19*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ROOT
from src.deploy.demo_data import (
    CONDITION_COLS,
    ID_COL,
    SENSOR_COLS,
    TIME_COL,
    make_ahu_data,
)
from src.deploy.pipeline import DetectorConfig, EquipmentHealthMonitor, EquipmentSpec
from src.viz import style
from src.viz.style import CAT, INK_SUB, MUTED, save


def fig18_ahu_health(scored: pd.DataFrame, truth: pd.DataFrame, cfg: DetectorConfig) -> None:
    """
    공조기 건강도 궤적과 실제 막힘 정도.

    시간 단위 2년치는 설비당 17,520점이라 그대로 그리면 시커먼 띠가 됩니다.
    **일 단위 중앙값으로 줄여서** 그립니다.
    (점을 다 찍는다고 정보가 더 전달되지는 않습니다)
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    daily = (
        scored.set_index(TIME_COL)
        .groupby(ID_COL)["health"]
        .resample("1D").median()
        .reset_index()
    )
    daily_truth = (
        truth.set_index(TIME_COL)
        .groupby(ID_COL)["_막힘정도"]
        .resample("1D").mean()
        .reset_index()
    )

    units = sorted(daily[ID_COL].unique())
    for u in units:
        g = daily[daily[ID_COL] == u]
        axes[0].plot(g[TIME_COL], g["health"], color=MUTED, linewidth=1.0, alpha=0.7)
        t = daily_truth[daily_truth[ID_COL] == u]
        axes[1].plot(t[TIME_COL], t["_막힘정도"], color=MUTED, linewidth=1.0, alpha=0.7)

    lead = units[0]
    g = daily[daily[ID_COL] == lead]
    t = daily_truth[daily_truth[ID_COL] == lead]
    axes[0].plot(g[TIME_COL], g["health"], color=CAT[0], linewidth=2.4, zorder=5)
    axes[1].plot(t[TIME_COL], t["_막힘정도"], color=CAT[0], linewidth=2.4, zorder=5)

    first_alarm = scored.loc[(scored[ID_COL] == lead) & scored["alarm"], TIME_COL]
    if len(first_alarm):
        axes[0].axvline(first_alarm.iloc[0], color=CAT[1], linewidth=1.6)
        axes[0].annotate(f"{lead} 경보  {first_alarm.iloc[0]:%Y-%m-%d}",
                         xy=(first_alarm.iloc[0], 97), xytext=(8, 0),
                         textcoords="offset points", color=CAT[1],
                         fontsize=9, fontweight="bold", va="top")

    # 정상 구간(1년차)을 음영으로 표시
    split = scored[TIME_COL].min() + pd.Timedelta(days=365)
    for ax in axes:
        ax.axvspan(scored[TIME_COL].min(), split, color=CAT[2], alpha=0.07, zorder=0)

    axes[0].axhline(cfg.alarm_health, color=CAT[1], linewidth=1.2, zorder=1)
    axes[0].annotate(f"경보선 {cfg.alarm_health:.0f}", xy=(scored[TIME_COL].min(), cfg.alarm_health),
                     xytext=(6, 6), textcoords="offset points",
                     color=CAT[1], fontsize=9)
    axes[0].set_ylabel("건강도 (일 중앙값)")
    axes[0].set_ylim(0, 108)
    axes[0].set_title(f"건강도 — 굵은 선은 {lead}", fontsize=10, loc="left")

    axes[1].set_ylabel("실제 필터 막힘 정도")
    axes[1].set_title("정답 (모델에는 주지 않은 값)", fontsize=10, loc="left")
    axes[1].set_xlabel("일시")

    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)

    fig.suptitle("공조기 8대에 그대로 적용 — 코드 수정 없음",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005,
             "연한 음영 = 1년차 정상 구간(기준선 학습에 사용) · "
             "모의 데이터이며 성능 근거가 아니라 이식 가능성 확인용입니다.",
             ha="center", color=INK_SUB, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    save(fig, "18_ahu_health_trajectory.png")


def fig19_lead_time(lead_days: pd.Series) -> None:
    """경보가 막힘 시작 후 얼마나 빨리 울렸는가."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh(range(len(lead_days)), lead_days.to_numpy(), color=CAT[0], height=0.6)
    ax.set_yticks(range(len(lead_days)), lead_days.index)

    for i, v in enumerate(lead_days):
        ax.annotate(f"{v:.0f}일", xy=(v, i), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK_SUB, fontsize=9)

    ax.set_xlim(0, lead_days.max() * 1.25)
    ax.set_title("막힘이 시작되고 며칠 만에 경보가 울렸는가")
    ax.set_xlabel("막힘 시작 → 경보까지 (일)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    save(fig, "19_ahu_alarm_leadtime.png")


def main() -> None:
    style.setup()

    print("=== 5단계: 실제 설비 적용 ===\n")
    print("데이터: 공조기(AHU) 8대, 2년치 1시간 간격 — 항공기 엔진과 공통점 없음")
    print("  컬럼 이름, 센서 종류, 시간 단위가 전부 다름")
    print("  1년차는 정상(기준선), 2년차부터 필터가 서서히 막힘\n")

    # 2년치: 1년차는 정상(사계절 전체가 기준선), 2년차부터 필터가 막혀감
    data = make_ahu_data(n_units=8, years=2, healthy_years=1.0)
    truth = data[[ID_COL, TIME_COL, "_막힘정도"]].copy()
    data = data.drop(columns=["_막힘정도"])          # 정답은 모델에 주지 않음

    spec = EquipmentSpec(
        id_col=ID_COL,
        time_col=TIME_COL,
        sensor_cols=SENSOR_COLS,
        condition_cols=CONDITION_COLS,
    )
    # 시간 단위 데이터에 맞춘 설정
    #   평활 창 24시간(일주기 1회분), 지속 조건 72시간(평활 창의 3배 = 3일 연속)
    #   지속 조건을 24로 두면 정상 구간에서도 경보가 납니다 (아래 문서 참고)
    cfg = DetectorConfig(smoothing_window=24, alarm_persistence=72)

    monitor = EquipmentHealthMonitor(spec, cfg)

    # --- 1. 데이터 점검 -------------------------------------------------------
    check = monitor.check_data(data)
    print("[1] 데이터 점검 — 이 데이터로 예지보전이 가능한가")
    for k, v in check["summary"].items():
        print(f"    {k:<20} {v}")
    for w in check["warnings"]:
        print(f"    (주의) {w}")
    print(f"    판정: {'사용 가능' if check['ok'] else '사용 불가'}")
    if not check["ok"]:
        for i in check["issues"]:
            print(f"      - {i}")
        return

    # --- 2. 학습 (고장 라벨 없음) --------------------------------------------
    print("\n[2] 학습 — 각 설비의 초기 구간만 사용, 고장 이력 불필요")
    monitor.fit(data)
    print(f"    사용 센서 {len(monitor.sensors_)}개: {monitor.sensors_}")
    print(f"    운전조건 구간 {len(monitor._cond_mean_)}개로 분할 (외기온 x 부하율)")

    # --- 3. 채점 --------------------------------------------------------------
    scored = monitor.score(data)
    merged = scored.merge(truth, on=[ID_COL, TIME_COL])

    print("\n[3] 결과")
    rows = []
    for u, g in merged.groupby(ID_COL):
        clog_start = g.loc[g["_막힘정도"] > 1, TIME_COL]
        alarm = g.loc[g["alarm"], TIME_COL]
        if len(clog_start) and len(alarm):
            lead = (alarm.iloc[0] - clog_start.iloc[0]).total_seconds() / 86400
            clog_at_alarm = g.loc[g[TIME_COL] == alarm.iloc[0], "_막힘정도"].iloc[0]
            rows.append({"설비": u, "경보까지(일)": lead,
                         "경보 시점 막힘": clog_at_alarm,
                         "최종 막힘": g["_막힘정도"].max()})
    res = pd.DataFrame(rows).set_index("설비")

    print(f"    {'설비':<10}{'막힘시작→경보':>14}{'경보시점 막힘':>14}{'최종 막힘':>12}")
    for u, r in res.iterrows():
        print(f"    {u:<10}{r['경보까지(일)']:>12.0f}일"
              f"{r['경보 시점 막힘']:>13.0f}{r['최종 막힘']:>12.0f}")

    print(f"\n    경보가 울린 설비: {len(res)}/{data[ID_COL].nunique()}대")
    print(f"    막힘이 최종 수준의 {(res['경보 시점 막힘']/res['최종 막힘']).mean()*100:.0f}% "
          f"진행됐을 때 평균적으로 경보")

    # --- 4. 원인 설명 ---------------------------------------------------------
    print("\n[4] 경보 원인 설명 — 현장에 전달할 내용")
    # 설명은 전체 데이터를 넘기고 대상 행만 지정한다 (일부만 넘기면 편차가 0 이 됨)
    sample = merged[merged["alarm"]].groupby(ID_COL).head(1).head(3)
    why = monitor.explain(data, targets=sample.index)
    for idx, r in why.iterrows():
        u = data.loc[idx, ID_COL]
        print(f"    {u}: {r['주요 원인']}")
    print("    → '점수가 낮다'가 아니라 '어느 센서가 얼마나 벗어났다'까지 나와야")
    print("      정비원이 확인하러 갑니다.")

    print("\n=== 그림 생성 ===")
    fig18_ahu_health(scored, truth, cfg)
    fig19_lead_time(res["경보까지(일)"])
    print("\n완료.")
    print("\n※ 이 데이터는 모의 데이터입니다. 성능 근거가 아니라 이식 가능성 확인용입니다.")


if __name__ == "__main__":
    main()
