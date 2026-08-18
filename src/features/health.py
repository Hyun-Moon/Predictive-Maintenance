"""
건강도 점수용 특징 만들기
========================

EDA 에서 확인한 두 가지 사실을 코드로 옮긴 모듈입니다.

  1. 설비마다 정상 상태의 출발점이 다르다 (개체차가 열화폭의 77%)
     -> 절대값이 아니라 "자기 자신의 초기 정상 대비 얼마나 벗어났는가"로 봐야 한다.

  2. 한 시점의 관측치에는 노이즈가 섞여 있다 (열화폭의 12~16%)
     -> 최근 구간의 평균/기울기를 같이 보면 판단이 안정된다.

중요 — 이 모듈은 RUL(정답)을 전혀 쓰지 않습니다.
    실제 회사 설비에는 고장 이력이 없기 때문에,
    "정상 데이터만으로 만들 수 있는 특징"만 사용해야 현장에 가져갈 수 있습니다.
    여기서 정상 구간은 "설비 도입 초기 N 사이클" 로만 정의합니다.
"""

from __future__ import annotations

import pandas as pd

from src.data.load import ID_COL, TIME_COL

# 설비별 '정상 기준선'으로 삼을 초기 사이클 수.
# 실무로 치면 "신규 설비 시운전 직후 안정화된 구간" 입니다.
BASELINE_CYCLES = 30


def baseline_table(df: pd.DataFrame, cols: list[str], n_cycles: int = BASELINE_CYCLES) -> pd.DataFrame:
    """
    설비별 초기 구간의 평균과 표준편차 표를 만든다.

    이것이 "이 설비의 정상 상태는 원래 이 정도였다"는 기준이 됩니다.
    """
    early = df[df[TIME_COL] <= n_cycles]
    agg = early.groupby(ID_COL)[cols].agg(["mean", "std"])
    return agg


def add_deviation(
    df: pd.DataFrame, cols: list[str], n_cycles: int = BASELINE_CYCLES
) -> tuple[pd.DataFrame, list[str]]:
    """
    각 센서를 '자기 설비의 초기값 대비 편차'로 바꾼 컬럼을 추가한다.

    편차 = (현재값 - 그 설비의 초기 평균)

    이렇게 하면 "원래 높게 출발한 설비"와 "열화되어 높아진 설비"를 구분할 수 있습니다.
    """
    base = baseline_table(df, cols, n_cycles)
    out = df.copy()
    dev_cols = []
    for c in cols:
        mean_map = base[(c, "mean")]
        out[f"{c}_dev"] = out[c] - out[ID_COL].map(mean_map)
        dev_cols.append(f"{c}_dev")
    return out, dev_cols


def add_rolling(
    df: pd.DataFrame, cols: list[str], window: int = 10
) -> tuple[pd.DataFrame, list[str]]:
    """
    설비별 이동평균 컬럼을 추가한다.

    주의: 반드시 설비 단위(groupby)로 계산해야 합니다.
    전체를 한 줄로 굴리면 A설비 마지막 값과 B설비 첫 값이 섞입니다.

    min_periods=1 : 초반에 데이터가 부족해도 있는 만큼으로 계산 (NaN 방지)
    """
    out = df.copy()
    roll_cols = []
    for c in cols:
        out[f"{c}_ma{window}"] = out.groupby(ID_COL)[c].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        roll_cols.append(f"{c}_ma{window}")
    return out, roll_cols


def normal_mask(df: pd.DataFrame, n_cycles: int = BASELINE_CYCLES) -> pd.Series:
    """
    '정상으로 간주할 구간' 마스크.

    라벨을 쓰지 않고 운전 시간만으로 정의합니다.
    -> 고장 이력이 없는 실제 설비에서도 똑같이 만들 수 있습니다.
    """
    return df[TIME_COL] <= n_cycles
