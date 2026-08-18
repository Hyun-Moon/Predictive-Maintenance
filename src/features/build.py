"""
RUL 예측용 특징 생성
====================

1~2단계에서 배운 것을 특징(feature)으로 옮깁니다.

    EDA 에서 확인한 것              ->  만들 특징
    ------------------------------------------------------------
    개체차가 열화폭의 77%           ->  설비별 baseline 대비 편차
    노이즈가 열화폭의 12~16%        ->  이동평균 (구간을 뭉쳐서 봄)
    열화가 선형이 아니라 가속함      ->  기울기 (얼마나 빨리 나빠지고 있나)
    운전 시간 자체가 정보            ->  cycle

**인과성(causality) 규칙**: 모든 특징은 "그 시점까지의 과거"만으로 계산합니다.
    미래 값을 조금이라도 쓰면 검증 성능만 좋아지고 실제로는 전혀 안 맞습니다.
    (데이터 누수 leakage — 예측 모델에서 가장 흔하고 치명적인 실수)

    단 하나 예외적으로 조심할 것이 baseline 입니다.
    baseline 은 '설비 초기 30사이클'로 계산하므로,
    cycle 30 이전 시점에서 보면 미래를 조금 쓰는 셈입니다.
    그래서 학습·평가 모두 **cycle 30 이후 구간만** 사용합니다.
    실제 현장에서도 "설비를 30구간 돌려 기준선을 잡은 뒤부터 예측 시작"이 자연스럽습니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load import ID_COL, TIME_COL
from src.features.health import BASELINE_CYCLES, baseline_table

MA_WINDOWS = (10, 30)
SLOPE_WINDOW = 20


def _slope(s: pd.Series, window: int) -> pd.Series:
    """
    최근 window 구간 동안 값이 얼마나 빨리 변했는가 (1사이클당 변화량).

    정식 회귀 대신 (현재값 - window 전 값) / window 를 씁니다.
    계산이 훨씬 빠르고, 이동평균을 먼저 걸었기 때문에 노이즈에도 충분히 견딥니다.
    """
    return (s - s.shift(window)) / window


def build_features(
    df: pd.DataFrame,
    sensors: list[str],
    baseline: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    센서 원본에서 예측용 특징을 만든다.

    Parameters
    ----------
    baseline : 미리 계산해 둔 설비별 기준선. None 이면 이 df 로 직접 계산.
               (train 과 test 는 서로 다른 설비이므로 각자 자기 기준선을 씁니다)

    Returns
    -------
    (특징이 추가된 DataFrame, 특징 그룹별 컬럼 이름 사전)
        그룹을 나눠 돌려주는 이유: "센서 원본만", "거기에 편차 추가", ...
        이런 식으로 특징을 하나씩 쌓아가며 기여도를 측정하기 위해서입니다.
    """
    out = df.copy().sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
    groups: dict[str, list[str]] = {}

    # --- (1) 운전 시간 -------------------------------------------------------
    groups["cycle"] = [TIME_COL]

    # --- (2) 센서 원본 -------------------------------------------------------
    groups["raw"] = list(sensors)

    # --- (3) 설비별 기준선 대비 편차 ----------------------------------------
    if baseline is None:
        baseline = baseline_table(out, sensors, BASELINE_CYCLES)
    dev_cols = []
    for c in sensors:
        out[f"{c}_dev"] = out[c] - out[ID_COL].map(baseline[(c, "mean")])
        dev_cols.append(f"{c}_dev")
    groups["dev"] = dev_cols

    # --- (4) 이동평균 + 기울기 ----------------------------------------------
    roll_cols = []
    g = out.groupby(ID_COL)
    for c in dev_cols:
        for w in MA_WINDOWS:
            name = f"{c}_ma{w}"
            out[name] = g[c].transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
            roll_cols.append(name)

        name = f"{c}_slope{SLOPE_WINDOW}"
        smooth = out[f"{c}_ma{MA_WINDOWS[0]}"]
        out[name] = smooth.groupby(out[ID_COL]).transform(
            lambda s: _slope(s, SLOPE_WINDOW)
        ).fillna(0.0)
        roll_cols.append(name)
    groups["rolling"] = roll_cols

    return out, groups


def stack_feature_sets(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    특징을 하나씩 쌓아가는 실험 세트를 만든다.

    이렇게 누적해서 비교하면 "무엇을 추가했더니 얼마나 좋아졌는지"가 드러납니다.
    한 번에 다 넣고 좋은 점수만 보고하면, 실제로 무엇이 기여했는지 알 수 없습니다.
    """
    sets: dict[str, list[str]] = {}
    sets["① 운전시간만"] = list(groups["cycle"])
    sets["② + 센서 원본"] = sets["① 운전시간만"] + groups["raw"]
    sets["③ + 개체 baseline 편차"] = sets["② + 센서 원본"] + groups["dev"]
    sets["④ + 이동평균·기울기"] = sets["③ + 개체 baseline 편차"] + groups["rolling"]
    return sets


def usable_rows(df: pd.DataFrame, min_cycle: int = BASELINE_CYCLES) -> pd.Series:
    """기준선이 확정된 이후 구간만 사용 (위 인과성 규칙 참고)."""
    return df[TIME_COL] > min_cycle
