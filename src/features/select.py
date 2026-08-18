"""
쓸모없는 컬럼 골라내기
=====================

센서를 달아놨다고 다 정보가 있는 것은 아닙니다.
값이 전혀 안 변하는 센서는 모델에 넣어봐야 계산만 늘리고 도움이 안 됩니다.
(실제 현장에서도 "달려는 있는데 값이 안 들어오는" 포인트가 흔합니다)
"""

from __future__ import annotations

import pandas as pd

# 부동소수점 오차를 감안한 기본 허용치.
#
# 왜 0 이 아닌가?
#   컴퓨터는 소수를 2진수로 근사해서 저장합니다.
#   값이 전부 14.62 로 똑같은 컬럼이라도, 2만 개를 더하고 나누는 과정에서
#   미세한 오차가 쌓여 표준편차가 정확히 0 이 아니라 5.3e-15 같은 값이 나옵니다.
#   `std() == 0` 으로 검사하면 이런 컬럼을 놓칩니다. (실제로 겪은 문제)
#   센서 값의 실제 크기에 비해 1e-9 는 사실상 0 이므로 이 정도를 기준으로 씁니다.
DEFAULT_TOL = 1e-9


def constant_columns(df: pd.DataFrame, cols: list[str], tol: float = DEFAULT_TOL) -> list[str]:
    """표준편차가 tol 이하인 컬럼(= 값이 사실상 변하지 않는 컬럼) 목록을 반환."""
    return [c for c in cols if df[c].std() <= tol]


def low_variation_columns(df: pd.DataFrame, cols: list[str], max_unique: int = 2) -> list[str]:
    """
    고유값이 max_unique 개 이하인 컬럼 목록.

    표준편차만 보면 놓치는 경우가 있습니다.
    예: 값이 21.60 과 21.61 두 개뿐이면 표준편차는 0 이 아니지만(1.4e-3),
    사실상 정보가 없는 컬럼입니다.
    """
    return [c for c in cols if df[c].nunique() <= max_unique]


def usable_sensors(
    df: pd.DataFrame, cols: list[str], max_unique: int = 2
) -> tuple[list[str], list[str]]:
    """
    (쓸 컬럼, 버릴 컬럼) 을 나눠서 돌려준다.

    버리는 기준: 상수이거나, 고유값이 max_unique 개 이하.
    """
    drop = set(constant_columns(df, cols)) | set(low_variation_columns(df, cols, max_unique))
    keep = [c for c in cols if c not in drop]
    return keep, [c for c in cols if c in drop]
