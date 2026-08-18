"""
데이터 로더 스모크 테스트
========================

"테스트"라고 하면 거창해 보이지만, 여기서 하는 일은 단순합니다:
    데이터를 읽었을 때 당연히 성립해야 할 사실들을 코드로 박아두는 것.

이걸 해두면, 나중에 전처리 코드를 고치다가 실수로 데이터를 망가뜨렸을 때
    pytest -q
한 줄로 즉시 알아챌 수 있습니다.

실행:  pytest -q
"""

from __future__ import annotations

import pytest

from src.config import load_config
from src.data.load import (
    ALL_COLS,
    ID_COL,
    SENSOR_COLS,
    TIME_COL,
    cmapss_dir,
    load_cmapss,
)

# 데이터가 아직 없으면 테스트 전체를 건너뛴다.
# (저장소를 clone 한 직후 pytest 가 빨갛게 뜨는 것을 방지)
pytestmark = pytest.mark.skipif(
    not (cmapss_dir() / "train_FD001.txt").exists(),
    reason="C-MAPSS 데이터 없음. python -m src.data.download --dataset cmapss 를 먼저 실행하세요.",
)

# 각 하위 세트의 (train 개체 수, test 개체 수).
#
# 주의: NASA 공식 readme.txt 에는 FD004 가 train 248 / test 249 로 적혀 있으나,
#       실제 파일은 train 249 / test 248 입니다.
#       RUL_FD004.txt 의 행 수(248)가 test 개체 수와 일치하는 것으로 교차 확인했습니다.
#       -> 공식 문서를 그대로 믿지 말고 데이터로 검증해야 한다는 좋은 예시.
EXPECTED_UNITS = {
    "FD001": (100, 100),
    "FD002": (260, 259),
    "FD003": (100, 100),
    "FD004": (249, 248),
}


@pytest.mark.parametrize("subset", list(EXPECTED_UNITS))
def test_shape_and_columns(subset: str) -> None:
    """컬럼이 제대로 붙었고 설비 개체 수가 공식 문서와 맞는가."""
    train, test = load_cmapss(subset)

    assert list(train.columns) == [*ALL_COLS, "RUL"]
    assert list(test.columns) == [*ALL_COLS, "RUL"]
    assert len(SENSOR_COLS) == 21

    n_train, n_test = EXPECTED_UNITS[subset]
    assert train[ID_COL].nunique() == n_train
    assert test[ID_COL].nunique() == n_test


@pytest.mark.parametrize("subset", list(EXPECTED_UNITS))
def test_no_missing_values(subset: str) -> None:
    """C-MAPSS 는 결측치가 없는 시뮬레이션 데이터. 결측이 생겼다면 파싱이 잘못된 것."""
    train, test = load_cmapss(subset)
    assert train.isna().sum().sum() == 0
    assert test.isna().sum().sum() == 0


def test_cycle_is_monotonic_within_unit() -> None:
    """각 설비의 cycle 은 1, 2, 3... 으로 끊김 없이 증가해야 한다."""
    train, _ = load_cmapss("FD001")
    for unit, g in train.groupby(ID_COL):
        cycles = g[TIME_COL].to_numpy()
        assert cycles[0] == 1, f"unit {unit}: cycle 이 1부터 시작하지 않음"
        assert (cycles[1:] - cycles[:-1] == 1).all(), f"unit {unit}: cycle 에 구멍이 있음"


def test_train_rul_reaches_zero() -> None:
    """
    train 은 run-to-failure(고장까지 기록) 데이터이므로,
    모든 설비의 마지막 시점 RUL 은 정확히 0 이어야 한다.
    0 이 아니면 RUL 계산식이 틀린 것.
    """
    train, _ = load_cmapss("FD001")
    assert (train.groupby(ID_COL)["RUL"].min() == 0).all()


def test_test_rul_never_reaches_zero() -> None:
    """
    반대로 test 는 고장 '전에' 기록이 끊긴 데이터이므로,
    마지막 시점 RUL 이 0 이면 안 된다. (그게 바로 우리가 맞춰야 할 값)
    """
    _, test = load_cmapss("FD001", rul_cap=None)
    assert (test.groupby(ID_COL)["RUL"].min() > 0).all()


def test_rul_cap_applied() -> None:
    """config.yaml 의 RUL 상한(기본 125)이 실제로 적용되는가."""
    cap = load_config()["cmapss"]["rul"]["cap"]
    train, test = load_cmapss("FD001")
    assert train["RUL"].max() == cap
    assert test["RUL"].max() <= cap

    # 상한을 끄면 125 보다 큰 값이 나와야 한다
    train_uncapped, _ = load_cmapss("FD001", rul_cap=None)
    assert train_uncapped["RUL"].max() > cap


def test_rul_decreases_by_one_each_cycle() -> None:
    """
    상한이 걸리지 않은 구간에서는 cycle 이 1 늘 때 RUL 이 정확히 1 줄어야 한다.
    RUL 라벨이 시간축과 어긋나지 않았는지 확인하는 테스트.
    """
    train, _ = load_cmapss("FD001", rul_cap=None)
    g = train[train[ID_COL] == 1].sort_values(TIME_COL)
    assert list(g["RUL"].diff().dropna().unique()) == [-1]
