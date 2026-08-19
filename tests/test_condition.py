"""
운전조건 처리 테스트
===================

4단계에서 겪은 문제들이 다시 생기지 않도록 못박아 둡니다.

    4-3. 기준선 창이 관측보다 길어 편차가 항상 0 이 되던 문제
    4-4. 조건 효과가 섞인 자로 재려다 값이 폭발하던 문제

실행:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.load import ID_COL, OP_COLS, TIME_COL
from src.features.condition import (
    ConditionNormalizer,
    adaptive_baseline_mask,
    condition_key,
)


@pytest.fixture
def two_condition_data() -> tuple[pd.DataFrame, list[str]]:
    """
    설비 2대 x 40사이클. 조건 두 가지가 번갈아 나온다.

    센서 s1: 조건에 따라 100 vs 200 (조건 효과 100)
             + 시간에 따라 서서히 상승 (열화 효과 최대 4)
    -> 조건 효과가 열화의 25배. FD002 상황을 축소한 것.
    """
    rows = []
    for unit in (1, 2):
        for cyc in range(1, 41):
            cond = cyc % 2                       # 0, 1 번갈아
            rows.append(
                {
                    ID_COL: unit,
                    TIME_COL: cyc,
                    OP_COLS[0]: 0.0 if cond == 0 else 42.0,
                    OP_COLS[1]: 0.0,
                    OP_COLS[2]: 100.0,
                    "s1": (100.0 if cond == 0 else 200.0) + cyc * 0.1 + unit,
                }
            )
    return pd.DataFrame(rows), ["s1"]


# =============================================================================
# 조건 식별
# =============================================================================
def test_condition_key_separates_conditions(two_condition_data) -> None:
    df, _ = two_condition_data
    key = condition_key(df)
    assert key.nunique() == 2


def test_condition_key_is_stable_for_same_settings() -> None:
    """같은 운전 설정이면 항상 같은 키가 나와야 한다 (학습/시험 간 대응을 위해)."""
    a = pd.DataFrame({OP_COLS[0]: [42.003], OP_COLS[1]: [0.840], OP_COLS[2]: [100.0]})
    b = pd.DataFrame({OP_COLS[0]: [41.998], OP_COLS[1]: [0.841], OP_COLS[2]: [100.0]})
    assert condition_key(a).iloc[0] == condition_key(b).iloc[0]


# =============================================================================
# 조건별 정규화
# =============================================================================
def test_normalizer_removes_condition_effect(two_condition_data) -> None:
    """
    정규화 뒤에는 조건별 평균 차이가 사라져야 한다.
    이것이 이 모듈의 존재 이유 전부다.
    """
    df, sensors = two_condition_data
    key = condition_key(df)

    before = df.groupby(key)["s1"].mean()
    assert abs(before.iloc[1] - before.iloc[0]) > 90      # 정규화 전: 조건 차이 100

    z = ConditionNormalizer().fit(df, sensors).transform(df)
    after = z.groupby(key.to_numpy())["s1"].mean()
    assert abs(after.iloc[1] - after.iloc[0]) < 0.2       # 정규화 후: 거의 0


def test_normalizer_keeps_degradation_signal(two_condition_data) -> None:
    """조건 효과만 지우고 시간에 따른 열화 추세는 남아 있어야 한다."""
    df, sensors = two_condition_data
    z = ConditionNormalizer().fit(df, sensors).transform(df)
    early = z.loc[df[TIME_COL] <= 10, "s1"].mean()
    late = z.loc[df[TIME_COL] >= 31, "s1"].mean()
    assert late > early


def test_normalizer_handles_unseen_condition(two_condition_data) -> None:
    """
    학습에 없던 운전조건이 나와도 죽지 않아야 한다.
    (실제 운영에서 새 운전 모드가 생기는 일은 흔하다)
    """
    df, sensors = two_condition_data
    norm = ConditionNormalizer().fit(df, sensors)

    unseen = df.head(3).copy()
    unseen[OP_COLS[0]] = 77.0                     # 본 적 없는 조건
    out = norm.transform(unseen)
    assert np.isfinite(out.to_numpy()).all()


def test_normalizer_output_keeps_shape(two_condition_data) -> None:
    df, sensors = two_condition_data
    out = ConditionNormalizer().fit(df, sensors).transform(df)
    assert list(out.columns) == sensors
    assert len(out) == len(df)
    assert (out.index == df.index).all()


# =============================================================================
# 적응형 기준선
# =============================================================================
def test_adaptive_baseline_leaves_rows_for_short_units() -> None:
    """
    4-3 에서 겪은 문제를 막는 테스트.

    관측이 21사이클뿐인 설비도 기준선 이후 구간이 반드시 남아야 한다.
    (고정 30사이클이면 관측 전체가 기준선이 되어 편차가 항상 0 이 된다)
    """
    rows = []
    for unit, length in [(1, 21), (2, 300)]:
        rows += [{ID_COL: unit, TIME_COL: c} for c in range(1, length + 1)]
    df = pd.DataFrame(rows)

    mask = adaptive_baseline_mask(df)
    for unit in (1, 2):
        g = mask[df[ID_COL] == unit]
        assert g.sum() >= 5, "기준선 구간이 너무 짧음"
        assert (~g).sum() > 0, "예측할 구간이 남지 않음"


def test_adaptive_baseline_respects_bounds() -> None:
    """긴 설비는 상한(30)에서 멈추고, 짧은 설비는 하한(5) 아래로 안 내려간다."""
    long_df = pd.DataFrame({ID_COL: 1, TIME_COL: range(1, 501)})
    assert adaptive_baseline_mask(long_df).sum() == 30

    short_df = pd.DataFrame({ID_COL: 1, TIME_COL: range(1, 11)})
    assert adaptive_baseline_mask(short_df).sum() == 5


def test_adaptive_baseline_is_per_unit() -> None:
    """설비마다 길이가 다르면 기준선 길이도 달라야 한다."""
    rows = []
    for unit, length in [(1, 20), (2, 200)]:
        rows += [{ID_COL: unit, TIME_COL: c} for c in range(1, length + 1)]
    df = pd.DataFrame(rows)
    mask = adaptive_baseline_mask(df)
    assert mask[df[ID_COL] == 1].sum() < mask[df[ID_COL] == 2].sum()
