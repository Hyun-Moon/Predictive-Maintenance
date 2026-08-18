"""
RUL 예측 모듈 테스트
===================

성능이 아니라 **동작의 정확성**을 검증합니다.
특히 이 단계에서 가장 위험한 두 가지를 테스트로 막아 둡니다.

    1. 데이터 누수 — 특징이 미래 값을 참조하면 안 된다
    2. 설비 단위 분할 — 같은 설비가 학습과 검증에 동시에 들어가면 안 된다

실행:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.load import ID_COL, TIME_COL
from src.features.build import MA_WINDOWS, SLOPE_WINDOW, build_features, stack_feature_sets
from src.models.rul import (
    ConstantModel,
    evaluate,
    group_split,
    late_ratio,
    phm08_score,
    rmse,
)


@pytest.fixture
def toy() -> tuple[pd.DataFrame, list[str]]:
    """설비 3대 x 60사이클. 센서 2개는 선형 상승, 1개는 상수."""
    rows = []
    for unit in (1, 2, 3):
        for cyc in range(1, 61):
            rows.append(
                {
                    ID_COL: unit,
                    TIME_COL: cyc,
                    "RUL": 60 - cyc,
                    "s1": 100.0 + unit * 10 + cyc * 0.5,   # 설비마다 출발점이 다름
                    "s2": 50.0 - cyc * 0.2,
                    "s3": 7.0,
                }
            )
    return pd.DataFrame(rows), ["s1", "s2", "s3"]


# =============================================================================
# 특징 생성
# =============================================================================
def test_baseline_deviation_removes_unit_offset(toy) -> None:
    """
    설비마다 출발점이 다른 s1 을 편차로 바꾸면,
    같은 사이클에서 세 설비의 값이 같아져야 한다. (개체차 제거의 핵심)
    """
    df, sensors = toy
    feat, _ = build_features(df, sensors)

    at50 = feat[feat[TIME_COL] == 50]
    assert at50["s1"].std() > 1.0                    # 원본은 설비마다 다름
    assert at50["s1_dev"].std() == pytest.approx(0.0, abs=1e-9)  # 편차는 같아짐


def test_features_do_not_use_future(toy) -> None:
    """
    누수 방지 테스트 — 가장 중요한 테스트.

    미래 구간의 값을 크게 바꿔도, 과거 시점의 특징은 변하지 않아야 한다.
    (rolling 이나 shift 방향을 실수로 뒤집으면 여기서 걸린다)
    """
    df, sensors = toy
    base, groups = build_features(df, sensors)

    tampered = df.copy()
    future = (tampered[ID_COL] == 1) & (tampered[TIME_COL] > 45)
    tampered.loc[future, "s1"] += 999.0
    after, _ = build_features(tampered, sensors)

    cols = groups["dev"] + groups["rolling"]
    past = (base[ID_COL] == 1) & (base[TIME_COL] <= 45)
    pd.testing.assert_frame_equal(
        base.loc[past, cols].reset_index(drop=True),
        after.loc[past.to_numpy(), cols].reset_index(drop=True),
    )


def test_rolling_does_not_cross_units(toy) -> None:
    """
    이동평균이 설비 경계를 넘으면 안 된다.
    설비의 첫 사이클 이동평균은 그 설비 자신의 값이어야 한다.
    """
    df, sensors = toy
    feat, _ = build_features(df, sensors)
    first = feat[feat[TIME_COL] == 1]
    col = f"s1_dev_ma{MA_WINDOWS[0]}"
    assert np.allclose(first[col].to_numpy(), first["s1_dev"].to_numpy())


def test_slope_detects_direction(toy) -> None:
    """상승하는 센서는 기울기가 양수, 하락하는 센서는 음수여야 한다."""
    df, sensors = toy
    feat, _ = build_features(df, sensors)
    late = feat[feat[TIME_COL] > SLOPE_WINDOW + MA_WINDOWS[0]]
    assert late[f"s1_dev_slope{SLOPE_WINDOW}"].mean() > 0
    assert late[f"s2_dev_slope{SLOPE_WINDOW}"].mean() < 0


def test_feature_sets_are_cumulative(toy) -> None:
    """누적 비교를 위해, 뒤 세트는 앞 세트를 전부 포함해야 한다."""
    df, sensors = toy
    _, groups = build_features(df, sensors)
    sets = list(stack_feature_sets(groups).values())
    for smaller, larger in zip(sets, sets[1:]):
        assert set(smaller).issubset(set(larger))
        assert len(larger) > len(smaller)


# =============================================================================
# 분할
# =============================================================================
def test_group_split_keeps_units_separate() -> None:
    """같은 설비가 학습과 검증에 동시에 들어가면 안 된다. (성능 부풀리기 방지)"""
    units = np.repeat(np.arange(50), 20)
    fit, val = group_split(units, val_ratio=0.2, seed=0)
    assert set(fit).isdisjoint(set(val))
    assert len(fit) + len(val) == 50
    assert len(val) == 10


def test_group_split_is_reproducible() -> None:
    """같은 seed 면 같은 분할. (실험을 다시 돌렸을 때 결과가 흔들리면 안 됨)"""
    units = np.repeat(np.arange(30), 5)
    a = group_split(units, seed=7)
    b = group_split(units, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# =============================================================================
# 평가 지표
# =============================================================================
def test_phm08_penalises_late_more_than_early() -> None:
    """
    같은 크기로 틀려도 '늦게 본 쪽'의 벌점이 더 커야 한다.
    이 비대칭이 PHM08 을 쓰는 이유 전부다.
    """
    truth = np.array([100.0])
    early = phm08_score(truth, np.array([70.0]))   # 30 일찍
    late = phm08_score(truth, np.array([130.0]))   # 30 늦게
    assert late > early * 1.5


def test_phm08_is_zero_for_perfect_prediction() -> None:
    truth = np.array([50.0, 80.0, 10.0])
    assert phm08_score(truth, truth) == pytest.approx(0.0)


def test_rmse_matches_manual_calculation() -> None:
    y, p = np.array([10.0, 20.0]), np.array([12.0, 17.0])
    assert rmse(y, p) == pytest.approx(np.sqrt((4 + 9) / 2))


def test_late_ratio_counts_overestimates() -> None:
    y = np.array([10.0, 20.0, 30.0, 40.0])
    p = np.array([11.0, 19.0, 31.0, 40.0])   # 2개만 초과 (같은 값은 늦은 것이 아님)
    assert late_ratio(y, p) == pytest.approx(0.5)


def test_constant_model_predicts_training_mean() -> None:
    """기준선 모델은 항상 학습 평균을 답해야 한다."""
    X = np.zeros((5, 2))
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    m = ConstantModel().fit(X, y)
    assert np.allclose(m.predict(np.zeros((3, 2))), 30.0)


def test_evaluate_returns_all_metrics() -> None:
    y, p = np.array([10.0, 20.0]), np.array([12.0, 17.0])
    out = evaluate(y, p)
    assert set(out) == {"rmse", "phm08", "late_ratio", "mae"}
    assert all(np.isfinite(v) for v in out.values())
