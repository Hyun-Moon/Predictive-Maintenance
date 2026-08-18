"""
건강도 점수 모듈 테스트
======================

여기서 검증하는 것은 "성능이 좋은가"가 아니라 **"코드가 의도대로 동작하는가"** 입니다.
성능은 데이터에 따라 달라지지만, 아래 성질들은 항상 성립해야 합니다.

특히 2-2 에서 겪은 '건강도 스케일 오설계'가 다시 생기지 않도록
기준점 두 개(정상=100, 경보=50)를 테스트로 못박아 둡니다.

실행:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.health_index import (
    SCORERS,
    HealthScale,
    auc,
    debounced_alarm,
)

SEED = 0


@pytest.fixture
def normal_and_drifted() -> tuple[np.ndarray, np.ndarray]:
    """정상 데이터와, 거기서 뚜렷하게 벗어난 데이터 한 쌍."""
    rng = np.random.default_rng(SEED)
    normal = rng.normal(0, 1, size=(600, 5))
    drifted = rng.normal(0, 1, size=(300, 5)) + 4.0   # 4시그마 이동
    return normal, drifted


# =============================================================================
# 이상 점수 계산기
# =============================================================================
@pytest.mark.parametrize("name", list(SCORERS))
def test_scorer_flags_drift(name: str, normal_and_drifted) -> None:
    """
    어떤 방법을 쓰든, 정상에서 뚜렷하게 벗어난 데이터는
    더 높은 이상 점수를 받아야 한다. (부호가 뒤집혀 있으면 여기서 걸림)
    """
    normal, drifted = normal_and_drifted
    scorer = SCORERS[name]().fit(normal)
    assert scorer.score(drifted).mean() > scorer.score(normal).mean()


@pytest.mark.parametrize("name", list(SCORERS))
def test_scorer_output_shape(name: str, normal_and_drifted) -> None:
    """점수는 행마다 하나씩, 유한한 값이어야 한다."""
    normal, drifted = normal_and_drifted
    s = SCORERS[name]().fit(normal).score(drifted)
    assert s.shape == (len(drifted),)
    assert np.isfinite(s).all()


def test_pca_component_count_matters() -> None:
    """
    2-1 에서 겪은 문제를 못박는 테스트.

    분산 95% 기준으로 두면 주성분이 거의 전부 선택되어 압축이 일어나지 않는다.
    (그래서 이상탐지가 실패했다) 개수를 직접 지정하면 실제로 줄어들어야 한다.
    """
    rng = np.random.default_rng(SEED)
    X = rng.normal(0, 1, size=(500, 10))   # 서로 상관 없는 10차원 = 압축 불가

    many = SCORERS["pca_residual_var95"]().fit(X)
    few = SCORERS["pca_residual"]().fit(X)          # 기본 주성분 2개

    assert many.n_components_ > few.n_components_
    assert few.n_components_ == 2


# =============================================================================
# 0~100 건강도 변환
# =============================================================================
def test_health_scale_anchors() -> None:
    """
    기준점 두 개가 정확히 맞아야 한다.
        정상 중앙값 -> 100점
        경보 지점   -> 50점
    (한 개짜리 식을 쓰던 시절엔 정상이 75점에서 시작하는 문제가 있었다)
    """
    rng = np.random.default_rng(SEED)
    normal_scores = rng.gamma(shape=2.0, scale=1.5, size=5000)

    scale = HealthScale().fit(normal_scores)

    assert scale.to_health(np.array([scale.d_normal_]))[0] == pytest.approx(100.0)
    assert scale.to_health(np.array([scale.d_alarm_]))[0] == pytest.approx(50.0)


def test_health_scale_bounds_and_direction() -> None:
    """건강도는 0~100 을 벗어나지 않고, 이상 점수가 클수록 낮아야 한다."""
    rng = np.random.default_rng(SEED)
    normal_scores = rng.gamma(2.0, 1.5, size=2000)
    scale = HealthScale().fit(normal_scores)

    probe = np.array([0.0, scale.d_normal_, scale.d_alarm_, scale.d_alarm_ * 50])
    h = scale.to_health(probe)

    assert (h >= 0).all() and (h <= 100).all()
    assert (np.diff(h) <= 0).all(), "이상 점수가 커지는데 건강도가 올라감"


def test_health_scale_survives_degenerate_input() -> None:
    """정상 점수가 전부 같은 값이어도 0으로 나누지 않아야 한다."""
    scale = HealthScale().fit(np.full(100, 3.0))
    h = scale.to_health(np.array([3.0, 10.0]))
    assert np.isfinite(h).all()
    assert h[0] > h[1]


# =============================================================================
# 경보 규칙
# =============================================================================
def _alarm_frame() -> pd.DataFrame:
    """설비 2대. A 는 3사이클 연속 하락, B 는 1사이클만 튐."""
    return pd.DataFrame(
        {
            "unit": ["A"] * 6 + ["B"] * 6,
            "health": [100, 100, 40, 40, 40, 100] + [100, 40, 100, 100, 100, 100],
        }
    )


def test_debounce_off_flags_every_dip() -> None:
    """지속 조건이 없으면 잠깐 튄 것도 전부 경보로 잡힌다."""
    df = _alarm_frame()
    fired = debounced_alarm(df, "unit", "health", threshold=50, persistence=1)
    assert fired.sum() == 4          # A 3회 + B 1회


def test_debounce_filters_single_spike() -> None:
    """
    3사이클 연속 조건을 걸면 B 의 일시적인 튐은 걸러지고
    A 의 지속적인 하락만 경보가 된다. (오경보 대책의 핵심 동작)
    """
    df = _alarm_frame()
    fired = debounced_alarm(df, "unit", "health", threshold=50, persistence=3)
    assert fired[df["unit"] == "B"].sum() == 0
    assert fired[df["unit"] == "A"].sum() == 1   # 3회째 시점에 발화


def test_debounce_does_not_leak_across_units() -> None:
    """
    설비 경계를 넘어 연속으로 세면 안 된다.
    A 마지막 2회 + B 첫 1회를 이어 붙이면 3연속이 되지만, 서로 다른 설비이므로 경보가 아니다.
    """
    df = pd.DataFrame(
        {"unit": ["A", "A", "B", "B"], "health": [40, 40, 40, 100]}
    )
    fired = debounced_alarm(df, "unit", "health", threshold=50, persistence=3)
    assert fired.sum() == 0


# =============================================================================
# 평가 지표
# =============================================================================
def test_auc_is_direction_agnostic() -> None:
    """
    구분력만 보므로, 어느 쪽이 큰지와 무관하게 같은 값이 나와야 한다.
    (오르는 센서와 내리는 센서를 함께 비교하기 위한 성질)
    """
    a, b = np.array([1.0, 2, 3]), np.array([4.0, 5, 6])
    assert auc(a, b) == pytest.approx(1.0)
    assert auc(b, a) == pytest.approx(1.0)


def test_auc_of_identical_groups_is_half() -> None:
    """완전히 겹치는 두 집단은 구분력이 0.5(동전던지기)여야 한다."""
    x = np.arange(50.0)
    assert auc(x, x.copy()) == pytest.approx(0.5, abs=0.02)


def test_auc_handles_empty_group() -> None:
    """한쪽이 비면 계산 불가를 NaN 으로 알린다. (조용히 0 을 반환하면 안 됨)"""
    assert np.isnan(auc(np.array([]), np.array([1.0, 2.0])))
