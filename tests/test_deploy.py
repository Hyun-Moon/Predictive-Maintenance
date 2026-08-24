"""
현장 적용 파이프라인 테스트
==========================

5단계에서 겪은 문제들을 다시 만나지 않도록 못박아 둡니다.

    5-2. 평활 창과 지속 조건의 상호작용 (지속조건이 짧으면 오경보)
    5-3. explain() 에 일부 행만 넘겨 편차가 전부 0 이 되던 문제

실행:  pytest -q
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.deploy.demo_data import (
    CONDITION_COLS,
    ID_COL,
    SENSOR_COLS,
    TIME_COL,
    make_ahu_data,
)
from src.deploy.pipeline import DetectorConfig, EquipmentHealthMonitor, EquipmentSpec


@pytest.fixture(scope="module")
def ahu() -> tuple[pd.DataFrame, pd.DataFrame]:
    """공조기 3대, 2년치. 1년차 정상 / 2년차 필터 막힘."""
    data = make_ahu_data(n_units=3, years=2)
    truth = data[[ID_COL, TIME_COL, "_막힘정도"]].copy()
    return data.drop(columns=["_막힘정도"]), truth


@pytest.fixture(scope="module")
def spec() -> EquipmentSpec:
    return EquipmentSpec(ID_COL, TIME_COL, SENSOR_COLS, CONDITION_COLS)


@pytest.fixture(scope="module")
def fitted(ahu, spec):
    data, truth = ahu
    cfg = DetectorConfig(smoothing_window=24, alarm_persistence=72)
    return EquipmentHealthMonitor(spec, cfg).fit(data), data, truth


# =============================================================================
# 데이터 점검
# =============================================================================
def test_check_data_accepts_valid_input(ahu, spec) -> None:
    data, _ = ahu
    result = EquipmentHealthMonitor(spec).check_data(data)
    assert result["ok"], result["issues"]


def test_check_data_rejects_missing_columns(ahu, spec) -> None:
    data, _ = ahu
    result = EquipmentHealthMonitor(spec).check_data(data.drop(columns=["팬전류"]))
    assert not result["ok"]
    assert any("컬럼" in i for i in result["issues"])


def test_check_data_rejects_too_short_history(ahu, spec) -> None:
    """기준선만 있고 감시할 구간이 없으면 사용 불가여야 한다."""
    data, _ = ahu
    tiny = data.groupby(ID_COL).head(10)
    result = EquipmentHealthMonitor(spec).check_data(tiny)
    assert not result["ok"]


def test_check_data_warns_about_dead_sensor(ahu, spec) -> None:
    """값이 변하지 않는 센서는 경고로 알려야 한다 (1~3단계 내내 나온 문제)."""
    data, _ = ahu
    flat = data.copy()
    flat["팬전류"] = 5.0
    result = EquipmentHealthMonitor(spec).check_data(flat)
    assert any("변하지 않는" in w for w in result["warnings"])


# =============================================================================
# 학습 · 채점
# =============================================================================
def test_healthy_period_scores_near_100(fitted) -> None:
    """정상 구간(1년차)의 건강도 중앙값은 100 에 가까워야 한다."""
    monitor, data, truth = fitted
    scored = monitor.score(data).merge(truth, on=[ID_COL, TIME_COL])
    healthy = scored[scored["_막힘정도"] == 0]
    assert healthy["health"].median() > 90


def test_health_declines_as_degradation_grows(fitted) -> None:
    """막힘이 심해질수록 건강도가 낮아져야 한다 (방향이 뒤집히면 여기서 걸림)."""
    monitor, data, truth = fitted
    scored = monitor.score(data).merge(truth, on=[ID_COL, TIME_COL])
    light = scored[scored["_막힘정도"].between(0, 5)]["health"].median()
    heavy = scored[scored["_막힘정도"] > 30]["health"].median()
    assert heavy < light


def test_no_alarm_before_degradation_starts(fitted) -> None:
    """
    5-2 에서 겪은 문제를 막는 테스트.

    막힘이 시작되기 전에 경보가 울리면 안 된다.
    (평활 창과 지속 조건의 균형이 깨지면 여기서 걸린다)
    """
    monitor, data, truth = fitted
    scored = monitor.score(data).merge(truth, on=[ID_COL, TIME_COL])
    for unit, g in scored.groupby(ID_COL):
        alarms = g.loc[g["alarm"]]
        if len(alarms):
            assert alarms["_막힘정도"].iloc[0] > 0, f"{unit}: 막힘 전에 경보"


def test_alarm_fires_for_every_degrading_unit(fitted) -> None:
    """열화가 충분히 진행된 설비는 전부 경보가 울려야 한다."""
    monitor, data, truth = fitted
    scored = monitor.score(data).merge(truth, on=[ID_COL, TIME_COL])
    for unit, g in scored.groupby(ID_COL):
        if g["_막힘정도"].max() > 20:
            assert g["alarm"].any(), f"{unit}: 열화했는데 경보 없음"


def test_score_output_columns(fitted) -> None:
    monitor, data, _ = fitted
    out = monitor.score(data.head(5000))
    assert {"health", "anomaly", "alarm"}.issubset(out.columns)
    assert out["health"].between(0, 100).all()


# =============================================================================
# 설정 안전장치
# =============================================================================
def test_short_persistence_warns() -> None:
    """
    지속 조건이 평활 창의 3배보다 짧으면 경고해야 한다.

    이 규칙을 어겼을 때 실제로 8대 중 7대가 열화 1년 전에 경보를 냈다.
    """
    with pytest.warns(UserWarning, match="지속 조건"):
        DetectorConfig(smoothing_window=24, alarm_persistence=24)


def test_adequate_persistence_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DetectorConfig(smoothing_window=24, alarm_persistence=72)


# =============================================================================
# 원인 설명
# =============================================================================
def test_explain_needs_full_history(fitted) -> None:
    """
    5-3 에서 겪은 문제를 막는 테스트.

    explain 은 전체 데이터를 받아야 한다. 일부만 주면 그 안에서 기준선을 다시 잡아
    편차가 전부 0 이 되어 아무것도 설명하지 못한다.
    """
    monitor, data, truth = fitted
    scored = monitor.score(data).merge(truth, on=[ID_COL, TIME_COL])
    targets = scored[scored["_막힘정도"] > 30].head(5).index

    full = monitor.explain(data, targets=targets)
    assert len(full) == len(targets)
    assert not full["주요 원인"].str.contains(r"\+0\.0σ.*\+0\.0σ").all()


def test_explain_points_at_the_right_sensors(fitted) -> None:
    """
    필터 막힘을 넣었으므로, 심하게 열화된 시점의 주요 원인에는
    회전수·차압·전류 중 하나가 들어와야 한다. (모사한 물리와 맞는지 확인)
    """
    monitor, data, truth = fitted
    scored = monitor.score(data).merge(truth, on=[ID_COL, TIME_COL])
    targets = scored[scored["_막힘정도"] > 40].head(10).index
    causes = " ".join(monitor.explain(data, targets=targets)["주요 원인"])
    assert any(k in causes for k in ("팬회전수", "필터차압", "팬전류"))
