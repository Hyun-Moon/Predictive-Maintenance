"""
RUL(잔여수명) 예측 모델과 평가 지표
==================================

2단계의 건강도 점수는 "지금 얼마나 나빠졌는가"를 알려줍니다.
이 단계는 한 걸음 더 나가서 **"앞으로 몇 사이클 더 쓸 수 있는가"** 를 맞힙니다.

건강도와 달리 여기서는 **고장 라벨(RUL)을 씁니다.**
    그래서 이 단계는 '실제 회사에 바로 가져갈 수 있는 것'이 아니라
    **방법론이 제대로 동작하는지 벤치마크로 검증하는 단계**입니다.
    (이 구분은 README 와 문서에 명확히 적어 둡니다)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb


# =============================================================================
# 평가 지표
# =============================================================================
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """평균적으로 몇 사이클 틀리는가. 작을수록 좋음."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    PHM08 대회 공식 점수. 작을수록 좋음.

    RMSE 와 결정적으로 다른 점: **늦게 예측하는 쪽에 훨씬 큰 벌점**을 줍니다.

        d = 예측 - 실제
        d < 0 (실제보다 짧게 봄 = 일찍 정비하러 감) : exp(-d/13) - 1
        d > 0 (실제보다 길게 봄 = 고장을 놓침)      : exp( d/10) - 1

    30 사이클을 일찍 본 벌점은 약 9.1,
    30 사이클을 늦게 본 벌점은 약 19.1 로 두 배 가까이 됩니다.

    왜 이렇게 만들었나: 현장 논리 그대로입니다.
    일찍 정비하면 비용이 조금 더 들 뿐이지만, 늦으면 설비가 멈춥니다.
    RMSE 만 보면 이 비대칭이 안 보이므로 두 지표를 같이 봅니다.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalty = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(penalty))


def late_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """실제보다 길게 본(= 고장을 놓칠 위험이 있는) 예측의 비율."""
    return float(np.mean(np.asarray(y_pred) > np.asarray(y_true)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "phm08": phm08_score(y_true, y_pred),
        "late_ratio": late_ratio(y_true, y_pred),
        "mae": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))),
    }


# =============================================================================
# 모델
# =============================================================================
class ConstantModel:
    """
    아무것도 배우지 않는 기준선 — 항상 학습 데이터의 평균 RUL 을 답한다.

    반드시 있어야 하는 모델입니다.
    이것보다 못하면 그 모델은 아무 가치가 없다는 뜻이고,
    "RMSE 20 이 좋은 건가?"라는 질문에 답하려면 비교 대상이 있어야 합니다.
    """

    name = "상수(평균)"

    def fit(self, X, y):
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


def ridge_model() -> object:
    """
    선형 회귀(Ridge) — 두 번째 기준선.

    "센서와 RUL 이 단순한 직선 관계라면 이 정도는 나온다"는 선을 그어 줍니다.
    트리 모델이 이보다 크게 낫지 않다면 굳이 복잡한 모델을 쓸 이유가 없습니다.
    스케일 차이가 크므로 표준화를 앞에 붙입니다.
    """
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def lgbm_model(seed: int = 42) -> lgb.LGBMRegressor:
    """
    LightGBM — 표 형태 데이터에서 가장 강력한 편인 부스팅 트리.

    파라미터는 과적합을 막는 쪽으로 보수적으로 잡았습니다.
    엔진이 100대뿐이라 데이터가 많지 않아, 트리를 깊게 키우면 금방 외워버립니다.
    """
    return lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


MODELS = {
    "상수(평균)": ConstantModel,
    "선형회귀(Ridge)": ridge_model,
    "LightGBM": lgbm_model,
}


# =============================================================================
# 설비 단위 분할
# =============================================================================
def group_split(
    unit_ids: np.ndarray, val_ratio: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    검증 데이터를 **설비 개체 단위**로 나눈다.

    시계열에서 행 단위로 무작위 분할하면 안 됩니다.
    같은 엔진의 100번째 사이클이 학습에, 101번째가 검증에 들어가면
    모델이 사실상 정답을 옆에서 보고 맞히는 셈이 되어 성능이 크게 부풀려집니다.
    (예지보전 프로젝트에서 가장 흔한 함정)

    Returns
    -------
    (학습에 쓸 설비 번호, 검증에 쓸 설비 번호)
    """
    units = np.unique(unit_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    n_val = max(1, int(len(units) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]
