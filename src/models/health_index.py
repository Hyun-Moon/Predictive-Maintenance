"""
건강도 점수 (Health Index) 모델
==============================

"지금 상태가 정상 범위에서 얼마나 벗어났는가"를 하나의 숫자로 만드는 방법들입니다.

공통 규칙 — 전부 **정상 데이터만으로 학습**합니다 (비지도).
    고장 라벨을 쓰지 않기 때문에, 고장 이력이 없는 실제 회사 설비에도
    그대로 적용할 수 있습니다. 이것이 이 프로젝트의 핵심 설계 원칙입니다.

네 가지 방법을 비교합니다.

    1) ZScoreSum      각 센서가 정상 대비 몇 표준편차 벗어났는지 평균 (가장 단순한 기준선)
    2) Mahalanobis    센서 간 상관관계까지 고려한 거리 (고전적인 다변량 이상탐지)
    3) PCAResidual    정상 패턴으로 압축했다 복원했을 때의 오차
    4) IsolationForest 정상 데이터에서 얼마나 '떼어내기 쉬운' 점인지

왜 여러 개를 만드나?
    어느 것이 좋은지는 데이터마다 다릅니다. 하나만 만들어놓고 "잘 된다"고 하는 것보다
    같은 조건에서 비교해서 고른 근거를 남기는 편이 훨씬 신뢰를 얻습니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 공통 인터페이스
# =============================================================================
class AnomalyScorer:
    """
    이상 점수 계산기의 공통 틀.

    fit(정상데이터)  -> 정상이 어떤 모습인지 학습
    score(데이터)    -> 각 행이 정상에서 얼마나 벗어났는지 (클수록 이상)

    네 가지 방법이 전부 이 틀을 따르므로, 실험 코드에서 똑같이 바꿔 끼울 수 있습니다.
    """

    name = "base"

    def fit(self, X: np.ndarray) -> "AnomalyScorer":
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ZScoreSum(AnomalyScorer):
    """
    가장 단순한 방법: 각 센서가 정상 평균에서 몇 표준편차 떨어졌는지를 평균낸다.

    장점: 계산이 단순하고 "어느 센서가 얼마나 기여했는지" 바로 설명됨.
    한계: 센서끼리의 관계를 무시함.
          예를 들어 '회전수가 오르면 온도도 오르는 것이 정상'인데,
          이 방법은 둘 다 올랐다고 두 배로 이상하다고 판단해 버립니다.
    """

    name = "zscore_sum"

    def fit(self, X: np.ndarray) -> "ZScoreSum":
        self.scaler_ = StandardScaler().fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.abs(self.scaler_.transform(X)).mean(axis=1)


class Mahalanobis(AnomalyScorer):
    """
    센서 간 상관관계를 고려한 거리.

    직관: '회전수와 온도가 같이 오르는 것'은 정상 패턴이므로 거리를 조금만 준다.
          반대로 '회전수는 그대로인데 온도만 오르는 것'은 정상에서 벗어난 조합이므로
          같은 크기의 변화라도 거리를 크게 준다.

    공분산 역행렬이 필요한데, 센서끼리 거의 똑같이 움직이면 역행렬이 불안정해집니다.
    그래서 pinv(유사역행렬)를 쓰고, 대각선에 아주 작은 값을 더해 안정화합니다.
    """

    name = "mahalanobis"

    def __init__(self, ridge: float = 1e-6):
        self.ridge = ridge

    def fit(self, X: np.ndarray) -> "Mahalanobis":
        self.mean_ = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        cov = cov + np.eye(cov.shape[0]) * self.ridge
        self.inv_cov_ = np.linalg.pinv(cov)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        d = X - self.mean_
        # 각 행에 대해 sqrt(d @ inv_cov @ d.T) 를 계산 (einsum 으로 한 번에)
        m2 = np.einsum("ij,jk,ik->i", d, self.inv_cov_, d)
        return np.sqrt(np.clip(m2, 0, None))


class PCAResidual(AnomalyScorer):
    """
    정상 데이터의 주요 패턴만 남기고 압축했다가 되돌렸을 때의 오차.

    직관: 정상 상태의 센서 14개는 사실 몇 개의 '공통 움직임'으로 거의 설명됩니다
          (부하가 오르면 온도·압력·회전수가 함께 움직이는 식).
          정상이면 압축했다 펴도 원래대로 복원되지만,
          정상에서 벗어난 새로운 패턴이 생기면 복원이 안 되고 오차가 커집니다.

    n_components 를 몇 개로 잡느냐가 결정적입니다. (실제로 크게 데인 부분)
        "분산의 95%를 담는 만큼" 이라는 흔한 기본값을 쓰면
        이 데이터에서는 14개 중 12~13개 주성분이 선택됩니다.
        거의 압축을 안 하는 셈이라 무엇을 넣어도 그대로 복원되고,
        따라서 재구성 오차가 이상 신호를 못 담습니다. (AUC 0.708)
        주성분을 1~2개로 강하게 제한해 '정상의 큰 줄기'만 남기면
        나머지가 전부 오차로 드러나면서 성능이 뒤집힙니다. (AUC 0.940)

    n_components 인자는 sklearn 규칙을 그대로 따릅니다.
        정수 -> 그 개수만큼 사용,  0~1 실수 -> 그 비율의 분산을 담는 만큼 사용
    """

    name = "pca_residual"

    def __init__(self, n_components: float | int = 2):
        self.var_ratio = n_components

    def fit(self, X: np.ndarray) -> "PCAResidual":
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        self.pca_ = PCA(n_components=self.var_ratio, svd_solver="full").fit(Z)
        self.n_components_ = self.pca_.n_components_
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler_.transform(X)
        recon = self.pca_.inverse_transform(self.pca_.transform(Z))
        return np.sqrt(((Z - recon) ** 2).sum(axis=1))


class IForest(AnomalyScorer):
    """
    Isolation Forest — 무작위로 자르기를 반복했을 때 빨리 혼자 남는 점일수록 이상하다고 본다.

    장점: 분포 모양을 가정하지 않음. 이상치에 강함.
    한계: 결과가 확률적이라 seed 를 고정해야 재현됨. 해석이 앞의 세 방법보다 어려움.
    """

    name = "isolation_forest"

    def __init__(self, n_estimators: int = 300, seed: int = 42):
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, X: np.ndarray) -> "IForest":
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.seed, n_jobs=-1
        ).fit(self.scaler_.transform(X))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # score_samples 는 정상일수록 큰 값 -> 부호를 뒤집어 '클수록 이상'으로 통일
        return -self.model_.score_samples(self.scaler_.transform(X))


class PCAResidualVar95(PCAResidual):
    """분산 95% 기준 PCA. 처음에 썼다가 실패한 설정 — 비교를 위해 남겨둡니다."""

    name = "pca_residual(분산95%)"

    def __init__(self):
        super().__init__(n_components=0.95)


SCORERS = {
    "zscore_sum": ZScoreSum,
    "mahalanobis": Mahalanobis,
    "pca_residual": PCAResidual,          # 주성분 2개 (재시도 후 설정)
    "pca_residual_var95": PCAResidualVar95,  # 실패한 설정, 비교용
    "isolation_forest": IForest,
}


# =============================================================================
# 이상 점수 -> 0~100 건강도 점수
# =============================================================================
@dataclass
class HealthScale:
    """
    이상 점수(거리)를 현장에서 읽을 수 있는 0~100 점으로 바꾼다.

    변환식:  HI = 100 * 0.5 ** ((d - d_normal) / (d_alarm - d_normal))   (0~100 으로 자름)

    기준점 두 개를 잡아 직선처럼 맞추는 방식입니다.
        d = d_normal (정상의 전형적인 거리, 중앙값) -> HI = 100
        d = d_alarm  (정상 분포의 상위 5% 지점)      -> HI = 50   <- 경보선
        그보다 커지면 부드럽게 0 으로 수렴

    왜 기준점이 두 개여야 하는가 — 처음에 한 개로 만들었다가 실패했습니다.
        처음에는 `HI = 100 * 0.5 ** (d / d_alarm)` 로 두었습니다.
        즉 "거리 0 = 건강도 100" 이라고 본 것인데,
        정상 상태의 거리는 애초에 0 이 아니라 어떤 중앙값(이 데이터에서는 2.8)입니다.
        그래서 **멀쩡한 설비가 처음부터 75점으로 표시**되고,
        경보선(50점)과의 여유가 25점밖에 안 남아 오경보가 크게 늘었습니다.
        기준점을 두 개로 바꾸면 정상이 제대로 100점에서 출발합니다.

    alarm_percentile 은 현장에서 조정하는 손잡이입니다.
        95 는 "정상인데도 20번에 1번은 경보선에 닿는다"는 뜻.
        올리면 둔감해지고(오경보 감소), 내리면 예민해집니다.
    """

    alarm_percentile: float = 95.0
    normal_percentile: float = 50.0
    d_normal_: float = field(default=np.nan, init=False)
    d_alarm_: float = field(default=np.nan, init=False)

    def fit(self, normal_scores: np.ndarray) -> "HealthScale":
        self.d_normal_ = float(np.percentile(normal_scores, self.normal_percentile))
        self.d_alarm_ = float(np.percentile(normal_scores, self.alarm_percentile))
        # 두 기준점이 붙어버리면 나눗셈이 폭발하므로 최소 간격을 보장
        if self.d_alarm_ <= self.d_normal_:
            self.d_alarm_ = self.d_normal_ + max(abs(self.d_normal_) * 1e-3, 1e-9)
        return self

    def to_health(self, scores: np.ndarray) -> np.ndarray:
        t = (scores - self.d_normal_) / (self.d_alarm_ - self.d_normal_)
        return np.clip(100.0 * np.power(0.5, t), 0.0, 100.0)


# =============================================================================
# 경보 규칙
# =============================================================================
def debounced_alarm(
    df: pd.DataFrame,
    id_col: str,
    health_col: str = "health",
    threshold: float = 50.0,
    persistence: int = 1,
) -> pd.Series:
    """
    '건강도가 임계 미만인 상태가 N사이클 연속될 때만 경보' 규칙.

    왜 필요한가?
        한 시점만 보고 경보를 울리면, 노이즈로 잠깐 튄 값에도 알람이 납니다.
        현장에서 오경보가 몇 번 반복되면 그 다음부터는 아무도 알람을 안 봅니다.
        (실제 설비관리에서 가장 흔한 실패 방식입니다)

        persistence 를 키우면 오경보는 줄지만 그만큼 경보가 늦어집니다.
        이 값이 현장에서 조정하는 손잡이가 됩니다.

    Returns
    -------
    각 행이 '경보 상태'인지 나타내는 bool Series (원본 index 유지).
    """
    below = df[health_col] < threshold
    if persistence <= 1:
        return below

    fired = (
        below.groupby(df[id_col])
        .transform(lambda s: s.rolling(persistence, min_periods=persistence).sum() == persistence)
        .fillna(False)
        .astype(bool)
    )
    return fired


# =============================================================================
# 평가
# =============================================================================
def auc(reference: np.ndarray, target: np.ndarray) -> float:
    """
    두 집단의 구분력. 0.5 = 못 구분, 1.0 = 완벽.
    (EDA 와 같은 지표를 써야 단계별 비교가 가능합니다)
    """
    if len(reference) == 0 or len(target) == 0:
        return np.nan
    both = np.concatenate([reference, target])
    ranks = pd.Series(both).rank().to_numpy()
    n1, n2 = len(reference), len(target)
    v = (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)
    return float(max(v, 1 - v))
