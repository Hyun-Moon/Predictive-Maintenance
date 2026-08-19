"""
운전조건 처리
=============

**이 모듈이 필요한 이유 (숫자로 확인한 것)**

FD002 에서 운전조건에 따른 센서값 차이가 열화로 인한 변화보다
    sensor_11: 15배,  sensor_04: 15배,  sensor_02: 93배
나 컸습니다.

이 상태로 "센서값이 평소보다 높다"를 찾으면 열화가 아니라
**지금 어떤 조건으로 돌고 있는지**를 찾게 됩니다.

    현장으로 옮기면 정확히 이런 이야기입니다.
    "오늘 공조기 소비전력이 평소보다 높다" -> 열화인가, 아니면 그냥 외기온이 높은 날인가?
    외기온 5℃일 때와 30℃일 때의 전력을 그냥 비교하면 안 됩니다.
    조건이 같을 때끼리 비교해야 합니다.

**해결 순서 (이 순서가 중요합니다)**

    1. 운전조건을 식별한다           (외기온/부하율 구간을 나누는 것과 같음)
    2. 조건별로 정규화한다           <- 조건 효과를 먼저 제거
    3. 그 다음 설비별 기준선을 잡는다  <- 남은 것이 순수한 개체차 + 열화

    2번을 건너뛰고 3번을 하면, 설비 기준선이 "그 설비가 주로 어떤 조건으로
    돌았는지"를 담게 되어 엉뚱한 값이 됩니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load import ID_COL, OP_COLS, TIME_COL


# =============================================================================
# 1. 운전조건 식별
# =============================================================================
def condition_key(df: pd.DataFrame, op_cols: list[str] | None = None) -> pd.Series:
    """
    각 행이 어떤 운전조건인지 문자열 키로 만든다.

    C-MAPSS 의 운전조건은 값이 뚜렷하게 떨어져 있어서(예: 고도 0/10/20/25/35/42)
    반올림만으로 깔끔하게 나뉩니다.
    KMeans(k=6) 결과와 완전히 일치하는 것을 확인했으므로,
    굳이 군집화를 돌리지 않고 반올림을 씁니다. (단순하고 재현이 확실함)

    실제 설비라면 외기온 5℃ 구간, 부하율 10% 구간처럼
    도메인 지식으로 구간을 정하는 것이 이 단계에 해당합니다.
    """
    cols = op_cols or OP_COLS
    return df[cols].round(0).astype(int).astype(str).agg("|".join, axis=1)


# =============================================================================
# 2. 조건별 정규화
# =============================================================================
class ConditionNormalizer:
    """
    조건별 평균·표준편차로 센서값을 z점수로 바꾼다.

    핵심 설계: 조건별 통계를 **설비 한 대가 아니라 전체 설비(fleet)에서** 구합니다.
        설비 한 대의 초기 구간만으로는 조건 6개 × 센서 19개의 통계를
        안정적으로 낼 만큼 표본이 모이지 않습니다.
        (실제로 어떤 엔진은 특정 조건이 초기 구간에 1번밖에 안 나옵니다)
        조건에 따른 센서 반응은 설비 개체와 무관한 물리 특성이므로,
        전체에서 구해도 됩니다. 개체차는 다음 단계(설비별 기준선)에서 다룹니다.

    학습 데이터에서만 fit 하고 시험 데이터에는 transform 만 적용합니다.
    """

    def __init__(self, min_samples: int = 20):
        self.min_samples = min_samples

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "ConditionNormalizer":
        self.cols_ = list(cols)
        key = condition_key(df)
        g = df.groupby(key)[self.cols_]

        self.mean_ = g.mean()
        self.std_ = g.std().replace(0.0, np.nan)
        self.counts_ = g.size()

        # 표본이 너무 적은 조건은 통계를 믿을 수 없으므로 전체 평균으로 대체
        thin = self.counts_[self.counts_ < self.min_samples].index
        if len(thin):
            self.mean_.loc[thin] = df[self.cols_].mean()
            self.std_.loc[thin] = df[self.cols_].std()

        self.global_mean_ = df[self.cols_].mean()
        self.global_std_ = df[self.cols_].std().replace(0.0, np.nan)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """조건 효과가 제거된 z점수 DataFrame 을 돌려준다 (컬럼명 유지)."""
        key = condition_key(df)
        # 학습에 없던 조건이 나오면 전체 통계로 처리 (실제 운영에서 충분히 생기는 일)
        mean = self.mean_.reindex(key.to_numpy())
        std = self.std_.reindex(key.to_numpy())
        mean = mean.fillna(self.global_mean_).to_numpy()
        std = std.fillna(self.global_std_).to_numpy()

        z = (df[self.cols_].to_numpy() - mean) / std
        return pd.DataFrame(np.nan_to_num(z), index=df.index, columns=self.cols_)


# =============================================================================
# 3. 설비 길이에 맞춘 기준선 구간
# =============================================================================
def adaptive_baseline_mask(
    df: pd.DataFrame,
    target: int = 30,
    frac: float = 0.3,
    min_cycles: int = 5,
) -> pd.Series:
    """
    설비마다 '초기 정상 구간'을 몇 사이클로 잡을지 알아서 정한다.

    왜 고정값(30)이면 안 되는가 — 실제로 겪은 문제:
        FD002 시험 세트에는 관측이 21사이클뿐인 엔진이 있습니다.
        30사이클을 기준선으로 잡으면 그 엔진은 **관측 전체가 기준선**이 되어,
        "자기 자신 대비 편차"가 항상 0 이 됩니다.
        모델은 이 엔진을 영원히 건강하다고 판단하게 됩니다.
        FD002 에서 6대, FD004 에서 11대가 여기 해당했습니다.

    규칙: 그 설비 전체 길이의 frac(기본 30%), 단 min_cycles ~ target 사이로 자름.
        관측이 짧으면 기준선도 짧게 잡되, 예측할 구간은 반드시 남깁니다.
    """
    length = df.groupby(ID_COL)[TIME_COL].transform("max")
    n = np.clip(np.round(length * frac), min_cycles, target)
    return df[TIME_COL] <= n
