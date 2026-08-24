"""
현장 적용용 파이프라인
=====================

1~4단계에서 검증한 방법을 **설비 종류와 무관하게** 쓸 수 있도록 하나로 묶은 것입니다.

    C-MAPSS(항공기 엔진)로 만들었지만, 이 클래스는 엔진에 대해 아무것도 모릅니다.
    아는 것은 "설비 번호 / 시간 / 센서 여러 개 / (선택) 운전조건" 이라는 데이터 형태뿐입니다.
    공조기든 펌프든 냉동기든 이 형태로만 넣으면 그대로 동작합니다.

**이것이 이 프로젝트의 핵심 주장을 코드로 증명하는 부분입니다.**
"방법론은 설비 종류와 무관하다"는 말은 설명으로 하는 것보다
컬럼 이름만 바꿔서 다른 설비에 돌아가는 것을 보여주는 편이 확실합니다.

고장 라벨이 필요 없습니다 (2단계 결론).
    학습에 쓰는 것은 각 설비의 **초기 정상 구간**뿐입니다.
    실제 현장에는 run-to-failure 기록이 거의 없지만 정상 운전 데이터는 어디에나 있습니다.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.health_index import SCORERS, HealthScale


@dataclass
class EquipmentSpec:
    """
    데이터가 어떻게 생겼는지 알려주는 설정.

    실제 회사 데이터로 갈아끼울 때 **여기만 바꾸면 됩니다.**

    예) 공조기
        id_col="설비번호", time_col="일시",
        sensor_cols=["급기온도","환기온도","차압","팬전류","팬RPM","냉수밸브개도"],
        condition_cols=["외기온","부하율"]
    """

    id_col: str
    time_col: str
    sensor_cols: list[str]
    # 운전조건 컬럼 (외기온·부하율 등). 없으면 빈 리스트.
    #   4단계에서 확인: 조건이 여러 개인데 이걸 안 주면 조건 차이가 열화를 덮습니다.
    condition_cols: list[str] = field(default_factory=list)
    # 조건을 몇 개 구간으로 나눌지 (외기온 5℃ 간격 같은 것)
    condition_bins: int = 6


@dataclass
class DetectorConfig:
    """모델 쪽 설정. 기본값은 1~4단계에서 검증된 값입니다."""

    # 설비 이력의 앞 몇 %를 정상 기준으로 삼을지 (4단계의 적응형 규칙)
    #
    # 계절을 타는 설비(공조기·냉동기)는 이 구간이 **사계절을 모두 덮어야** 합니다.
    # 겨울 데이터만으로 기준선을 잡으면 여름이 통째로 이상으로 판정됩니다.
    # (실제로 겪은 문제 — 막힘이 시작되기 전에 경보가 울렸습니다)
    baseline_ratio: float = 0.5
    baseline_min: int = 20
    baseline_max: int = 24 * 365     # 최대 1년치까지는 기준선으로 허용
    smoothing_window: int = 10       # 이동평균 (2단계: 개체차 보정 후에 효과가 남)
    scorer: str = "mahalanobis"
    alarm_percentile: float = 95.0
    alarm_health: float = 50.0

    # N회 연속 경보선 아래일 때만 실제 경보 (2단계의 오경보 대책)
    #
    # **평활 창과 독립이 아닙니다.** 실제로 여기서 크게 데었습니다.
    #   평활을 걸면 값이 부드러워지는 대신 한번 내려가면 오래 머뭅니다.
    #   그래서 평활 창을 키우면서 지속 조건을 그대로 두면,
    #   정상 구간의 일시적인 이탈도 지속 조건을 통과해 경보가 됩니다.
    #   (평활 24 / 지속 24 로 두었더니 8대 중 7대가 열화 시작 1년 전에 경보)
    #
    #   측정해 보니 **지속 조건이 평활 창의 3배 이상**이면 오경보가 사라졌습니다.
    #   아래 __post_init__ 에서 이 조건을 검사합니다.
    alarm_persistence: int = 30

    # 지속 조건 / 평활 창 최소 비율
    min_persistence_ratio: float = 3.0

    def __post_init__(self) -> None:
        need = int(np.ceil(self.smoothing_window * self.min_persistence_ratio))
        if self.alarm_persistence < need:
            warnings.warn(
                f"지속 조건({self.alarm_persistence})이 평활 창({self.smoothing_window})에 비해 "
                f"짧습니다. 정상 구간의 일시적 이탈도 경보가 될 수 있습니다. "
                f"{need} 이상을 권장합니다.",
                stacklevel=2,
            )


class EquipmentHealthMonitor:
    """
    설비 건강도 감시기.

    사용:
        monitor = EquipmentHealthMonitor(spec).fit(history_df)
        result = monitor.score(new_df)
    """

    def __init__(self, spec: EquipmentSpec, config: DetectorConfig | None = None):
        self.spec = spec
        self.cfg = config or DetectorConfig()

    # -------------------------------------------------------------------------
    # 데이터 점검 — 학습 전에 반드시
    # -------------------------------------------------------------------------
    def check_data(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        이 데이터로 예지보전이 가능한지 먼저 확인한다.

        현장에서 가장 흔한 실패는 모델이 나빠서가 아니라
        **애초에 안 되는 데이터로 시작해서**입니다.
        그래서 학습 전에 조건을 명시적으로 검사합니다.
        """
        s = self.spec
        issues: list[str] = []
        warnings: list[str] = []

        missing = [c for c in [s.id_col, s.time_col, *s.sensor_cols] if c not in df.columns]
        if missing:
            issues.append(f"컬럼이 없습니다: {missing}")
            return {"ok": False, "issues": issues, "warnings": warnings, "summary": {}}

        n_units = df[s.id_col].nunique()
        per_unit = df.groupby(s.id_col).size()
        shortest = int(per_unit.min())

        if shortest < self.cfg.baseline_min * 2:
            issues.append(
                f"이력이 너무 짧은 설비가 있습니다 (최소 {shortest}행). "
                f"기준선({self.cfg.baseline_min}행)을 잡고도 감시할 구간이 남아야 합니다."
            )

        # 값이 변하지 않는 센서 (3단계까지 계속 나온 문제)
        dead = [c for c in s.sensor_cols if df[c].std(skipna=True) <= 1e-9]
        if dead:
            warnings.append(f"값이 변하지 않는 센서 {len(dead)}개는 제외됩니다: {dead}")

        na_ratio = df[s.sensor_cols].isna().mean()
        bad_na = na_ratio[na_ratio > 0.3]
        if len(bad_na):
            warnings.append(f"결측이 30%를 넘는 센서: {list(bad_na.index)}")

        if s.condition_cols and not all(c in df.columns for c in s.condition_cols):
            warnings.append("운전조건 컬럼이 없어 조건 정규화를 건너뜁니다.")

        usable = [c for c in s.sensor_cols if c not in dead]
        if len(usable) < 3:
            issues.append(f"쓸 수 있는 센서가 {len(usable)}개뿐입니다. 최소 3개는 필요합니다.")

        return {
            "ok": not issues,
            "issues": issues,
            "warnings": warnings,
            "summary": {
                "설비 수": n_units,
                "전체 행": len(df),
                "설비당 행(최소/중앙)": f"{shortest} / {int(per_unit.median())}",
                "쓸 수 있는 센서": len(usable),
                "제외된 센서": len(dead),
            },
        }

    # -------------------------------------------------------------------------
    # 내부 처리
    # -------------------------------------------------------------------------
    def _condition_key(self, df: pd.DataFrame) -> pd.Series:
        """
        운전조건을 구간으로 나눈다.

        C-MAPSS 는 조건이 딱 떨어졌지만, 실제 설비의 외기온·부하율은 연속값입니다.
        그래서 분위수로 나눕니다 (외기온 하위 1/6, 2/6, ... 식).
        분위수를 쓰는 이유: 각 구간에 표본이 고르게 들어가 통계가 안정됩니다.
        """
        if not self.spec.condition_cols:
            return pd.Series("all", index=df.index)

        parts = []
        for c in self.spec.condition_cols:
            if c not in df.columns:
                continue
            edges = self._cond_edges_[c]
            parts.append(pd.cut(df[c], bins=edges, labels=False,
                                include_lowest=True).fillna(-1).astype(int).astype(str))
        if not parts:
            return pd.Series("all", index=df.index)
        return pd.concat(parts, axis=1).agg("|".join, axis=1)

    def _baseline_mask(self, df: pd.DataFrame) -> pd.Series:
        """설비별로 이력 앞부분을 '정상 기준'으로 잡는다 (4단계의 적응형 규칙)."""
        s, c = self.spec, self.cfg
        order = df.groupby(s.id_col).cumcount()
        length = df.groupby(s.id_col)[s.time_col].transform("size")
        n = np.clip(np.round(length * c.baseline_ratio), c.baseline_min, c.baseline_max)
        return order < n

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        """조건 정규화 → 설비별 기준선 편차 → 이동평균. 1~4단계에서 정한 순서 그대로."""
        s, c = self.spec, self.cfg
        work = df.sort_values([s.id_col, s.time_col]).copy()

        # 1) 조건 정규화 (4단계)
        key = self._condition_key(work)
        mean = self._cond_mean_.reindex(key.to_numpy()).fillna(self._global_mean_)
        std = self._cond_std_.reindex(key.to_numpy()).fillna(self._global_std_)
        z = (work[self.sensors_].to_numpy() - mean.to_numpy()) / std.to_numpy()
        z = np.nan_to_num(z)

        # 2) 설비별 기준선 대비 편차 (1~2단계)
        zdf = pd.DataFrame(z, index=work.index, columns=self.sensors_)
        zdf[s.id_col] = work[s.id_col].to_numpy()
        base = zdf[self._baseline_mask(work).to_numpy()].groupby(s.id_col)[self.sensors_].mean()
        dev = zdf[self.sensors_] - zdf[s.id_col].map(base.to_dict("index")).apply(pd.Series)[self.sensors_]

        # 3) 이동평균 (2단계: 개체차 보정 이후에 효과가 있음)
        dev[s.id_col] = zdf[s.id_col]
        sm = dev.groupby(s.id_col)[self.sensors_].transform(
            lambda x: x.rolling(c.smoothing_window, min_periods=1).mean()
        )
        return np.nan_to_num(sm.to_numpy()), work.index

    # -------------------------------------------------------------------------
    # 학습 / 채점
    # -------------------------------------------------------------------------
    def fit(self, history: pd.DataFrame) -> "EquipmentHealthMonitor":
        """정상 이력으로 학습한다. **고장 라벨은 쓰지 않는다.**"""
        s = self.spec
        self.sensors_ = [c for c in s.sensor_cols if history[c].std(skipna=True) > 1e-9]

        work = history.sort_values([s.id_col, s.time_col]).copy()

        # 조건 구간 경계를 학습 데이터에서 확정 (이후 새 데이터에도 같은 경계 사용)
        self._cond_edges_ = {}
        for c in s.condition_cols:
            if c in work.columns:
                q = np.linspace(0, 1, s.condition_bins + 1)
                edges = np.unique(work[c].quantile(q).to_numpy())
                edges[0], edges[-1] = -np.inf, np.inf
                self._cond_edges_[c] = edges

        key = self._condition_key(work)
        g = work.groupby(key)[self.sensors_]
        self._cond_mean_ = g.mean()
        self._cond_std_ = g.std().replace(0.0, np.nan)
        self._global_mean_ = work[self.sensors_].mean()
        self._global_std_ = work[self.sensors_].std().replace(0.0, np.nan)

        X, _ = self._features(work)
        normal = self._baseline_mask(work).to_numpy()

        self.scorer_ = SCORERS[self.cfg.scorer]().fit(X[normal])
        self.scale_ = HealthScale(alarm_percentile=self.cfg.alarm_percentile).fit(
            self.scorer_.score(X[normal])
        )
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        건강도 점수와 경보 상태를 붙여 돌려준다.

        Returns 컬럼: health(0~100), anomaly(이상 점수), alarm(경보 여부)
        """
        s, c = self.spec, self.cfg
        X, idx = self._features(df)
        anomaly = self.scorer_.score(X)
        health = self.scale_.to_health(anomaly)

        out = df.loc[idx, [s.id_col, s.time_col]].copy()
        out["anomaly"] = anomaly
        out["health"] = health

        below = out["health"] < c.alarm_health
        if c.alarm_persistence > 1:
            out["alarm"] = (
                below.groupby(out[s.id_col])
                .transform(lambda x: x.rolling(c.alarm_persistence,
                                               min_periods=c.alarm_persistence).sum()
                           == c.alarm_persistence)
                .fillna(False)
                .astype(bool)
            )
        else:
            out["alarm"] = below
        return out.sort_index()

    def explain(
        self, df: pd.DataFrame, targets: pd.Index | None = None, top_k: int = 3
    ) -> pd.DataFrame:
        """
        건강도가 왜 떨어졌는지 — 어느 센서가 얼마나 벗어났는지 돌려준다.

        현장에서 "점수가 낮다"만으로는 아무도 움직이지 않습니다.
        "필터차압이 평소보다 3.2 표준편차 높다"까지 나와야 정비원이 확인하러 갑니다.

        Parameters
        ----------
        df : **전체** 데이터. 일부만 넘기면 안 됩니다.
             기준선과 이동평균은 그 설비의 이력 전체가 있어야 계산됩니다.
             (처음에 설명할 행만 잘라서 넘겼다가 편차가 전부 0 으로 나왔습니다)
        targets : 설명이 필요한 행의 index. None 이면 전체.
        """
        X, idx = self._features(df)
        dev = pd.DataFrame(X, index=idx, columns=self.sensors_)
        if targets is not None:
            dev = dev.loc[dev.index.intersection(targets)]

        rows = []
        for i, r in dev.iterrows():
            top = r.abs().nlargest(top_k)
            rows.append({
                "index": i,
                "주요 원인": ", ".join(f"{k} {r[k]:+.1f}σ" for k in top.index),
            })
        return pd.DataFrame(rows).set_index("index")
