"""
공조기(AHU) 모의 데이터 생성기
==============================

**주의 — 이것은 성능 검증용이 아닙니다.**

    제가 물리 관계를 손으로 넣어 만든 가짜 데이터입니다.
    여기서 잘 맞는 것은 당연합니다 (제가 만든 규칙을 제가 찾는 것이므로).
    성능에 대한 근거는 오직 C-MAPSS 실험(1~4단계)에서만 나옵니다.

    이 데이터의 용도는 하나입니다:
    **파이프라인이 항공기 엔진이 아닌 설비에서도 그대로 도는지 확인하는 것.**
    컬럼 이름도, 센서 종류도, 시간 단위도 전부 다릅니다.

기간 설계 (여기서 한 번 데었습니다)
    처음에는 120일치만 만들었습니다. 그랬더니 **외기온과 경과시간의 상관이 0.86** 이 나왔습니다.
    1월에 시작해 4월에 끝나니 시간이 갈수록 더워지기만 한 것입니다.
    이러면 "더워진 것"과 "열화된 것"을 구분할 방법이 원리적으로 없습니다.
    게다가 기준선을 초기 구간에서 잡으니 **겨울 데이터만으로 기준을 만들고
    여름을 전부 이상이라고 판정**했습니다. 실제로 막힘이 시작되기도 전에 경보가 울렸습니다.

    그래서 2년치로 바꿨습니다.
        1년차 = 정상 (사계절을 모두 포함하는 기준선 구간)
        2년차 = 필터가 서서히 막혀감
    이것이 실제 현장에서도 맞는 설계입니다.
    **계절을 타는 설비는 기준선이 최소 1년을 덮어야 합니다.**

모사한 물리
    필터가 막힘 -> 차압 상승 -> 같은 풍량을 내려고 팬 회전수·전류 상승
    C-MAPSS 의 HPC 열화(압축기 효율 저하 -> 회전수·온도 상승)와 같은 구조입니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 실제 BAS 에서 흔히 보는 태그 이름을 그대로 씀
ID_COL = "설비번호"
TIME_COL = "일시"
SENSOR_COLS = ["급기온도", "환기온도", "필터차압", "팬전류", "팬회전수", "냉수밸브개도"]
CONDITION_COLS = ["외기온", "부하율"]

HOURS_PER_YEAR = 365 * 24


def make_ahu_data(
    n_units: int = 8,
    years: int = 2,
    healthy_years: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    공조기 n대의 운전 이력을 만든다.

    healthy_years : 앞의 몇 년을 정상 구간으로 둘지.
        기본 1년 — 사계절이 모두 들어가야 기준선이 제대로 잡힙니다.
    """
    rng = np.random.default_rng(seed)
    n = int(HOURS_PER_YEAR * years)
    t = np.arange(n)
    onset_floor = int(HOURS_PER_YEAR * healthy_years)

    frames = []
    for u in range(1, n_units + 1):
        # --- 운전조건: 연주기 + 일주기 (2년이면 계절이 두 번 돕니다) --------
        season = 12 * np.sin(2 * np.pi * t / HOURS_PER_YEAR - 1.2)
        daily = 5 * np.sin(2 * np.pi * t / 24 - 1.5)
        oat = 14 + season + daily + rng.normal(0, 1.2, n)
        load = np.clip(0.35 + 0.45 * (oat - 5) / 25 + rng.normal(0, 0.06, n), 0.1, 1.0)

        # --- 개체차: 설비마다 출발점이 다르다 (1단계에서 확인한 성질) ------
        off_dp = rng.normal(0, 12)
        off_cur = rng.normal(0, 0.5)
        off_sat = rng.normal(0, 0.4)

        # --- 열화: 필터 막힘. 정상 구간이 끝난 뒤 시작 ---------------------
        rate = rng.uniform(0.5, 1.5)
        onset = onset_floor + int(rng.uniform(0, 0.25) * HOURS_PER_YEAR)
        clog = np.zeros(n)
        if onset < n:
            prog = np.arange(n - onset) / max(n - onset, 1)
            clog[onset:] = (prog ** 1.8) * 60 * rate

        # --- 센서 (물리 관계를 넣는다) ------------------------------------
        dp = 120 + off_dp + 60 * load + clog + rng.normal(0, 3, n)
        rpm = 700 + 520 * load + 1.4 * clog + rng.normal(0, 8, n)
        cur = 4.0 + off_cur + 5.2 * load + 0.030 * clog + rng.normal(0, 0.12, n)
        sat = 14.0 + off_sat + 0.10 * (oat - 14) + 0.012 * clog + rng.normal(0, 0.25, n)
        rat = sat + 8.0 + 0.18 * (oat - 14) + rng.normal(0, 0.3, n)
        vlv = np.clip(20 + 62 * load + 0.05 * clog + rng.normal(0, 2, n), 0, 100)

        frames.append(pd.DataFrame({
            ID_COL: f"AHU-{u:02d}",
            TIME_COL: pd.date_range("2024-01-01", periods=n, freq="1h"),
            "외기온": oat.round(2),
            "부하율": load.round(3),
            "급기온도": sat.round(2),
            "환기온도": rat.round(2),
            "필터차압": dp.round(1),
            "팬전류": cur.round(2),
            "팬회전수": rpm.round(0),
            "냉수밸브개도": vlv.round(1),
            # 아래는 정답 — 평가용이며 모델에는 절대 넣지 않습니다
            "_막힘정도": clog.round(2),
        }))

    return pd.concat(frames, ignore_index=True)
