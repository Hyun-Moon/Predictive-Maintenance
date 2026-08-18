"""
C-MAPSS 데이터 로더
===================

원본 txt 를 "바로 분석 가능한 pandas DataFrame" 으로 바꿔주는 모듈입니다.

원본 파일이 왜 그대로 쓰기 불편한가?
    - 컬럼명이 없습니다. 공백으로 구분된 숫자 26칸이 전부입니다.
    - 각 줄 끝에 공백이 2개 붙어 있어서, 그냥 읽으면 빈 컬럼 2개가 생깁니다.
    - RUL(잔여수명) 라벨이 train 파일에는 아예 없습니다. 직접 계산해야 합니다.

이 모듈이 그 세 가지를 대신 처리합니다.

용어 정리 (비전공자용):
    unit   : 설비 개체 1대. C-MAPSS 에서는 엔진 1기.
             (우리 회사로 치면 '공조기 3호기' 같은 개별 장비)
    cycle  : 그 설비가 몇 번째 운전 주기인지. 시간축이라고 보면 됩니다.
    RUL    : Remaining Useful Life, 잔여 유효 수명.
             "지금부터 몇 사이클 더 돌면 고장나는가"를 나타내는 숫자.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ROOT, load_config

# -----------------------------------------------------------------------------
# 컬럼 정의
# -----------------------------------------------------------------------------
ID_COL = "unit"
TIME_COL = "cycle"
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i:02d}" for i in range(1, 22)]
ALL_COLS = [ID_COL, TIME_COL, *OP_COLS, *SENSOR_COLS]

# C-MAPSS readme 기준 센서 물리적 의미 (EDA 해석과 README 설명에 사용)
SENSOR_MEANING = {
    "sensor_01": "팬 입구 온도 (T2)",
    "sensor_02": "LPC 출구 온도 (T24)",
    "sensor_03": "HPC 출구 온도 (T30)",
    "sensor_04": "LPT 출구 온도 (T50)",
    "sensor_05": "팬 입구 압력 (P2)",
    "sensor_06": "바이패스 덕트 압력 (P15)",
    "sensor_07": "HPC 출구 압력 (P30)",
    "sensor_08": "물리적 팬 회전수 (Nf)",
    "sensor_09": "물리적 코어 회전수 (Nc)",
    "sensor_10": "엔진 압력비 (epr)",
    "sensor_11": "HPC 출구 정압 (Ps30)",
    "sensor_12": "연료유량/Ps30 비 (phi)",
    "sensor_13": "보정 팬 회전수 (NRf)",
    "sensor_14": "보정 코어 회전수 (NRc)",
    "sensor_15": "바이패스 비 (BPR)",
    "sensor_16": "연소기 연료-공기비 (farB)",
    "sensor_17": "블리드 엔탈피 (htBleed)",
    "sensor_18": "요구 팬 회전수 (Nf_dmd)",
    "sensor_19": "요구 보정 팬 회전수 (PCNfR_dmd)",
    "sensor_20": "HPT 냉각 블리드 유량 (W31)",
    "sensor_21": "LPT 냉각 블리드 유량 (W32)",
}


# -----------------------------------------------------------------------------
# 경로
# -----------------------------------------------------------------------------
def cmapss_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return ROOT / cfg["cmapss"]["raw_dir"]


# -----------------------------------------------------------------------------
# 원본 읽기
# -----------------------------------------------------------------------------
def _read_raw(path: Path) -> pd.DataFrame:
    """
    공백 구분 txt 를 읽어 컬럼명을 붙인다.

    sep=r"\s+" : 공백이 하나든 여러 개든 모두 구분자로 취급 (줄 끝 공백 문제 해결)
    header=None: 첫 줄도 데이터임 (컬럼명 줄이 없으므로)
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

    # 혹시 빈 컬럼이 남으면 잘라낸다
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != len(ALL_COLS):
        raise ValueError(
            f"{path.name}: 컬럼 수가 {df.shape[1]}개입니다. {len(ALL_COLS)}개를 기대했습니다."
        )
    df.columns = ALL_COLS
    return df


# -----------------------------------------------------------------------------
# RUL 라벨 만들기
# -----------------------------------------------------------------------------
def add_train_rul(df: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """
    학습 데이터에 RUL 컬럼을 추가한다.

    train 파일은 "고장날 때까지" 기록되어 있습니다(run-to-failure).
    따라서 각 엔진의 마지막 사이클 = 고장 시점이고,
        RUL = (그 엔진의 최대 cycle) - (현재 cycle)
    로 계산하면 됩니다.

    cap 을 주면 그 값 위로는 잘라냅니다(예: 125).
    이유: 고장 300사이클 전이나 200사이클 전이나 현장에서는 똑같이 '정상'입니다.
          잘라주지 않으면 모델이 '아직 멀쩡한 구간'의 숫자를 맞추느라
          정작 중요한 '고장 임박 구간'을 소홀히 학습합니다.
    """
    df = df.copy()
    max_cycle = df.groupby(ID_COL)[TIME_COL].transform("max")
    df["RUL"] = max_cycle - df[TIME_COL]
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def add_test_rul(df: pd.DataFrame, rul_truth: pd.Series, cap: int | None = None) -> pd.DataFrame:
    """
    테스트 데이터에 RUL 컬럼을 추가한다.

    test 파일은 고장 '전에' 기록이 끊겨 있습니다.
    대신 RUL_FDxxx.txt 가 "각 엔진의 마지막 시점에서의 실제 잔여수명"을 알려줍니다.
    그래서
        RUL(현재) = (마지막 시점의 실제 RUL) + (마지막 cycle - 현재 cycle)
    이 됩니다.
    """
    df = df.copy()
    max_cycle = df.groupby(ID_COL)[TIME_COL].transform("max")
    # rul_truth 는 0-based 순서 -> unit 번호(1-based)에 맞춰 매핑
    truth_map = {i + 1: v for i, v in enumerate(rul_truth.tolist())}
    final_rul = df[ID_COL].map(truth_map)
    df["RUL"] = final_rul + (max_cycle - df[TIME_COL])
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df


# -----------------------------------------------------------------------------
# 공개 API
# -----------------------------------------------------------------------------
def load_cmapss(
    subset: str | None = None,
    cfg: dict | None = None,
    rul_cap: int | None = -1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    C-MAPSS 하위 세트 하나를 (train, test) DataFrame 으로 읽어온다.

    Parameters
    ----------
    subset : "FD001" ~ "FD004". None 이면 config.yaml 의 값 사용.
    rul_cap: RUL 상한. -1 이면 config.yaml 의 값 사용. None 이면 자르지 않음.

    Returns
    -------
    (train, test) : 둘 다 RUL 컬럼이 붙어 있음.
    """
    cfg = cfg or load_config()
    subset = subset or cfg["cmapss"]["subset"]
    if rul_cap == -1:
        rul_cap = cfg["cmapss"]["rul"]["cap"]

    d = cmapss_dir(cfg)
    train_path = d / f"train_{subset}.txt"
    test_path = d / f"test_{subset}.txt"
    rul_path = d / f"RUL_{subset}.txt"

    for p in (train_path, test_path, rul_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} 가 없습니다.\n"
                f"먼저 데이터를 내려받으세요:  python -m src.data.download --dataset cmapss"
            )

    train = add_train_rul(_read_raw(train_path), cap=rul_cap)

    rul_truth = pd.read_csv(rul_path, sep=r"\s+", header=None, engine="python").iloc[:, 0]
    test = add_test_rul(_read_raw(test_path), rul_truth, cap=rul_cap)

    return train, test


def describe(df: pd.DataFrame, name: str = "") -> str:
    """DataFrame 요약 한 줄. EDA 노트북/로그에서 쓰기 좋게."""
    return (
        f"{name:<8} rows={len(df):>7,}  units={df[ID_COL].nunique():>4}  "
        f"cycles/unit={df.groupby(ID_COL)[TIME_COL].max().mean():6.1f} (평균)  "
        f"cols={df.shape[1]}"
    )


if __name__ == "__main__":
    # 빠른 확인용:  python -m src.data.load
    cfg = load_config()
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        tr, te = load_cmapss(sub, cfg)
        print(f"[{sub}]")
        print("  " + describe(tr, "train"))
        print("  " + describe(te, "test"))
