"""
설정 로더
=========

config/config.yaml 을 읽어서 파이썬 dict 로 돌려줍니다.

왜 이런 파일이 필요한가?
    노트북마다 경로를 하드코딩하면 나중에 폴더 하나 옮길 때 전부 고쳐야 합니다.
    설정을 한 곳에 모아두면 "회사 설비 데이터로 갈아끼우기"가 훨씬 쉬워집니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """config.yaml 을 읽어 dict 로 반환."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = ROOT
    return cfg


def resolve(cfg: dict[str, Any], relative_path: str) -> Path:
    """config 안의 상대경로를 프로젝트 루트 기준 절대경로로 바꿔줍니다."""
    return ROOT / relative_path
