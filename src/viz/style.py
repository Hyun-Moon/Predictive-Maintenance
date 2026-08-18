"""
그림 스타일 공통 설정
====================

모든 그림이 같은 색/같은 톤을 쓰도록 한 곳에 모아둡니다.
노트북마다 색을 다르게 쓰면 보고서가 산만해 보입니다.

색 선택 원칙 (데이터 시각화 일반 원칙):
    - 범주형(서로 다른 항목): 정해진 순서대로 색을 씁니다. 색을 돌려쓰지 않습니다.
    - 순차형(크기의 많고 적음): 한 가지 색의 밝기만 바꿉니다. 무지개색 금지.
    - 발산형(양/음, 상승/하락): 따뜻한 색 <-> 차가운 색 + 중간은 회색.
    - 강조: 나머지를 회색으로 죽이고 하나만 색을 줍니다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# 한글 폰트 설정 (설치: pip install koreanize-matplotlib)
try:
    import koreanize_matplotlib  # noqa: F401  (import 만으로 폰트가 적용됨)

    KOREAN_FONT = True
except ImportError:  # 폰트가 없으면 한글이 네모(□)로 나옵니다
    KOREAN_FONT = False

from src.config import ROOT

# --- 색 팔레트 ---------------------------------------------------------------
SURFACE = "#fcfcfb"        # 그림 배경 (GitHub 라이트/다크 양쪽에서 읽히도록 밝은 고정색)
INK = "#0b0b0b"            # 주요 글자
INK_SUB = "#52514e"        # 보조 글자
GRID = "#e6e5e1"           # 격자선 (배경에서 한 톤만 벗어나게)

# 범주형: 이 순서대로만 사용
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# 발산형: 파랑 <-> 빨강, 중간은 회색
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#d8d7d3", "#e34948"

# 강조용 회색 (배경으로 물러나는 선들)
MUTED = "#b8b7b2"

FIG_DIR = ROOT / "reports" / "figures"


def setup() -> None:
    """matplotlib 전역 스타일 적용. 그림 그리기 전에 한 번 호출."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            # 축과 격자는 배경으로 물러나게 (실선 hairline, 점선 금지)
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            # 글자
            "text.color": INK,
            "axes.labelcolor": INK_SUB,
            "xtick.color": INK_SUB,
            "ytick.color": INK_SUB,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            # 선
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
        }
    )


def save(fig: plt.Figure, name: str) -> Path:
    """그림을 reports/figures/ 에 저장하고 경로를 출력한다."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  저장: {path.relative_to(ROOT)}")
    return path
