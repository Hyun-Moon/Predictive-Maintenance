"""
보고서용 프로그램 구성 흐름도 생성
==================================

실행:  python scripts/06_report_diagram.py

분석 결과가 아니라 **설명용 그림**입니다.
"이 프로그램이 무엇을 입력받아 무엇을 내보내는가"를 한 장으로 보여주기 위한 것으로,
보고서 앞부분에 배치합니다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.viz import style
from src.viz.style import CAT, GRID, INK, INK_SUB, MUTED, SURFACE, save


def box(ax, x, y, w, h, title, lines, accent, fill="#FFFFFF"):
    """제목 + 항목 목록을 담은 상자."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.3, edgecolor=accent, facecolor=fill, zorder=2))
    # 제목 띠
    ax.add_patch(FancyBboxPatch(
        (x, y + h - 0.075), w, 0.075,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0, facecolor=accent, zorder=3))
    ax.text(x + w / 2, y + h - 0.037, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=4)
    for i, t in enumerate(lines):
        ax.text(x + 0.022, y + h - 0.125 - i * 0.058, t, ha="left", va="center",
                fontsize=9.3, color=INK, zorder=4)


def arrow(ax, x1, y, x2):
    ax.add_patch(FancyArrowPatch(
        (x1, y), (x2, y), arrowstyle="-|>,head_width=5,head_length=9",
        linewidth=1.8, color=INK_SUB, zorder=5, mutation_scale=1))


def main() -> None:
    style.setup()
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    W, H, Y = 0.27, 0.60, 0.20

    box(ax, 0.015, Y, W, H, "입    력",
        ["· 설비 번호 / 계측 시각",
         "· 온도 · 압력 · 유량",
         "· 전류 · 회전수 · 차압",
         "· 외기온 · 부하율",
         "  (운전조건)",
         "",
         "BAS / BEMS 에서",
         "이미 수집 중인 값"],
        CAT[0])

    box(ax, 0.365, Y, W, H, "처    리",
        ["① 운전조건 식별",
         "② 조건별 정규화",
         "③ 설비별 정상 기준선 학습",
         "④ 기준선 대비 이탈도 산출",
         "⑤ 지속 조건 확인 후 경보",
         "",
         "정상 데이터만 사용",
         "고장 이력 불필요"],
        CAT[2])

    box(ax, 0.715, Y, W, H, "출    력",
        ["· 설비별 건강도 (0~100)",
         "· 경보 발령 여부",
         "· 원인 계측 항목 순위",
         "",
         "예) AHU-06  건강도 21",
         "    팬회전수 +0.8σ",
         "    필터차압 +0.5σ",
         "  판정: 필터 막힘 의심"],
        CAT[1])

    arrow(ax, 0.295, Y + H / 2, 0.355)
    arrow(ax, 0.645, Y + H / 2, 0.705)

    ax.text(0.5, 0.10,
            "고장 이력이 없어도 동작 — 각 설비의 초기 정상 구간만으로 기준선을 학습",
            ha="center", va="center", fontsize=9.5, color=INK_SUB)

    ax.text(0.5, 0.955, "설비 건강도 감시 프로그램 구성",
            ha="center", va="center", fontsize=13, fontweight="bold", color=INK)

    save(fig, "20_program_flow.png")


if __name__ == "__main__":
    main()
