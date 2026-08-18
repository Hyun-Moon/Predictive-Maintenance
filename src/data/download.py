"""
데이터셋 다운로드 스크립트
=========================

사용법 (프로젝트 루트에서):
    python -m src.data.download --dataset cmapss
    python -m src.data.download --dataset metropt3
    python -m src.data.download --dataset all
    python -m src.data.download --dataset cmapss --check-only   # 이미 받았는지 확인만

왜 스크립트로 만드나?
    원본 데이터는 git 에 올리지 않기 때문에, 이 저장소를 clone 한 사람이
    "명령어 한 줄"로 동일한 데이터를 갖출 수 있어야 재현 가능한 프로젝트가 됩니다.
    (포트폴리오에서 채용담당자가 가장 먼저 보는 것이 '재현 가능한가'입니다)

주의:
    사내망/방화벽 환경에서는 자동 다운로드가 막힐 수 있습니다.
    그럴 때는 --help 에 안내된 수동 다운로드 절차를 따르세요.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import requests
from tqdm import tqdm

# 프로젝트 루트 = 이 파일 기준 두 단계 위 (src/data/download.py -> 루트)
ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# 데이터셋 정의
# =============================================================================
@dataclass
class DatasetSpec:
    """데이터셋 하나에 대한 다운로드 명세."""

    key: str
    title: str
    # 여러 미러를 순서대로 시도합니다. 앞의 것이 막히면 다음 것으로 넘어감.
    mirrors: list[str]
    dest_dir: Path
    archive_name: str
    # 압축을 푼 뒤 반드시 존재해야 하는 파일들 (다운로드 성공 검증용)
    expected_files: list[str] = field(default_factory=list)
    # 압축 해제 후 폴더 이름을 이걸로 통일한다.
    # (원본 zip 은 "6. Turbofan Engine Degradation Simulation Data Set" 처럼
    #  공백과 점이 섞인 이름이라 코드에서 다루기 불편합니다)
    canonical_subdir: str = ""
    manual_url: str = ""
    manual_note: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "cmapss": DatasetSpec(
        key="cmapss",
        title="NASA C-MAPSS Turbofan Engine Degradation Simulation",
        mirrors=[
            # PHM Society 가 운영하는 NASA PCoE 공식 미러 (가장 안정적)
            "https://data.phmsociety.org/wp-content/uploads/sites/9/2023/07/"
            "6.-Turbofan-Engine-Degradation-Simulation-Data-Set.zip",
            # NASA 가 사용하는 S3 버킷
            "https://phm-datasets.s3.amazonaws.com/NASA/"
            "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
        ],
        dest_dir=ROOT / "data" / "raw" / "cmapss",
        archive_name="cmapss.zip",
        expected_files=[
            "train_FD001.txt",
            "test_FD001.txt",
            "RUL_FD001.txt",
            "train_FD004.txt",
        ],
        canonical_subdir="CMAPSSData",
        manual_url="https://www.kaggle.com/datasets/behrad3d/nasa-cmaps",
        manual_note=(
            "Kaggle 에서 'NASA Turbofan Jet Engine Data Set' 을 내려받아\n"
            "  압축 안의 CMAPSSData 폴더를 data/raw/cmapss/CMAPSSData/ 로 옮기세요.\n"
            "  (train_FD001.txt 가 data/raw/cmapss/CMAPSSData/train_FD001.txt 에 오면 성공)"
        ),
    ),
    "metropt3": DatasetSpec(
        key="metropt3",
        title="MetroPT-3 (지하철 공기압축기 APU 실측 데이터)",
        mirrors=[
            "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip",
        ],
        dest_dir=ROOT / "data" / "raw" / "metropt3",
        archive_name="metropt3.zip",
        expected_files=["MetroPT3(AirCompressor).csv"],
        manual_url="https://archive.ics.uci.edu/dataset/791/metropt+3+dataset",
        manual_note=(
            "UCI 페이지의 'Download' 버튼으로 zip 을 받아\n"
            "  data/raw/metropt3/ 에 풀어주세요. (약 500MB CSV 1개)"
        ),
    ),
}


# =============================================================================
# 유틸
# =============================================================================
def _human(n: int) -> str:
    """바이트 수를 사람이 읽기 좋은 단위로."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def find_expected(spec: DatasetSpec) -> dict[str, Path | None]:
    """
    기대 파일들이 dest_dir 어딘가에 존재하는지 재귀적으로 찾는다.

    압축 파일마다 내부 폴더 구조가 조금씩 달라서(예: CMAPSSData/ 가 한 겹 더 있거나 없거나)
    '정확한 경로'가 아니라 '파일 이름'으로 찾는 편이 훨씬 안정적입니다.
    """
    found: dict[str, Path | None] = {}
    for name in spec.expected_files:
        hits = list(spec.dest_dir.rglob(name)) if spec.dest_dir.exists() else []
        found[name] = hits[0] if hits else None
    return found


def is_ready(spec: DatasetSpec) -> bool:
    return all(p is not None for p in find_expected(spec).values())


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """파일 무결성 확인용 해시. 다운로드가 중간에 끊겼는지 판별에 씁니다."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# =============================================================================
# 다운로드
# =============================================================================
def download_file(url: str, dest: Path, timeout: int = 60) -> bool:
    """단일 URL 다운로드. 성공하면 True."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with tmp.open("wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.replace(dest)
        return True
    except Exception as exc:  # noqa: BLE001 - 어떤 실패든 다음 미러로 넘어가야 함
        print(f"    x 실패: {type(exc).__name__}: {exc}")
        if tmp.exists():
            tmp.unlink()
        return False


def extract(archive: Path, dest_dir: Path) -> None:
    """zip 을 풀되, 내부에 zip 이 또 있으면 한 겹 더 푼다 (C-MAPSS 가 이런 구조)."""
    print(f"    - 압축 해제: {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)

    for inner in list(dest_dir.rglob("*.zip")):
        if inner.resolve() == archive.resolve():
            continue
        print(f"    - 내부 압축 해제: {inner.name}")
        try:
            with zipfile.ZipFile(inner) as zf:
                zf.extractall(inner.parent)
            inner.unlink()
        except zipfile.BadZipFile:
            print(f"      (건너뜀: {inner.name} 은 정상 zip 이 아님)")


def normalize_layout(spec: DatasetSpec) -> None:
    """
    압축 해제 결과 폴더 이름을 spec.canonical_subdir 로 통일한다.

    예) data/raw/cmapss/6. Turbofan Engine Degradation Simulation Data Set/train_FD001.txt
     -> data/raw/cmapss/CMAPSSData/train_FD001.txt

    이렇게 해두면 config.yaml 의 경로가 항상 맞아떨어지고,
    Kaggle 로 수동 다운로드한 사람과도 폴더 구조가 같아집니다.
    """
    if not spec.canonical_subdir:
        return

    target = spec.dest_dir / spec.canonical_subdir
    if target.exists():
        return

    # 기대 파일이 실제로 들어있는 폴더를 찾는다.
    anchor = spec.expected_files[0] if spec.expected_files else None
    if anchor is None:
        return
    hits = list(spec.dest_dir.rglob(anchor))
    if not hits:
        return

    src = hits[0].parent
    if src == spec.dest_dir:
        # 파일이 곧바로 dest_dir 에 풀린 경우 -> 하위 폴더를 만들어 옮긴다
        target.mkdir(parents=True, exist_ok=True)
        for item in list(spec.dest_dir.iterdir()):
            if item != target:
                item.rename(target / item.name)
    else:
        src.rename(target)
    print(f"    - 폴더 정리: {target.relative_to(ROOT)}")


def print_manual_instructions(spec: DatasetSpec) -> None:
    print()
    print("  " + "-" * 68)
    print(f"  [수동 다운로드 안내] {spec.title}")
    print("  " + "-" * 68)
    print(f"  1) 아래 주소를 브라우저에서 엽니다:")
    print(f"     {spec.manual_url}")
    print(f"  2) {spec.manual_note}")
    print(f"  3) 다시 확인:")
    print(f"     python -m src.data.download --dataset {spec.key} --check-only")
    print("  " + "-" * 68)


def fetch(spec: DatasetSpec, force: bool = False) -> bool:
    print(f"\n[{spec.key}] {spec.title}")

    if not force and is_ready(spec):
        print("  v 이미 준비되어 있습니다. (건너뜀)")
        return True

    archive = spec.dest_dir / spec.archive_name
    ok = False
    for i, url in enumerate(spec.mirrors, 1):
        print(f"  - 미러 {i}/{len(spec.mirrors)} 시도: {url[:78]}...")
        if download_file(url, archive):
            ok = True
            break

    if not ok:
        print("  x 모든 미러에서 다운로드에 실패했습니다.")
        print("    (사내망 방화벽 / 프록시 차단이 가장 흔한 원인입니다)")
        print_manual_instructions(spec)
        return False

    print(f"    v 내려받음: {_human(archive.stat().st_size)}"
          f"  sha256={sha256_of(archive)[:16]}...")
    extract(archive, spec.dest_dir)
    archive.unlink(missing_ok=True)
    normalize_layout(spec)
    return verify(spec)


# =============================================================================
# 검증
# =============================================================================
def verify(spec: DatasetSpec) -> bool:
    print(f"  - 파일 검증 ({spec.dest_dir.relative_to(ROOT)})")
    found = find_expected(spec)
    all_ok = True
    for name, path in found.items():
        if path is None:
            print(f"      x {name}  (없음)")
            all_ok = False
        else:
            size = _human(path.stat().st_size)
            rel = path.relative_to(ROOT)
            print(f"      v {name}  {size:>9}  {rel}")

    if all_ok:
        print("  v 준비 완료.")
    else:
        print("  x 일부 파일이 없습니다.")
        print_manual_instructions(spec)
    return all_ok


# =============================================================================
# main
# =============================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="예지보전 프로젝트용 공개 데이터셋 다운로더",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS.keys(), "all"],
        default="cmapss",
        help="받을 데이터셋 (기본: cmapss)",
    )
    parser.add_argument("--force", action="store_true", help="이미 있어도 다시 받기")
    parser.add_argument(
        "--check-only", action="store_true", help="다운로드 없이 존재 여부만 확인"
    )
    args = parser.parse_args(argv)

    keys = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    results = {}
    for key in keys:
        spec = DATASETS[key]
        if args.check_only:
            print(f"\n[{spec.key}] {spec.title}")
            results[key] = verify(spec)
        else:
            results[key] = fetch(spec, force=args.force)

    print("\n" + "=" * 72)
    for key, ok in results.items():
        print(f"  {'v 준비완료' if ok else 'x 미완료  '}  {key}")
    print("=" * 72)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
