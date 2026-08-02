"""检查部署所需模型产物是否完整并符合 GitHub 单文件限制。"""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024
WARNING_FILE_BYTES = 50 * 1024 * 1024

REQUIRED_ASSETS = [
    "data/models/tfidf_matrix.npz",
    "data/models/nn_neighbors.pkl",
    "data/models/vectorizer.pkl",
    "data/models/books_for_rec.csv",
    "data/models/rating_predictor.pkl",
    "data/models/coldstart_meta.pkl",
    "data/models/coldstart_model.joblib",
    "data/models/coldstart_model_lower.joblib",
    "data/models/coldstart_model_upper.joblib",
]


def tracked_files():
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    return {path.decode("utf-8") for path in result.stdout.split(b"\0") if path}


def main():
    tracked = tracked_files()
    errors = []
    total_bytes = 0

    for relative_path in REQUIRED_ASSETS:
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            errors.append(f"缺少运行时产物: {relative_path}")
            continue
        if relative_path not in tracked:
            errors.append(f"运行时产物未被 Git 跟踪: {relative_path}")
        size = path.stat().st_size
        total_bytes += size
        if size >= MAX_GITHUB_FILE_BYTES:
            errors.append(f"超过 GitHub 100MB 单文件限制: {relative_path}")
        elif size >= WARNING_FILE_BYTES:
            print(f"警告: {relative_path} 为 {size / 1024 / 1024:.2f}MB，接近托管限制")

    print(f"已检查 {len(REQUIRED_ASSETS)} 个运行时产物，共 {total_bytes / 1024 / 1024:.2f}MB")
    if errors:
        for error in errors:
            print(f"错误: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
