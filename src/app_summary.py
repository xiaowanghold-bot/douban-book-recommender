"""生成 Streamlit 导航栏和首页使用的轻量统计快照。"""

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


def _read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_app_summary(data_dir=DATA_DIR):
    """从已生成的数据与模型产物汇总首页所需的稳定统计口径。"""
    data_dir = Path(data_dir)
    processed_dir = data_dir / "processed"
    models_dir = data_dir / "models"

    books = pd.read_csv(processed_dir / "books_scored.csv", encoding="utf-8-sig")
    details = pd.read_csv(data_dir / "raw" / "Books_detail.csv", encoding="utf-8-sig")
    descriptions = _read_json(processed_dir / "book_descriptions.json")
    covers = _read_json(processed_dir / "book_covers.json")
    model_meta = _read_json(models_dir / "model_meta.json")
    publishers = pd.read_csv(
        processed_dir / "publisher_stats.csv", encoding="utf-8-sig", index_col=0
    )
    authors = pd.read_csv(
        processed_dir / "author_stats.csv", encoding="utf-8-sig", index_col=0
    )

    successful_details = (
        int((details["crawl_status"] == "success").sum())
        if "crawl_status" in details
        else len(details)
    )
    return {
        "book_count": len(books),
        "average_rating": round(float(books["rating"].mean()), 4),
        "votes_1000_count": int((books["votes"] >= 1_000).sum()),
        "votes_10000_count": int((books["votes"] >= 10_000).sum()),
        "detail_count": len(details),
        "detail_success_count": successful_details,
        "description_count": len(descriptions),
        "cover_count": len(covers),
        "publisher_count": len(publishers),
        "author_count": len(authors),
        "recommendation_count": int(model_meta["n_books"]),
    }


def main():
    output_path = DATA_DIR / "processed" / "app_summary.json"
    summary = build_app_summary()
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"[保存] {output_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
