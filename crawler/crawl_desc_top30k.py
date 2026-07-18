# -*- coding: utf-8 -*-
"""
批量爬取豆瓣图书简介 + 标签 (Top 30K by votes)
=============================================
- 断点续爬，增量保存
- 随机UA池，3-6s随机延迟
- 连续失败5次暂停10分钟
- --limit N 控制单次运行量

用法: python crawler/crawl_desc_top30k.py --limit 50
"""
import argparse
import json
import os
import random
import re
import sys
import time
import tempfile
import shutil
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ============================================================
# 路径配置
# ============================================================
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

DESC_FILE = DATA_PROCESSED / "book_descriptions.json"
TAGS_FILE = DATA_PROCESSED / "book_tags.json"
PROGRESS_FILE = DATA_RAW / "desc30k_progress.txt"
FAILED_FILE = DATA_RAW / "desc30k_failed.txt"
BOOKS_SCORED = DATA_PROCESSED / "books_scored.csv"

SAVE_EVERY = 200          # 每200本保存一次
COOLDOWN_FAIL = 600       # 连续失败5次后暂停10分钟
MAX_CONSECUTIVE_FAILS = 5
DELAY_MIN = 3.0            # 请求间隔 3-6 秒
DELAY_MAX = 6.0

# ============================================================
# UA 池
# ============================================================
UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0",
]


def random_headers():
    return {
        "User-Agent": random.choice(UA_POOL),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Cache-Control": "max-age=0",
    }


def atomic_write_json(path, data):
    """原子写入 JSON：先写临时文件再替换，避免中断损坏。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    shutil.move(str(tmp), str(path))


def load_json(path, default=None):
    if default is None:
        default = {}
    if path.exists():
        try:
            return json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            print(f"  [WARN] {path.name} 损坏，从空开始")
    return default


def load_progress(path):
    """加载进度文件，返回已完成的 ID 集合。"""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_progress(path, ids):
    """覆盖写入进度文件。"""
    with open(path, "w", encoding="utf-8") as f:
        for bid in sorted(ids):
            f.write(str(bid) + "\n")


def extract_tags_from_html(soup):
    """
    从页面 HTML 提取真实豆瓣标签。
    只提取 /tag/ 链接文本，不使用 jieba 等伪造标签。
    宁缺毋假——抓不到就不写。
    """
    tags = []
    seen = set()
    for a in soup.find_all("a"):
        href = a.get("href", "")
        text = a.get_text(strip=True)
        if "/tag/" in href and len(text) >= 2 and text not in seen:
            seen.add(text)
            tags.append(text)
    return tags[:15]


def scrape_book(session, bid_str, delay_range=(DELAY_MIN, DELAY_MAX)):
    """抓取单本书的简介和标签。
    返回 (status, intro_text, tags_list, redirect_to).
    status: "OK" | "NOT_FOUND" | "RATE_LIMITED" | "ERROR"
    redirect_to: 重定向目标 ID 或 None"""
    url = f"https://book.douban.com/subject/{bid_str}/"
    
    time.sleep(random.uniform(*delay_range))
    
    try:
        r = session.get(url, timeout=20, headers=random_headers(), allow_redirects=True)
    except requests.exceptions.Timeout:
        return "ERROR", None, [], None
    except requests.exceptions.ConnectionError:
        return "ERROR", None, [], None
    
    if r.status_code == 403:
        return "RATE_LIMITED", None, [], None
    if r.status_code == 404:
        return "NOT_FOUND", None, [], None
    if r.status_code != 200:
        return "ERROR", None, [], None
    
    # 检查是否被重定向到不同 ID
    redirect_to = None
    final_url = r.url
    m = re.search(r"/subject/(\d+)/", final_url)
    if m and m.group(1) != bid_str:
        redirect_to = m.group(1)
    
    try:
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return "ERROR", None, [], None
    
    # 检查页面是否有效
    title_tag = soup.find("title")
    if title_tag and "条目不存在" in title_tag.get_text():
        return "NOT_FOUND", None, [], None
    if len(soup.find_all("div")) < 30:
        return "NOT_FOUND", None, [], None
    
    # 提取简介
    intro_text = ""
    intro_divs = soup.select("div.intro")
    for div in intro_divs:
        ps = div.find_all("p")
        for p in ps:
            text = p.get_text(strip=True)
            if len(text) > 20:
                intro_text = text[:800]
                break
        if intro_text:
            break
    
    if not intro_text:
        return "ERROR", None, [], None
    
    # 提取标签（仅从 /tag/ 链接提取，宁缺毋假）
    tags = extract_tags_from_html(soup)
    
    return "OK", intro_text, tags, redirect_to

def get_target_ids(limit=None):
    """获取待爬取的目标 ID 列表。"""
    import pandas as pd
    df = pd.read_csv(BOOKS_SCORED, encoding="utf-8-sig")
    
    # 按 votes 降序取前 30,000
    top30k = df.nlargest(30000, "votes")
    target_ids = [str(int(i)) for i in top30k["id"]]
    
    # 排除已有简介
    existing_descs = load_json(DESC_FILE)
    target_ids = [bid for bid in target_ids if bid not in existing_descs]
    
    # 排除已完成进度
    done_set = load_progress(PROGRESS_FILE)
    target_ids = [bid for bid in target_ids if bid not in done_set]
    
    if limit and limit < len(target_ids):
        target_ids = target_ids[:limit]
    
    return target_ids


def main():
    parser = argparse.ArgumentParser(description="豆瓣图书简介+标签批量爬虫 (Top 30K)")
    parser.add_argument("--limit", type=int, default=None,
                        help="单次爬取数量上限（用于测试或分批运行）")
    parser.add_argument("--min-success", type=int, default=None,
                        help="成功抓取N条简介后自动停止（用于冒烟测试）")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  豆瓣图书简介+标签爬虫 (Top 30K by votes)")
    print("=" * 60)
    
    # 加载已有数据
    descs = load_json(DESC_FILE)
    tags_data = load_json(TAGS_FILE)
    done_set = load_progress(PROGRESS_FILE)
    failed_set = load_progress(FAILED_FILE)
    
    # 获取目标ID
    target_ids = get_target_ids(limit=args.limit)
    print(f"已有简介: {len(descs)} | 已有标签: {len(tags_data)} | 已完成: {len(done_set)}")
    print(f"目标: {len(target_ids)} 本{' (limit={})'.format(args.limit) if args.limit else ''}")
    
    if not target_ids:
        print("全部完成！")
        return
    
    # 创建session
    session = requests.Session()
    session.headers.update(random_headers())
    
    success_desc = 0
    success_tags = 0
    not_found = 0
    redirected = 0
    consecutive_fails = 0
    total_fails = 0
    total_requests = 0
    
    # 重定向映射
    REDIRECT_FILE = DATA_PROCESSED / "id_redirects.json"
    redirects_map = load_json(REDIRECT_FILE)
    
    # 成功样例收集（用于最终报告）
    success_samples = []
    
    for i, bid_str in enumerate(target_ids):
        # --min-success 提前终止
        if args.min_success and success_desc >= args.min_success:
            print(f"\n[达成] 已抓取 {success_desc} 条新简介，停止。")
            break
        
        # 速率限制冷却检查
        if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
            print(f"\n[冷却] 连续失败{consecutive_fails}次，暂停{COOLDOWN_FAIL//60}分钟...")
            sys.stdout.flush()
            time.sleep(COOLDOWN_FAIL)
            consecutive_fails = 0
            session = requests.Session()
        
        total_requests += 1
        print(f"[{total_requests}] ID={bid_str} ", end="", flush=True)
        
        status, intro, tags, redirect_to = scrape_book(session, bid_str)
        
        if status == "RATE_LIMITED":
            print("[WARN] 403限流")
            consecutive_fails += 1
            total_fails += 1
            failed_set.add(bid_str)
            time.sleep(30)
            continue
        
        if status == "NOT_FOUND":
            print("[FAIL] 死链")
            not_found += 1
            consecutive_fails = 0
            done_set.add(bid_str)
            continue
        
        if status == "ERROR":
            total_fails += 1
            consecutive_fails += 1
            print(f"[NO] 失败 (累计{consecutive_fails})")
            continue
        
        # status == "OK"
        # 处理重定向
        if redirect_to and redirect_to != bid_str:
            redirects_map[bid_str] = redirect_to
            redirected += 1
            print(f" -> {redirect_to} ", end="", flush=True)
        
        # 保存简介（用旧 id 做 key）
        descs[bid_str] = intro
        success_desc += 1
        consecutive_fails = 0
        
        # 保存标签（宁缺毋假，只有真实 /tag/ 标签才写）
        if tags:
            tags_data[bid_str] = tags
            success_tags += 1
        
        done_set.add(bid_str)
        
        # 记录样例
        title_text = ""
        try:
            import pandas as pd
            df_lookup = pd.read_csv(BOOKS_SCORED, encoding="utf-8-sig")
            match = df_lookup[df_lookup["id"].astype(str) == bid_str]
            if not match.empty:
                title_text = str(match.iloc[0]["title"])[:40]
        except Exception:
            pass
        
        success_samples.append({
            "id": bid_str,
            "title": title_text,
            "intro": intro[:80],
            "tags": tags,
            "redirected": redirect_to is not None,
        })
        
        tag_preview = tags[:3] if tags else []
        print(f"[OK] {title_text[:20]} | {len(intro)}字 | 标签:{tag_preview}")
        
        # 增量保存
        if success_desc % SAVE_EVERY == 0:
            atomic_write_json(DESC_FILE, descs)
            atomic_write_json(TAGS_FILE, tags_data)
            atomic_write_json(REDIRECT_FILE, redirects_map)
            save_progress(PROGRESS_FILE, done_set)
            save_progress(FAILED_FILE, failed_set)
            print(f"  [保存] {success_desc}条新简介 | 标签:{success_tags} | 死链:{not_found} | 失败:{total_fails}")
            sys.stdout.flush()
    
    # 最终保存
    atomic_write_json(DESC_FILE, descs)
    atomic_write_json(TAGS_FILE, tags_data)
    atomic_write_json(REDIRECT_FILE, redirects_map)
    save_progress(PROGRESS_FILE, done_set)
    save_progress(FAILED_FILE, failed_set)
    
    print(f"\n{'='*60}")
    print(f"  本次统计")
    print(f"  请求数: {total_requests} | 成功: {success_desc} | 重定向: {redirected}")
    print(f"  死链: {not_found} | 失败: {total_fails}")
    print(f"  标签命中: {success_tags}/{success_desc} ({100*success_tags//max(1,success_desc)}%)")
    print(f"  累计简介: {len(descs)} | 累计标签: {len(tags_data)} | 累计重定向: {len(redirects_map)}")
    print(f"{'='*60}")
    
    # 输出成功样例
    if success_samples:
        print(f"\n--- 新简介样例 (共{len(success_samples)}条) ---")
        for s in success_samples:
            tag_str = ", ".join(s["tags"][:5]) if s["tags"] else "(无标签)"
            redirect_note = f" [重定向]" if s["redirected"] else ""
            print(f"  ID={s['id']} {s['title']}{redirect_note}")
            print(f"    简介: {s['intro']}...")
            print(f"    标签: {tag_str}")
        
        # 标签统计
        tagged_books = [s for s in success_samples if s["tags"]]
        all_tags = []
        for s in tagged_books:
            all_tags.extend(s["tags"])
        from collections import Counter
        tag_counts = Counter(all_tags)
        if tag_counts:
            print(f"\n  标签统计: {len(tagged_books)}/{len(success_samples)} 本有标签")
            print(f"  热门标签: {tag_counts.most_common(10)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
