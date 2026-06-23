"""
豆瓣图书详细信息爬虫
基于已有的图书ID，爬取作者、出版社、价格、ISBN、页数等元数据
"""
import requests
import re
import time
import random
import csv
import os
import sys
from datetime import datetime

# ========== 配置区 ==========
DATA_DIR = r"C:\Users\33672\Documents\New project1\data"
RAW_DATA = os.path.join(DATA_DIR, "raw", "Books_1.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "raw", "Books_detail.csv")
PROGRESS_FILE = os.path.join(DATA_DIR, "raw", "crawl_progress.txt")

MIN_VOTES = 2000          # 最少评价人数
START_INDEX = 0           # 从第几本开始（支持断点续爬）
MAX_BOOKS = None          # None = 全部, 或设置数量限制
DELAY_MIN = 1.5           # 最小延时（秒）
DELAY_MAX = 3.0           # 最大延时（秒）
REQUEST_TIMEOUT = 20      # 请求超时（秒）
MAX_RETRIES = 3           # 失败重试次数
SAVE_INTERVAL = 50        # 每爬N本保存一次

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# 需要提取的字段及其正则
FIELD_PATTERNS = {
    "author":        r'<span class="pl">\s*作者[：:]?</span>\s*(.*?)(?:<br\s*/?>|</span>)',
    "publisher":     r'<span class="pl">\s*出版社[：:]?</span>\s*(.*?)(?:<br\s*/?>|</span>)',
    "pub_year":      r'<span class="pl">\s*出版年[：:]?</span>\s*(.*?)(?:<br\s*/?>|</span>)',
    "pages":         r'<span class="pl">\s*页\s*数[：:]?</span>\s*(.*?)(?:<br\s*/?>|</span>)',
    "price":         r'<span class="pl">\s*定价[：:]?</span>\s*(.*?)(?:<br\s*/?>|</span>)',
    "binding":       r'<span class="pl">\s*装帧[：:]?</span>\s*(.*?)(?:<br\s*/?>|</span>)',
    "isbn":          r'<span class="pl">\s*ISBN[：:]?</span>\s*(.*?)(?:<br\s*/?>|</span>)',
    "original_title":r'<span class="pl">\s*原作名[：:]?</span>\s*(.*?)(?:<br\s*/?>|</span>)',
    "translator":    r'<span class="pl">\s*译者[：:]?</span>\s*(.*?)(?:<br\s*/?>|</span>)',
    "subtitle":      r'<span class="pl">\s*副标题[：:]?</span>\s*(.*?)(?:<br\s*/?>|</span>)',
    "series":        r'<span class="pl">\s*丛书[：:]?</span>\s*(.*?)(?:<br\s*/?>|</span>)',
}


def clean_html(text):
    """去除HTML标签和多余空白"""
    if not text:
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    # Remove leading colon/whitespace artifacts
    text = re.sub(r'^[：:\s]+', '', text)
    return text


def extract_info(html):
    """从HTML中提取图书详细信息"""
    info = {}
    
    # 提取info区块
    m = re.search(r'<div id="info".*?>(.+?)</div>', html, re.DOTALL)
    if not m:
        return info
    
    info_html = m.group(1)
    
    for field, pattern in FIELD_PATTERNS.items():
        m2 = re.search(pattern, info_html, re.DOTALL)
        if m2:
            val = clean_html(m2.group(1))
            info[field] = val
    
    return info


def load_book_ids(csv_path, min_votes):
    """加载需要爬取的图书ID列表，按评价人数降序排列"""
    import pandas as pd
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    
    # 筛选有效评分且评价人数 >= min_votes
    df_filtered = df[(df["Votes"] >= min_votes) & (df["Rating"] > 0)].copy()
    df_filtered = df_filtered.sort_values("Votes", ascending=False)
    
    print(f"待爬取图书: {len(df_filtered)} 本 (Votes >= {min_votes})")
    return df_filtered[["ID", "Rating", "Votes", "Title"]]


def load_progress():
    """读取断点续爬进度"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 0


def save_progress(index):
    """保存爬取进度"""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        f.write(str(index))


def append_to_csv(rows, is_first=False):
    """追加或创建CSV文件"""
    fieldnames = ["ID", "Rating", "Votes", "Title",
                  "author", "publisher", "pub_year", "pages", "price",
                  "binding", "isbn", "original_title", "translator",
                  "subtitle", "series", "crawl_status"]
    
    mode = "w" if is_first else "a"
    write_header = is_first
    
    with open(OUTPUT_FILE, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def crawl():
    """主爬取函数"""
    print("=" * 60)
    print("  豆瓣图书详细信息爬虫")
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 加载图书列表
    books = load_book_ids(RAW_DATA, MIN_VOTES)
    total = len(books)
    
    # 加载进度
    start = load_progress()
    if START_INDEX > 0:
        start = max(start, START_INDEX)
    
    if MAX_BOOKS:
        end = min(start + MAX_BOOKS, total)
    else:
        end = total
    
    print(f"爬取范围: 第 {start+1} ~ {end} 本 (共 {total} 本)")
    print(f"延时范围: {DELAY_MIN}s ~ {DELAY_MAX}s")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"进度文件: {PROGRESS_FILE}")
    print("-" * 60)
    
    batch = []
    success_count = 0
    fail_count = 0
    
    for i in range(start, end):
        row = books.iloc[i]
        book_id = int(row["ID"])
        title = row["Title"]
        
        url = f"https://book.douban.com/subject/{book_id}/"
        
        # 重试逻辑
        html = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    html = resp.text
                    break
                elif resp.status_code == 404:
                    break  # 页面不存在，不重试
                else:
                    time.sleep(2 * (attempt + 1))
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"  #{i+1}: ID={book_id} 请求失败: {e}")
                time.sleep(2 * (attempt + 1))
        
        # 提取信息
        if html:
            info = extract_info(html)
            status = "success"
            success_count += 1
        else:
            info = {}
            status = "404" if (resp and resp.status_code == 404) else "fail"
            fail_count += 1
        
        # 构建记录
        record = {
            "ID": book_id,
            "Rating": row["Rating"],
            "Votes": row["Votes"],
            "Title": title,
            "author": info.get("author", ""),
            "publisher": info.get("publisher", ""),
            "pub_year": info.get("pub_year", ""),
            "pages": info.get("pages", ""),
            "price": info.get("price", ""),
            "binding": info.get("binding", ""),
            "isbn": info.get("isbn", ""),
            "original_title": info.get("original_title", ""),
            "translator": info.get("translator", ""),
            "subtitle": info.get("subtitle", ""),
            "series": info.get("series", ""),
            "crawl_status": status,
        }
        batch.append(record)
        
        # 进度显示
        pct = (i - start + 1) / (end - start) * 100
        print(f"\r  [{pct:5.1f}%] {i+1}/{end} 成功:{success_count} 失败:{fail_count}   ", end="")
        
        # 定时保存
        if len(batch) >= SAVE_INTERVAL:
            is_first = (i - len(batch) + 1 == start) and not os.path.exists(OUTPUT_FILE)
            append_to_csv(batch, is_first=is_first)
            save_progress(i + 1)
            batch = []
        
        # 延时
        delay = random.uniform(DELAY_MIN, DELAY_MAX)
        time.sleep(delay)
    
    # 保存剩余数据
    if batch:
        is_first = (start == 0) and not os.path.exists(OUTPUT_FILE)
        append_to_csv(batch, is_first=is_first)
    
    save_progress(end)
    
    print(f"\n{'=' * 60}")
    print(f"  爬取完成!")
    print(f"  成功: {success_count}  失败: {fail_count}")
    print(f"  数据保存至: {OUTPUT_FILE}")
    print(f"  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    crawl()

