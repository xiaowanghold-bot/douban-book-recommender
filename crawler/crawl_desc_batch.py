# -*- coding: utf-8 -*-
"""批量爬取豆瓣图书简介 - BeautifulSoup版本"""
import requests, time, random, json, sys, os
from pathlib import Path
from bs4 import BeautifulSoup

BASE = Path(__file__).resolve().parent.parent
DESC_FILE = BASE / "data" / "processed" / "book_descriptions.json"
COVER_FILE = BASE / "data" / "processed" / "book_covers.json"
PROGRESS_FILE = BASE / "data" / "raw" / "desc_crawl_progress.txt"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

descs = json.load(open(DESC_FILE, encoding="utf-8")) if DESC_FILE.exists() else {}
covers = json.load(open(COVER_FILE, encoding="utf-8")) if COVER_FILE.exists() else {}

all_ids = [str(bid) for bid in covers.keys()]
needed = [bid for bid in all_ids if bid not in descs]

# 断点续传
if PROGRESS_FILE.exists():
    with open(PROGRESS_FILE, "r") as f:
        done_set = set(line.strip() for line in f)
    needed = [bid for bid in needed if bid not in done_set]

print(f"封面: {len(covers)} | 简介: {len(descs)} | 缺: {len(needed)}")
sys.stdout.flush()

if not needed:
    print("全部完成！")
    sys.exit(0)

SAVE_EVERY = 25
COOLDOWN_403 = 300
COOLDOWN_418 = 180
success = 0
fail = 0
no_intro = 0

session = requests.Session()
session.headers.update(HEADERS)

for i, bid_str in enumerate(needed):
    try:
        url = f"https://book.douban.com/subject/{bid_str}/"
        r = session.get(url, timeout=15)
        
        if r.status_code == 403:
            print(f"[{i+1}/{len(needed)}] 403 限流，等待{COOLDOWN_403}s...")
            time.sleep(COOLDOWN_403)
            r = session.get(url, timeout=15)
            if r.status_code != 200:
                print(f"  仍限流，保存后退出。成功{success}条")
                break
        
        if r.status_code == 418:
            print(f"[{i+1}/{len(needed)}] 418，等待{COOLDOWN_418}s...")
            time.sleep(COOLDOWN_418)
            continue
        
        if r.status_code != 200:
            fail += 1
            time.sleep(random.uniform(0.5, 1.5))
            continue
        
        soup = BeautifulSoup(r.text, "html.parser")
        intro_divs = soup.select("div.intro")
        
        found = False
        for div in intro_divs:
            # 取最后一个p标签（通常是内容简介而非作者简介）
            ps = div.find_all("p")
            for p in ps:
                text = p.get_text(strip=True)
                if len(text) > 20:
                    descs[bid_str] = text[:500]
                    success += 1
                    found = True
                    break
            if found:
                break
        
        if not found:
            no_intro += 1
        
        time.sleep(random.uniform(0.8, 2.0))
        
        if (i + 1) % SAVE_EVERY == 0:
            json.dump(descs, open(DESC_FILE, "w", encoding="utf-8"), ensure_ascii=False)
            with open(PROGRESS_FILE, "a") as pf:
                for j in range(i - SAVE_EVERY + 1, i + 1):
                    pf.write(needed[j] + "\n")
            print(f"  [保存] {i+1}/{len(needed)} | 新获:{success} | 无简介:{no_intro} | 总:{len(descs)}")
            sys.stdout.flush()
    
    except requests.exceptions.Timeout:
        fail += 1
        time.sleep(2)
    except Exception as e:
        fail += 1
        time.sleep(1)

# 最终保存
json.dump(descs, open(DESC_FILE, "w", encoding="utf-8"), ensure_ascii=False)
with open(PROGRESS_FILE, "w") as pf:
    for bid_str in all_ids:
        if bid_str in descs:
            pf.write(bid_str + "\n")

print(f"\n========== 完成 ==========")
print(f"新增:{success} | 无简介:{no_intro} | 失败:{fail} | 总:{len(descs)}")
