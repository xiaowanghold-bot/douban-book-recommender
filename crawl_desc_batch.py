import requests, re, time, random, json, sys

DESC_FILE = "data/processed/book_descriptions.json"
COVER_FILE = "data/processed/book_covers.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

descs = json.load(open(DESC_FILE, encoding="utf-8"))
covers = json.load(open(COVER_FILE, encoding="utf-8"))
needed_ids = [bid for bid in covers if bid not in descs]
print("Total books:", len(covers))
print("Have descriptions:", len(descs))
print("Need descriptions:", len(needed_ids))
sys.stdout.flush()

# Test if Douban is accessible
test_url = "https://book.douban.com/subject/1770782/"
r = requests.get(test_url, headers=HEADERS, timeout=15)
if r.status_code != 200:
    print("Douban is blocked (status %d). Waiting 60s to retry..." % r.status_code)
    time.sleep(60)
    r = requests.get(test_url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        print("Still blocked. Please wait and run this script later.")
        print("Run: python crawl_desc_batch.py")
        sys.exit(1)

print("Douban accessible! Starting crawl...")
new_cnt = 0
save_interval = 50

for i, bid_str in enumerate(needed_ids[:500]):
    try:
        url = "https://book.douban.com/subject/" + bid_str + "/"
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            time.sleep(random.uniform(2, 5))
            continue
        
        html = r.text
        idx = html.find('class="intro"')
        if idx >= 0:
            snip = html[idx:idx+2000]
            m = re.search(r"<p>(.*?)</p>", snip, re.DOTALL)
            if m:
                text = re.sub(r"<[^>]+>", "", m.group(1)).strip()
                text = re.sub(r"\s+", " ", text)
                if len(text) > 15:
                    descs[bid_str] = text[:500]
                    new_cnt += 1
        
        time.sleep(random.uniform(1.5, 3.0))
        
        if i % save_interval == 0:
            json.dump(descs, open(DESC_FILE, "w", encoding="utf-8"), ensure_ascii=False)
            print("%d/%d processed, descs: %d (+%d)" % (i, len(needed_ids[:500]), len(descs), new_cnt))
            sys.stdout.flush()
    
    except KeyboardInterrupt:
        json.dump(descs, open(DESC_FILE, "w", encoding="utf-8"), ensure_ascii=False)
        print("Interrupted! Saved %d descriptions." % len(descs))
        sys.exit(0)
    except:
        time.sleep(1)

json.dump(descs, open(DESC_FILE, "w", encoding="utf-8"), ensure_ascii=False)
print("DONE: +%d new, total %d descriptions" % (new_cnt, len(descs)))
