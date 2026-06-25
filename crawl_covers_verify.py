import requests, re, time, random, json, os, sys
from datetime import datetime

COVER_DIR = "app/covers"
COVER_FILE = "data/processed/book_covers.json"
DESC_FILE = "data/processed/book_descriptions.json"
VERIFIED_FILE = "data/processed/verified_covers.json"

os.makedirs(COVER_DIR, exist_ok=True)
covers = json.load(open(COVER_FILE, encoding="utf-8"))
descs = json.load(open(DESC_FILE, encoding="utf-8"))

if os.path.exists(VERIFIED_FILE):
    verified = set(json.load(open(VERIFIED_FILE, encoding="utf-8")))
else:
    verified = set()
print("Already verified: " + str(len(verified)))

targets = [bid for bid in descs if int(bid) not in verified]
print("Total targets: " + str(len(targets)))

BATCH = 500
batch_targets = targets[:BATCH]
print("This batch: " + str(len(batch_targets)))
sys.stdout.flush()

CDN_NODES = ["img1", "img2", "img3", "img9"]

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "zh-CN,zh;q=0.9",
})

new_cnt = 0
errors = 0
no_cover = 0

def try_download_image(session, cover_url, referer):
    """Try downloading from multiple CDN nodes"""
    for node in CDN_NODES:
        try:
            url = re.sub(r'img\d+', node, cover_url)
            session.headers["Referer"] = referer
            r = session.get(url, timeout=20)
            if r.status_code == 200 and len(r.content) > 5000:
                # Verify it's actually an image
                ct = r.headers.get("content-type", "")
                if "image" in ct or len(r.content) > 10000:
                    return r.content, url
        except:
            continue
    return None, None

for i, bid_str in enumerate(batch_targets):
    bid = int(bid_str)
    try:
        url = "https://book.douban.com/subject/" + str(bid) + "/"
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            if r.status_code == 404:
                no_cover += 1
            else:
                errors += 1
                if errors <= 5:
                    print("  [" + str(bid) + "] HTTP " + str(r.status_code))
            time.sleep(random.uniform(0.5, 1.0))
            continue

        html = r.text
        cover_url = None

        m = re.search(r'<a class="nbg"[^>]*href="(https://img\d+\.doubanio\.com/[^"]+)"', html)
        if m:
            cover_url = m.group(1)

        if not cover_url:
            no_cover += 1
            if no_cover <= 3:
                print("  [" + str(bid) + "] No cover found")
            time.sleep(random.uniform(0.5, 1.0))
            continue

        # Try multiple CDN nodes
        img_data, final_url = try_download_image(session, cover_url, url)
        if img_data:
            ext = "jpg"
            if ".png" in final_url:
                ext = "png"
            elif ".webp" in final_url:
                ext = "webp"
            fname = str(bid) + "." + ext
            fpath = os.path.join(COVER_DIR, fname)
            with open(fpath, "wb") as f:
                f.write(img_data)
            covers[bid_str] = fname
            verified.add(bid)
            new_cnt += 1

    except KeyboardInterrupt:
        print("\nInterrupted at " + str(i) + "/" + str(len(batch_targets)))
        break
    except Exception as e:
        errors += 1
        if errors <= 10:
            print("  [" + str(bid) + "] Error: " + str(e))
        time.sleep(random.uniform(0.5, 1.0))
        continue

    time.sleep(random.uniform(1.0, 2.0))

    if (i + 1) % 50 == 0:
        ts = datetime.now().strftime("%H:%M:%S")
        print("[" + ts + "] " + str(i+1) + "/" + str(len(batch_targets)) + " | +" + str(new_cnt) + " verified | " + str(errors) + " err | " + str(no_cover) + " no-cover")
        sys.stdout.flush()
        json.dump(covers, open(COVER_FILE, "w", encoding="utf-8"), ensure_ascii=False)
        json.dump(sorted(list(verified)), open(VERIFIED_FILE, "w", encoding="utf-8"), ensure_ascii=False)

json.dump(covers, open(COVER_FILE, "w", encoding="utf-8"), ensure_ascii=False)
json.dump(sorted(list(verified)), open(VERIFIED_FILE, "w", encoding="utf-8"), ensure_ascii=False)

print("\n=== BATCH DONE ===")
print("New verified: +" + str(new_cnt))
print("Total verified: " + str(len(verified)))
print("Errors: " + str(errors))
print("No cover: " + str(no_cover))
print("Remaining: " + str(len(targets) - len(batch_targets)))
