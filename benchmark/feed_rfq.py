#!/usr/bin/env python3
"""Feed the questionnaire corpus (docs-rfq.jsonl) into the local Vespa rfq schema.

The schema derives every embedding at index time — three nomic fields, a
cross-encoder token tensor and a ColBERT token matrix — so feeding is the slow
part of a run, not the querying. Use --workers to parallelise it.

  python3 feed_rfq.py --purge          # replace the corpus wholesale
  python3 feed_rfq.py --workers 8      # add/overwrite by id
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
ENDPOINT = "http://127.0.0.1:8080"
CLUSTER = "content"
TS = 1750000000  # fixed epoch: freshness is not a variable in this corpus


def put(doc):
    body = {"fields": {
        "id": doc["id"],
        "tenant": doc["tenant"],
        "doc_type": doc["doc_type"],
        "question_id": doc["question_id"],
        "control_id": doc["control_id"],
        "title": doc["title"],
        "text": doc["text"],
        "chunks": [doc["text"]],
        "created_timestamp": TS,
        "modified_timestamp": TS,
        "last_opened_timestamp": TS,
        "open_count": 0,
        "favorite": False,
    }}
    req = urllib.request.Request(
        f"{ENDPOINT}/document/v1/rfq/rfq/docid/{urllib.parse.quote(doc['id'], safe='')}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                r.read()
                return None
        except urllib.error.HTTPError as e:
            return f"{doc['id']}: HTTP {e.code} {e.read()[:200].decode(errors='ignore')}"
        except Exception as e:
            if attempt == 3:
                return f"{doc['id']}: {e}"
            time.sleep(2 * (attempt + 1))
    return f"{doc['id']}: exhausted retries"


def purge():
    url = (f"{ENDPOINT}/document/v1/rfq/rfq/docid?selection=true"
           f"&cluster={CLUSTER}&timeChunk=60")
    removed = 0
    while True:
        req = urllib.request.Request(url, method="DELETE")
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
        removed += d.get("documentCount", 0)
        cont = d.get("continuation")
        if not cont:
            break
        url = url.split("&continuation=")[0] + f"&continuation={cont}"
    print(f"purged {removed} existing rfq docs")


def count():
    q = (f"{ENDPOINT}/search/?yql=" + urllib.parse.quote("select * from rfq where true")
         + "&hits=0")
    with urllib.request.urlopen(q, timeout=60) as r:
        return json.load(r)["root"]["fields"]["totalCount"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=str(HERE / "docs-rfq.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--purge", action="store_true", help="delete all rfq docs first")
    args = ap.parse_args()

    docs = [json.loads(l) for l in open(args.docs)]
    print(f"feeding {len(docs)} docs from {Path(args.docs).name}")
    if args.purge:
        purge()

    t0 = time.time()
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, err in enumerate(ex.map(put, docs), start=1):
            if err:
                errors.append(err)
            if i % 100 == 0:
                print(f"  {i}/{len(docs)}  ({time.time()-t0:.0f}s)")
    print(f"fed {len(docs)-len(errors)}/{len(docs)} in {time.time()-t0:.0f}s")
    for e in errors[:10]:
        print("  !", e, file=sys.stderr)
    time.sleep(3)
    print(f"rfq docs now in index: {count()}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
