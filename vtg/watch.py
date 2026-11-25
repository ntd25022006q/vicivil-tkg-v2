"""watch: P2 realtime ingestion loop for ViCivil-TKG v2.
Polls congbao.chinhphu.vn (official gazette — same channel proven by law_crawl.py v1),
hash-diffs a persistent registry, stages deltas for new/changed documents,
mines amendment facts from extractable text, and NEVER mutates the published
graph registries directly (integration is explicit, via vtg.run_all / merge step).

Usage:
    python3 -m vtg.watch --once [--dry-run] [--pages 2] [--max-docs 5] [--download]
    python3 -m vtg.watch --status
Cron:  23 */6 * * * cd ~/vicivil_tkg && python3 -m vtg.watch --once >> watch.log 2>&1
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

from .textops import (_RE_ARTICLE_AMEND, _AMEND_WORDS, fold_text,
                      law_id_from_folded, _RE_DIEU)

BASE = "https://congbao.chinhphu.vn"
UA = {"User-Agent": "Mozilla/5.0 Windows NT 10.0 Win64 x64 (academic research; rate-limited)"}
DELAY = 1.2
HOME = os.path.expanduser("~/vicivil_tkg")
REG_PATH = os.path.join(HOME, "watch_registry.json")
DELTA_DIR = os.path.join(HOME, "delta")
LOCK = os.path.join(HOME, ".watch.lock")


def get(url, binary=False, tries=2, timeout=40):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read() if binary else r.read().decode("utf-8", "ignore")
        except Exception as e:
            print(f"  retry {a} {url[:70]} {repr(e)[:70]}", flush=True)
            time.sleep(3 + 4 * a)
    return None


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------ registry
def load_registry():
    try:
        return json.load(open(REG_PATH, encoding="utf-8"))
    except Exception:
        return {"issues": {}, "docs": {}, "last_run": None, "runs": 0}


def save_registry(reg):
    json.dump(reg, open(REG_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ text extraction
def extract_text_from_pdf(path):
    """Best-effort PDF text: pypdf / PyPDF2 / pdfminer / pdftotext — else None."""
    try:
        import pypdf
        r = pypdf.PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in r.pages[:60])
    except Exception:
        pass
    try:
        import PyPDF2
        r = PyPDF2.PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in r.pages[:60])
    except Exception:
        pass
    try:
        from pdfminer.high_level import extract_text
        return extract_text(path, maxpages=60)
    except Exception:
        pass
    try:
        out = subprocess.run(["pdftotext", path, "-"], capture_output=True, text=True, timeout=120)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
    except Exception:
        pass
    return None


def extract_text_from_doc(path):
    try:
        import olefile
        ole = olefile.OleFileIO(path)
        raw = ole.openstream("WordDocument").read().decode("utf-16-le", errors="ignore")
        return "\n".join(p.strip() for p in re.split(r"[\r\n\x07]+", raw) if len(p.strip()) > 2)
    except Exception:
        return None


# ------------------------------------------------------------------ mining
def mine_amendments(text, doc_key):
    """Article-level amendment facts + cited laws from extracted text (ASCII-folded)."""
    ftext = fold_text(text)
    facts = []
    for m in _RE_ARTICLE_AMEND.finditer(ftext):
        w0, w1 = max(0, m.start() - 150), min(len(ftext), m.end() + 150)
        if not _AMEND_WORDS.search(ftext[w0:w1]):
            continue
        amender = law_id_from_folded(m.group(2))
        facts.append({"type": "amend_fact", "doc_key": doc_key,
                      "amender_text": m.group(2)[:120],
                      "amender_law_id": amender,
                      "amender_dieu": m.group(1)})
    return facts


def mine_citations(text, doc_key):
    ftext = fold_text(text)
    out = []
    for m in _RE_DIEU.finditer(ftext[:20000]):
        before = ftext[max(0, m.start() - 300):m.start()]
        lid = law_id_from_folded(before)
        if lid:
            out.append({"type": "cites", "doc_key": doc_key,
                        "tgt": f"{lid}|Dieu{m.group(1)}"})
    return list({json.dumps(x, sort_keys=True): x for x in out}.values())[:30]


# ------------------------------------------------------------------ poll cycle
def poll_cycle(reg, pages=2, max_docs=5, download=False, dry_run=False, log=print):
    ts = time.strftime("%Y%m%dT%H%M%S")
    changes = {"new_issues": 0, "new_docs": 0, "changed_docs": 0, "amend_facts": 0,
               "cites": 0, "pdf_hashed": 0, "text_extracted": 0}
    delta = []

    # ---- 1. listing pages -> issue paths + direct doc paths
    issue_paths = set()
    doc_paths_direct = set()
    for pg in range(1, pages + 1):
        h = get(f"{BASE}/cong-bao/trang-{pg}.htm")
        if not h:
            log(f"[poll] listing page {pg} FAILED (network/authority?)")
            continue
        for m in re.findall(r"/cong-bao/(cong-bao-so-\d+-ngay-\d+-\d+-\d+(?:-\d+)?)\.htm", h):
            issue_paths.add(f"/cong-bao/{m}.htm")
        # 11/2026 layout: listing embeds doc links directly, flat form /van-ban/<slug>-<id>.htm
        for dpath in re.findall(r"(/van-ban/[a-z0-9\-]+-\d+\.htm)", h):
            doc_paths_direct.add(dpath)
        time.sleep(DELAY)
    if not issue_paths and not doc_paths_direct:
        log("[poll] no issue/doc paths from listing — aborting cycle (no registry write)")
        return changes, delta

    new_issues = [p for p in sorted(issue_paths) if p not in reg["issues"]]
    log(f"[poll] listing issues seen={len(issue_paths)} new={len(new_issues)}")
    changes["new_issues"] = len(new_issues)
    # register ALL seen issues immediately (details fetched lazily on later cycles)
    for p in new_issues:
        n = re.search(r"cong-bao-so-(\d+)-ngay-(\d+)-(\d+)-(\d+)", p)
        reg["issues"][p] = {
            "so": n.group(1) if n else None,
            "ngay": f"{n.group(4)}-{n.group(3)}-{n.group(2)}" if n else None,
            "first_seen": ts, "docs": [], "detail_fetched": False,
        }

    # ---- 2. fetch details of up to `pages` newest issues (docs inside)
    doc_paths = set(doc_paths_direct)
    for p in new_issues[:pages]:
        h = get(BASE + p)
        time.sleep(DELAY)
        if not h:
            continue
        doc_links = set(re.findall(r"(/van-ban/[a-z0-9\-]+/\d+\.htm)", h)) | \
                    set(re.findall(r"(/van-ban/[a-z0-9\-]+-\d+\.htm)", h))
        reg["issues"][p]["docs"] = sorted(doc_links)
        reg["issues"][p]["detail_fetched"] = True
        delta.append({"type": "new_issue_detail", "issue": p,
                      "so": reg["issues"][p]["so"], "ngay": reg["issues"][p]["ngay"],
                      "n_docs": len(doc_links)})
        doc_paths.update(doc_links)

    # ---- 3. new docs — cap per cycle
    fresh = [p for p in sorted(doc_paths) if p not in reg["docs"]][:max_docs]
    log(f"[poll] new docs this cycle={len(fresh)} (cap={max_docs})")
    for du in fresh:
        h = get(BASE + du)
        time.sleep(DELAY)
        if not h:
            continue
        tmatch = re.search(r"<title>([^<]{5,200})</title>", h)
        title = tmatch.group(1).strip() if tmatch else du
        # so_hieu: title carries proper diacritics — match RAW title first,
        # fallback to folded page text with folded literals (lesson: đ/U+0111 + tone stacks)
        tm = re.search(r"(Luật số|Bộ luật số|Nghị định số|Nghị quyết số|Thông tư số|Quyết định số)\s*:?\s*([0-9/\-A-ZĐNQHTT]+)",
                       title)
        if tm:
            so_hieu = (tm.group(1) + " " + tm.group(2)).strip()
        else:
            sm = re.search(r"(luat so|bo luat so|nghi dinh so|nghi quyet so|thong tu so|quyet dinh so)\s*:?\s*([0-9/a-z\-]+)",
                           fold_text(re.sub(r"<[^>]+>", " ", h)))
            so_hieu = (sm.group(1) + " " + sm.group(2)) if sm else None
        pdfs = sorted(set(re.findall(r"https?://[^\"' ]+\.pdf[^\"' ]*", h)))[:3]
        entry = {"title": title, "so_hieu": so_hieu, "pdf_url": pdfs[0] if pdfs else None,
                 "first_seen": ts, "last_seen": ts, "status": "registered"}
        # stable change-signal hash (NOT whole HTML — pages churn with dynamic junk)
        entry["content_hash"] = sha(f"{title}|{so_hieu}|{entry['pdf_url']}".encode("utf-8"))
        reg["docs"][du] = entry
        changes["new_docs"] += 1
        delta.append({"type": "new_doc", "doc": du, "so_hieu": so_hieu,
                      "title": title[:160], "pdf_url": entry["pdf_url"]})

        # ---- 4. text -> amendment mining (only if download enabled and pdf available)
        if download and entry["pdf_url"]:
            raw = get(entry["pdf_url"], binary=True, tries=2, timeout=120)
            if raw and len(raw) > 1000:
                entry["pdf_hash"] = sha(raw)
                changes["pdf_hashed"] += 1
                ext = ".pdf" if raw[:4] == b"%PDF" else ".doc"
                tmp = os.path.join("/tmp", f"vtg_{ts}_{abs(hash(du))}{ext}")
                open(tmp, "wb").write(raw)
                text = (extract_text_from_pdf(tmp) if ext == ".pdf"
                        else extract_text_from_doc(tmp))
                os.unlink(tmp)
                if text and len(text) > 500:
                    entry["status"] = "text_extracted"
                    entry["text_chars"] = len(text)
                    changes["text_extracted"] += 1
                    facts = mine_amendments(text, du)
                    cites = mine_citations(text, du)
                    delta.extend(facts)
                    delta.extend(cites)
                    changes["amend_facts"] += len(facts)
                    changes["cites"] += len(cites)
                else:
                    entry["status"] = "pdf_no_text"  # scanned or extraction failed
        # changed-doc detection: same doc re-registered with different page hash
        prev = reg["docs"].get(du, {}).get("content_hash")

    # ---- 5. changed-doc sweep (stable hash: title + so_hieu + pdf_url)
    for du, e in list(reg["docs"].items())[:200]:
        if e.get("status") in ("registered", "text_extracted") and du not in fresh:
            h = get(BASE + du, tries=1)
            time.sleep(DELAY)
            if not h:
                continue
            tmatch = re.search(r"<title>([^<]{5,200})</title>", h)
            title = tmatch.group(1).strip() if tmatch else du
            hsh = sha(f"{title}|{e.get('so_hieu')}|{e.get('pdf_url')}".encode("utf-8"))
            if hsh != e.get("content_hash"):
                e["prev_hash"] = e.get("content_hash")
                e["content_hash"] = hsh
                e["last_seen"] = ts
                changes["changed_docs"] += 1
                delta.append({"type": "changed_doc", "doc": du,
                              "so_hieu": e.get("so_hieu")})

    return changes, delta


# ------------------------------------------------------------------ lock
class Lock:
    def __enter__(self):
        if os.path.exists(LOCK):
            age = time.time() - os.path.getmtime(LOCK)
            if age > 7200:
                os.unlink(LOCK)  # stale
            else:
                raise RuntimeError("another watch cycle is running (lock present)")
        os.makedirs(HOME, exist_ok=True)
        open(LOCK, "w").write(str(os.getpid()))

    def __exit__(self, *a):
        try:
            os.unlink(LOCK)
        except OSError:
            pass


# ------------------------------------------------------------------ CLI
def main():
    argv = sys.argv[1:]
    if "--status" in argv:
        reg = load_registry()
        print(json.dumps({"issues": len(reg["issues"]), "docs": len(reg["docs"]),
                          "last_run": reg.get("last_run"), "runs": reg.get("runs", 0),
                          "deltas": sorted(os.listdir(DELTA_DIR)) if os.path.isdir(DELTA_DIR) else []},
                         ensure_ascii=False, indent=1))
        return
    once = "--once" in argv or not argv
    dry = "--dry-run" in argv
    pages = int(argv[argv.index("--pages") + 1]) if "--pages" in argv else 2
    max_docs = int(argv[argv.index("--max-docs") + 1]) if "--max-docs" in argv else 5
    download = "--download" in argv

    with Lock():
        reg = load_registry()
        t0 = time.time()
        changes, delta = poll_cycle(reg, pages=pages, max_docs=max_docs,
                                    download=download, dry_run=dry)
        if not dry:
            os.makedirs(DELTA_DIR, exist_ok=True)
            dpath = os.path.join(DELTA_DIR, f"delta_{time.strftime('%Y%m%dT%H%M%S')}.jsonl")
            with open(dpath, "w", encoding="utf-8") as f:
                for d in delta:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
            reg["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            reg["runs"] = reg.get("runs", 0) + 1
            save_registry(reg)
            print(f"[watch] delta written: {dpath} ({len(delta)} records)")
        else:
            print(f"[watch] DRY-RUN — no registry/delta writes ({len(delta)} would-be records)")
        print(f"[watch] changes={changes} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
