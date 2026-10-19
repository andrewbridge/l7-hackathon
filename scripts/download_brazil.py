#!/usr/bin/env python3
"""Politely download Brazil's e-SUS Notifica flu-syndrome notifications via the official open-data API.

Source: Ministério da Saúde, Portal de Dados Abertos do SUS
    https://apidadosabertos.saude.gov.br/vigilancia-e-meio-ambiente/notificacoes-de-sindrome-gripal-leve-<YEAR>

The bulk CSV files on dadosabertos.saude.gov.br return 403, so this pages through the API instead.

API notes (verified Oct 2026):
  * `limit` is capped at 1000.
  * `offset` is a ROW offset, despite the docs describing it as a page number.
  * An empty list means you've passed the end. Approximate sizes: 2020 ~135k rows,
    2021 ~188k, 2022 ~277k (so ~600 requests for all three).
  * No filtering parameters, so everything is downloaded and filtered later by the loader.

Politeness:
  * One request at a time (no concurrency).
  * A pause between requests (default 3s, +/- jitter), never shorter than the server's own last
    response time multiplied by --adaptive-factor. If the server slows down, we slow down too.
  * A longer rest every --rest-every pages.
  * Exponential backoff with full jitter on errors / 429 / 5xx, honouring Retry-After.
  * Gives up after --max-failures consecutive failures. Just rerun to resume.

Resumable: each page is written atomically to <out>/<year>/offset_<NNNNNNNN>.json.gz; existing pages are
skipped on rerun. A `_COMPLETE.json` marker is written once a year's final page is reached.

Usage:
    python scripts/download_brazil.py                       # 2020-2022 into data/brazil/
    python scripts/download_brazil.py --years 2020 --max-pages 10   # quick trial
    python scripts/download_brazil.py --delay 5 --rest-every 25     # gentler
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import signal
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

BASE_URL = "https://apidadosabertos.saude.gov.br/vigilancia-e-meio-ambiente/notificacoes-de-sindrome-gripal-leve-{year}"
RESULT_KEY = "notificacoes_sindrome_gripal_leve"
MAX_LIMIT = 1000
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
APPROX_ROWS = {2020: 135_000, 2021: 188_000, 2022: 277_000}

_stop_requested = False


def _request_stop(signum, frame):
    global _stop_requested
    if _stop_requested:  # second Ctrl-C: exit immediately
        sys.exit(130)
    _stop_requested = True
    log("Stop requested: finishing the current page, then exiting. Press Ctrl-C again to abort immediately.")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def polite_sleep(seconds: float) -> None:
    """Sleep in small steps so Ctrl-C stays responsive."""
    end = time.monotonic() + seconds
    while not _stop_requested and time.monotonic() < end:
        time.sleep(min(0.5, end - time.monotonic()))


def retry_after_seconds(err: urllib.error.HTTPError) -> float | None:
    value = err.headers.get("Retry-After") if err.headers else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, (parsedate_to_datetime(value) - datetime.now(parsedate_to_datetime(value).tzinfo)).total_seconds())
        except (TypeError, ValueError):
            return None


def fetch_page(year: int, offset: int, limit: int, user_agent: str, timeout: float) -> tuple[list[dict], float]:
    """Fetch one page. Returns (rows, elapsed_seconds). Raises on HTTP / network / parse errors."""
    url = f"{BASE_URL.format(year=year)}?limit={limit}&offset={offset}"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json",
                                               "Accept-Encoding": "gzip"})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
    elapsed = time.monotonic() - start
    payload = json.loads(body)
    if RESULT_KEY not in payload or not isinstance(payload[RESULT_KEY], list):
        raise ValueError(f"Unexpected response shape (keys: {list(payload)[:5]})")
    return payload[RESULT_KEY], elapsed


def write_page(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    tmp.replace(path)  # atomic: a half-written page never looks complete


def download_year(year: int, args: argparse.Namespace) -> str:
    """Download one year. Returns 'complete', 'stopped', 'max-pages' or 'failed'."""
    out_dir = args.out / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "_COMPLETE.json"
    if marker.exists() and not args.force:
        log(f"{year}: already complete ({json.loads(marker.read_text())['rows']:,} rows). Skipping.")
        return "complete"

    offset, pages_fetched, rows_total, failures = args.start_offset, 0, 0, 0
    delay, started = args.delay, time.monotonic()
    approx = APPROX_ROWS.get(year)

    while not _stop_requested:
        page_path = out_dir / f"offset_{offset:08d}.json.gz"
        if page_path.exists() and not args.force:  # resume: skip pages already on disk
            with gzip.open(page_path, "rt", encoding="utf-8") as f:
                n = len(json.load(f))
            rows_total += n
            if n < args.limit:
                break
            offset += args.limit
            continue

        if args.max_pages and pages_fetched >= args.max_pages:
            log(f"{year}: reached --max-pages={args.max_pages}. Rerun without it to continue.")
            return "max-pages"

        try:
            rows, elapsed = fetch_page(year, offset, args.limit, args.user_agent, args.timeout)
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, json.JSONDecodeError) as err:
            status = getattr(err, "code", None)
            if isinstance(err, urllib.error.HTTPError) and status not in RETRYABLE_STATUS:
                log(f"{year}: HTTP {status} at offset {offset} is not retryable. Stopping. ({err})")
                return "failed"
            failures += 1
            if failures > args.max_failures:
                log(f"{year}: {failures - 1} consecutive failures. Giving up for now; rerun later to resume.")
                return "failed"
            backoff = random.uniform(0, min(args.max_backoff, args.backoff_base * 2 ** (failures - 1)))
            if isinstance(err, urllib.error.HTTPError):
                backoff = max(backoff, retry_after_seconds(err) or 0)
            log(f"{year}: error at offset {offset} ({status or type(err).__name__}: {err}). "
                f"Retry {failures}/{args.max_failures} in {backoff:.0f}s.")
            polite_sleep(backoff)
            continue

        failures = 0
        write_page(page_path, rows)
        pages_fetched += 1
        rows_total += len(rows)

        if approx:
            done = min(offset + len(rows), approx)
            # Per-page cost includes the pause and the periodic rest, not just the request itself
            per_page = max((time.monotonic() - started) / pages_fetched, elapsed + args.delay)
            per_page += args.rest_seconds / args.rest_every if args.rest_every else 0
            remaining = max(0, (approx - done) / args.limit) * per_page
            progress = f"~{done / approx:.0%} of ~{approx:,}, ETA ~{timedelta(seconds=int(remaining))}"
        else:
            progress = "total unknown"
        log(f"{year}: offset {offset:>8,} -> {len(rows):>4} rows in {elapsed:.1f}s ({progress})")

        if len(rows) < args.limit:
            marker.write_text(json.dumps({"year": year, "rows": offset + len(rows),
                                          "finished": datetime.now().isoformat(timespec="seconds")}))
            log(f"{year}: complete, {offset + len(rows):,} rows.")
            return "complete"
        offset += args.limit

        # Adaptive pacing: never ask again faster than the server answered, plus jitter
        delay = max(args.delay, elapsed * args.adaptive_factor)
        pause = delay * random.uniform(1 - args.jitter, 1 + args.jitter)
        if args.rest_every and pages_fetched % args.rest_every == 0:
            pause += args.rest_seconds
            log(f"{year}: resting {pause:.0f}s after {pages_fetched} pages.")
        polite_sleep(pause)

    if _stop_requested:
        log(f"{year}: stopped at offset {offset:,}. Rerun the same command to resume.")
        return "stopped"
    # Loop exited via resume-skip reaching a short final page
    marker.write_text(json.dumps({"year": year, "rows": rows_total, "finished": datetime.now().isoformat(timespec="seconds")}))
    log(f"{year}: complete (from existing pages), {rows_total:,} rows.")
    return "complete"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022], help="Years to download")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "brazil",
                   help="Output directory")
    p.add_argument("--limit", type=int, default=MAX_LIMIT, help=f"Rows per request (max {MAX_LIMIT})")
    p.add_argument("--start-offset", type=int, default=0, help="Row offset to start from")
    p.add_argument("--max-pages", type=int, default=0, help="Stop after fetching this many new pages per year (0 = no limit)")
    p.add_argument("--delay", type=float, default=3.0, help="Minimum seconds between requests")
    p.add_argument("--jitter", type=float, default=0.3, help="Random +/- fraction applied to each pause")
    p.add_argument("--adaptive-factor", type=float, default=1.0,
                   help="Pause at least this multiple of the last response time")
    p.add_argument("--rest-every", type=int, default=50, help="Take a longer rest every N pages (0 = never)")
    p.add_argument("--rest-seconds", type=float, default=60.0, help="Length of the longer rest")
    p.add_argument("--backoff-base", type=float, default=10.0, help="Initial backoff after an error (seconds)")
    p.add_argument("--max-backoff", type=float, default=600.0, help="Cap on a single backoff (seconds)")
    p.add_argument("--max-failures", type=int, default=8, help="Consecutive failures before giving up")
    p.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout (seconds)")
    p.add_argument("--user-agent", default="l7-hackathon-research/1.0 (educational COVID-19 symptom study; polite sequential downloader)",
                   help="User-Agent header. Consider adding a contact address.")
    p.add_argument("--force", action="store_true", help="Re-download pages that already exist")
    args = p.parse_args(argv)
    if not 1 <= args.limit <= MAX_LIMIT:
        p.error(f"--limit must be between 1 and {MAX_LIMIT}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    log(f"Downloading years {args.years} to {args.out} (delay {args.delay}s, rest {args.rest_seconds:.0f}s "
        f"every {args.rest_every} pages). Ctrl-C to stop safely; rerun to resume.")
    results = {}
    for year in args.years:
        if _stop_requested:
            break
        results[year] = download_year(year, args)
        if results[year] == "failed":
            break
        if not _stop_requested and year != args.years[-1]:
            polite_sleep(args.rest_seconds)
    log(f"Done: {results}")
    return 0 if all(r == "complete" for r in results.values()) and len(results) == len(args.years) else 1


if __name__ == "__main__":
    sys.exit(main())
