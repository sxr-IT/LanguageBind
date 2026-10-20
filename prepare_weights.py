"""Download LanguageBind weights into a Hugging Face compatible cache.

Uses parallel byte ranges for large files when the default Hub downloader stalls.
"""

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests


MODELS = {
    "image": "LanguageBind_Image",
    "audio": "LanguageBind_Audio_FT",
    "video": "LanguageBind_Video_FT",
}
CHUNK_SIZE = 32 * 1024 * 1024


def download_large(url, target, size, workers):
    partial = target.with_name(target.name + ".part")
    markers = target.with_name(target.name + ".parts")
    markers.mkdir(exist_ok=True)
    count = (size + CHUNK_SIZE - 1) // CHUNK_SIZE
    if partial.exists() and not any(markers.iterdir()):
        completed = min(partial.stat().st_size // CHUNK_SIZE, count)
        for index in range(completed):
            (markers / str(index)).touch()
    with partial.open("r+b" if partial.exists() else "w+b") as output:
        output.truncate(size)

    def one_segment(index):
        if (markers / str(index)).exists():
            return index
        start = index * CHUNK_SIZE
        end = min(size - 1, start + CHUNK_SIZE - 1)
        for attempt in range(1, 5):
            try:
                position = start
                with partial.open("r+b") as output:
                    output.seek(start)
                    while position <= end:
                        sub_end = min(end, position + 8 * 1024 * 1024 - 1)
                        with requests.get(url, headers={"Range": f"bytes={position}-{sub_end}"}, stream=True, timeout=(20, 120)) as response:
                            response.raise_for_status()
                            if response.status_code != 206 or response.headers.get("content-range") != f"bytes {position}-{sub_end}/{size}":
                                raise IOError(f"unexpected range response: {response.status_code}, {response.headers.get('content-range')}")
                            for block in response.iter_content(chunk_size=1024 * 1024):
                                if block:
                                    output.write(block)
                                    position += len(block)
                        if position != sub_end + 1:
                            raise IOError(f"range {index} incomplete at byte {position}")
                (markers / str(index)).touch()
                return index
            except (requests.RequestException, IOError) as exc:
                print(f"range {index} attempt {attempt}/4 failed: {exc}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(attempt * 3)

    pending = [i for i in range(count) if not (markers / str(i)).exists()]
    print(f"{target.name}: {size / 1e6:.0f} MB, {len(pending)} ranges pending, {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one_segment, index) for index in pending]
        done = count - len(pending)
        for future in as_completed(futures):
            future.result()
            done += 1
            print(f"{target.name}: {done}/{count} ranges ({min(done * CHUNK_SIZE, size) / 1e6:.0f} MB)", flush=True)
    if len(list(markers.iterdir())) != count:
        raise IOError("download lacks completed range markers")
    partial.replace(target)
    for marker in markers.iterdir():
        marker.unlink()
    markers.rmdir()
    print(f"saved: {target} ({size / 1e6:.1f} MB)", flush=True)


def download_file(url, target, workers):
    if target.exists():
        print(f"cached: {target.name}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=(20, 30)) as probe:
        probe.raise_for_status()
        content_range = probe.headers.get("content-range", "")
        match = re.search(r"/(\d+)$", content_range)
        size = int(match.group(1)) if match else int(probe.headers.get("content-length", "0"))
    if size >= CHUNK_SIZE and match:
        download_large(url, target, size, workers)
    else:
        for attempt in range(1, 4):
            try:
                with requests.get(url, timeout=(20, 120)) as response:
                    response.raise_for_status()
                    content = response.content
                if size and len(content) != size:
                    raise IOError(f"incomplete file: {len(content)} of {size}")
                target.write_bytes(content)
                print(f"saved: {target.name} ({len(content) / 1e6:.1f} MB)", flush=True)
                return
            except (requests.RequestException, IOError) as exc:
                if attempt == 3:
                    raise
                print(f"attempt {attempt}/3 failed: {exc}", flush=True)
                time.sleep(attempt * 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities", nargs="+", choices=tuple(MODELS), default=["image", "audio"])
    parser.add_argument("--cache-dir", type=Path, default=Path("D:/codex/languagebind_models"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    with requests.Session() as session:
        for modality in args.modalities:
            model = MODELS[modality]
            repo = f"LanguageBind/{model}"
            info = session.get(f"https://huggingface.co/api/models/{repo}", timeout=30)
            info.raise_for_status()
            data = info.json()
            revision = data["sha"]
            base = args.cache_dir / f"models--LanguageBind--{model}"
            snapshot = base / "snapshots" / revision
            (base / "refs").mkdir(parents=True, exist_ok=True)
            (base / "refs" / "main").write_text(revision, encoding="utf-8")
            for item in data["siblings"]:
                name = item["rfilename"]
                if name in {".gitattributes", "README.md"}:
                    continue
                url = f"https://huggingface.co/{repo}/resolve/{revision}/{quote(name)}"
                download_file(url, snapshot / name, args.workers)


if __name__ == "__main__":
    main()


