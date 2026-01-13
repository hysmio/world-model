#!/usr/bin/env python3
"""download VPT minecraft dataset - https://github.com/openai/Video-Pre-Training"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

VPT_INDEX_URL = "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_10xx_Jun_29.json"
STATE_FILE = "download_state.json"
VPT_FPS = 20


@dataclass
class DownloadState:
    """persistent download state - saves to json for resume"""

    completed: dict[str, dict] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    in_progress: set[str] = field(default_factory=set)
    index_total: int = 0
    total_bytes_downloaded: int = 0
    total_frames: int = 0
    started_at: str | None = None
    last_updated: str | None = None

    @classmethod
    def load(cls, path: Path) -> "DownloadState":
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                data["in_progress"] = set(data.get("in_progress", []))
                return cls(**data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"warning: could not load state, starting fresh: {e}")
        return cls()

    def save(self, path: Path) -> None:
        self.last_updated = datetime.now().isoformat()
        data = asdict(self)
        data["in_progress"] = list(self.in_progress)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def total_hours(self) -> float:
        return (self.total_frames / VPT_FPS) / 3600 if self.total_frames else 0

    @property
    def total_gb(self) -> float:
        return self.total_bytes_downloaded / (1024**3)


class ProgressTracker:
    """thread-safe progress tracking, saves state periodically"""

    def __init__(self, total: int, state: DownloadState, state_path: Path):
        self.total = total
        self.state = state
        self.state_path = state_path
        self.lock = Lock()
        self.completed_count = 0
        self.failed_count = 0
        self.start_time = time.time()
        self.last_save_time = time.time()
        self.save_interval = 10

    def update(self, basename: str, success: bool, size: int = 0,
               frames: int = 0, error: str | None = None) -> None:
        with self.lock:
            if success:
                self.completed_count += 1
                self.state.completed[basename] = {
                    "size": size,
                    "timestamp": datetime.now().isoformat(),
                    "duration_frames": frames,
                }
                self.state.total_bytes_downloaded += size
                self.state.total_frames += frames
                if basename in self.state.failed:
                    del self.state.failed[basename]
            else:
                self.failed_count += 1
                self.state.failed[basename] = error or "unknown error"

            self.state.in_progress.discard(basename)

            if time.time() - self.last_save_time > self.save_interval:
                self.state.save(self.state_path)
                self.last_save_time = time.time()

    def mark_in_progress(self, basename: str) -> None:
        with self.lock:
            self.state.in_progress.add(basename)

    def get_progress_line(self, basename: str, status: str) -> str:
        with self.lock:
            done = self.completed_count + self.failed_count
            pct = done / self.total * 100 if self.total else 0
            elapsed = time.time() - self.start_time

            if done > 0:
                eta_seconds = (elapsed / done) * (self.total - done)
                eta = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta = "..."

            return f"[{done}/{self.total}] ({pct:.1f}%) ETA: {eta} | {basename}: {status}"

    def finalize(self) -> None:
        with self.lock:
            self.state.in_progress.clear()
            self.state.save(self.state_path)


def download_file(url: str, dest: Path, chunk_size: int = 65536) -> tuple[bool, int]:
    """returns (success, file_size)"""
    try:
        req = Request(url, headers={"User-Agent": "VPT-Downloader/1.0"})
        with urlopen(req, timeout=60) as response:
            expected = response.headers.get("content-length")
            expected = int(expected) if expected else None

            with open(dest, "wb") as f:
                downloaded = 0
                while chunk := response.read(chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)

            if expected and downloaded != expected:
                dest.unlink(missing_ok=True)
                return False, 0

        return True, downloaded
    except (URLError, HTTPError, TimeoutError, OSError):
        dest.unlink(missing_ok=True)
        return False, 0


def download_with_retries(url: str, dest: Path, max_retries: int = 3) -> tuple[bool, int]:
    for attempt in range(max_retries):
        success, size = download_file(url, dest)
        if success:
            return True, size
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    return False, 0


def fetch_index(url: str) -> tuple[str, list[str]]:
    """returns (basedir, relpaths)"""
    print(f"fetching index from {url}...")
    try:
        with urlopen(url, timeout=60) as response:
            data = json.loads(response.read().decode())
            return data.get("basedir", ""), data.get("relpaths", [])
    except Exception as e:
        print(f"error fetching index: {e}")
        sys.exit(1)


def get_frame_count_from_actions(action_path: Path) -> int:
    """count lines in jsonl file"""
    if not action_path.exists():
        return 0
    try:
        with open(action_path) as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def download_video_pair(
    basedir: str,
    relpath: str,
    output_dir: Path,
    state: DownloadState,
    tracker: ProgressTracker | None = None,
    download_actions: bool = True,
) -> tuple[bool, str, int, int]:
    """returns (success, path_or_error, size, frames)"""
    basename = Path(relpath).stem
    if not basename:
        return False, "missing basename", 0, 0

    # skip if already done
    if basename in state.completed:
        info = state.completed[basename]
        return True, "already downloaded", info.get("size", 0), info.get("duration_frames", 0)

    video_url = f"{basedir}{relpath}"
    action_url = video_url.replace(".mp4", ".jsonl")
    video_path = output_dir / f"{basename}.mp4"
    action_path = output_dir / f"{basename}.jsonl"

    # recover from interrupted run
    if video_path.exists():
        return True, str(video_path), video_path.stat().st_size, get_frame_count_from_actions(action_path)

    if tracker:
        tracker.mark_in_progress(basename)

    success, size = download_with_retries(video_url, video_path)
    if not success:
        return False, f"failed: {video_url}", 0, 0

    frames = 0
    if download_actions:
        if download_with_retries(action_url, action_path)[0]:
            frames = get_frame_count_from_actions(action_path)

    return True, str(video_path), size, frames


def download_dataset(
    output_dir: Path,
    max_videos: int | None = None,
    num_workers: int = 4,
    download_actions: bool = True,
    retry_failed: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / STATE_FILE

    state = DownloadState.load(state_path)
    if not state.started_at:
        state.started_at = datetime.now().isoformat()

    basedir, relpaths = fetch_index(VPT_INDEX_URL)
    state.index_total = len(relpaths)

    print(f"found {len(relpaths)} videos in index")
    print(f"already completed: {len(state.completed)}")
    print(f"previously failed: {len(state.failed)}")

    def get_basename(rp: str) -> str:
        return Path(rp).stem

    if not retry_failed:
        relpaths = [rp for rp in relpaths if get_basename(rp) not in state.completed]
    else:
        failed = set(state.failed.keys())
        relpaths = [rp for rp in relpaths if get_basename(rp) not in state.completed or get_basename(rp) in failed]

    print(f"remaining: {len(relpaths)}")

    if max_videos:
        relpaths = relpaths[:max_videos]
        print(f"limiting to {max_videos} this session")

    if not relpaths:
        print("\nnothing to download!")
        print_status(output_dir)
        return []

    tracker = ProgressTracker(len(relpaths), state, state_path)
    print(f"\ndownloading {len(relpaths)} videos with {num_workers} workers...")
    print("-" * 60)

    downloaded = []
    session_failed = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_video_pair, basedir, rp, output_dir, state, tracker, download_actions): rp
            for rp in relpaths
        }

        for future in as_completed(futures):
            rp = futures[future]
            basename = get_basename(rp)
            success, result, size, frames = future.result()

            if success:
                if "already" not in result:
                    downloaded.append(Path(result))
                tracker.update(basename, True, size=size, frames=frames)
                status = "OK"
            else:
                session_failed.append(basename)
                tracker.update(basename, False, error=result)
                status = "FAIL"

            print(tracker.get_progress_line(basename, status))

    tracker.finalize()

    print("\n" + "=" * 60)
    print(f"downloaded: {len(downloaded)}, failed: {len(session_failed)}")
    print_status(output_dir)

    return downloaded


def print_status(output_dir: Path) -> None:
    """show download status"""
    state_path = output_dir / STATE_FILE
    state = DownloadState.load(state_path)

    print("\n" + "=" * 60)
    print("DOWNLOAD STATUS")
    print("=" * 60)

    print(f"\nIndex total:        {state.index_total:,} videos")
    print(f"Completed:          {len(state.completed):,} videos")
    print(f"Failed:             {len(state.failed):,} videos")
    remaining = state.index_total - len(state.completed)
    print(f"Remaining:          {remaining:,} videos")

    if state.index_total > 0:
        pct = len(state.completed) / state.index_total * 100
        print(f"Progress:           {pct:.1f}%")

    print(f"\nTotal downloaded:   {state.total_gb:.2f} GB")

    hours = state.total_hours
    print(f"Total footage:      {hours:.1f} hours ({hours * 60:.0f} minutes)")
    print(f"Total frames:       {state.total_frames:,}")

    if state.started_at:
        print(f"\nStarted at:         {state.started_at}")
    if state.last_updated:
        print(f"Last updated:       {state.last_updated}")

    if state.failed:
        print(f"\nFailed videos ({len(state.failed)} total):")
        for i, (basename, error) in enumerate(list(state.failed.items())[:5]):
            print(f"  - {basename}: {error[:50]}...")
        if len(state.failed) > 5:
            print(f"  ... and {len(state.failed) - 5} more")

    if output_dir.exists():
        video_files = list(output_dir.glob("*.mp4"))
        action_files = list(output_dir.glob("*.jsonl"))
        print(f"\nFiles on disk:")
        print(f"  Videos:  {len(video_files):,} files")
        print(f"  Actions: {len(action_files):,} files")


def verify_videos(video_dir: Path, remove_invalid: bool = False) -> tuple[list[Path], list[Path]]:
    """check videos with cv2, update state"""
    try:
        import cv2
    except ImportError:
        print("opencv-python required for verification: pip install opencv-python")
        sys.exit(1)

    state_path = video_dir / STATE_FILE
    state = DownloadState.load(state_path)

    valid = []
    invalid = []

    videos = list(video_dir.glob("*.mp4"))
    print(f"Verifying {len(videos)} videos...")

    for i, video_path in enumerate(videos):
        basename = video_path.stem
        cap = cv2.VideoCapture(str(video_path))

        is_valid = False
        frame_count = 0
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                is_valid = True
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                valid.append(video_path)
            else:
                invalid.append(video_path)
        else:
            invalid.append(video_path)
        cap.release()

        if is_valid and basename not in state.completed:
            state.completed[basename] = {
                "size": video_path.stat().st_size,
                "timestamp": datetime.now().isoformat(),
                "duration_frames": frame_count,
            }
            state.total_bytes_downloaded += video_path.stat().st_size
            state.total_frames += frame_count
        elif not is_valid and basename in state.completed:
            del state.completed[basename]

        if (i + 1) % 100 == 0:
            print(f"  Verified {i + 1}/{len(videos)}... ({len(valid)} valid, {len(invalid)} invalid)")

    state.save(state_path)

    print(f"\nVerification complete:")
    print(f"  Valid:   {len(valid)}")
    print(f"  Invalid: {len(invalid)}")

    if invalid:
        print(f"\nInvalid videos:")
        for p in invalid[:10]:
            print(f"  - {p.name}")
        if len(invalid) > 10:
            print(f"  ... and {len(invalid) - 10} more")

        if remove_invalid:
            print(f"\nremoving {len(invalid)} invalid videos...")
            for p in invalid:
                p.unlink()
                action_path = p.with_suffix(".jsonl")
                if action_path.exists():
                    action_path.unlink()
            print("done - run download again to re-fetch")

    return valid, invalid


def scan_existing(output_dir: Path) -> None:
    """rebuild state from existing files"""
    print(f"scanning {output_dir} for existing downloads...")

    state_path = output_dir / STATE_FILE
    state = DownloadState.load(state_path)

    videos = list(output_dir.glob("*.mp4"))
    print(f"found {len(videos)} video files")

    added = 0
    for video_path in videos:
        basename = video_path.stem
        if basename not in state.completed:
            action_path = video_path.with_suffix(".jsonl")
            frames = get_frame_count_from_actions(action_path)
            size = video_path.stat().st_size

            state.completed[basename] = {
                "size": size,
                "timestamp": datetime.now().isoformat(),
                "duration_frames": frames,
            }
            state.total_bytes_downloaded += size
            state.total_frames += frames
            added += 1

    state.save(state_path)
    print(f"added {added} videos to state")
    print_status(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download OpenAI VPT Minecraft dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 100 videos for testing
  python download_vpt.py --output data/vpt --max-videos 100

  # Download full dataset with 8 workers
  python download_vpt.py --output data/vpt --workers 8

  # Check download status
  python download_vpt.py --output data/vpt --status

  # Verify downloaded videos
  python download_vpt.py --output data/vpt --verify

  # Verify and remove invalid videos
  python download_vpt.py --output data/vpt --verify --remove-invalid

  # Retry failed downloads
  python download_vpt.py --output data/vpt --retry-failed

  # Scan existing files to rebuild state
  python download_vpt.py --output data/vpt --scan
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/vpt",
        help="Output directory for downloaded videos (default: data/vpt)",
    )
    parser.add_argument(
        "--max-videos",
        "-n",
        type=int,
        default=None,
        help="Maximum number of videos to download (default: all)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )
    parser.add_argument(
        "--no-actions", action="store_true", help="Skip downloading action label files"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing downloads are valid video files",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Alias for --verify (deprecated)",
    )
    parser.add_argument(
        "--remove-invalid",
        action="store_true",
        help="Remove invalid videos during verification (use with --verify)",
    )
    parser.add_argument(
        "--status",
        "-s",
        action="store_true",
        help="Show download status and exit",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry downloading previously failed videos",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan existing files on disk and rebuild state tracking",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Reset download state (will re-download everything)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.reset_state:
        state_path = output_dir / STATE_FILE
        if state_path.exists():
            state_path.unlink()
            print(f"reset state: {state_path}")
        return

    if args.status:
        if not output_dir.exists():
            print(f"directory not found: {output_dir}")
            return
        print_status(output_dir)
        return

    if args.scan:
        if not output_dir.exists():
            print(f"directory not found: {output_dir}")
            return
        scan_existing(output_dir)
        return

    if args.verify or args.verify_only:
        if not output_dir.exists():
            print(f"directory not found: {output_dir}")
            return
        verify_videos(output_dir, remove_invalid=args.remove_invalid)
        return

    download_dataset(
        output_dir=output_dir,
        max_videos=args.max_videos,
        num_workers=args.workers,
        download_actions=not args.no_actions,
        retry_failed=args.retry_failed,
    )

    print(f"\nnext steps:")
    print(f"  status:      python download_vpt.py -o {output_dir} --status")
    print(f"  verify:      python download_vpt.py -o {output_dir} --verify")
    print(f"  retry:       python download_vpt.py -o {output_dir} --retry-failed")
    print(f"  train:       python train.py --data {output_dir} --frame-size 64 --seq-len 8")


if __name__ == "__main__":
    main()
