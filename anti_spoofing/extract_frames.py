import cv2
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anti_spoofing.config import (
    DATA_ROOT,
    FRAMES_ROOT,
    REAL_DIRS,
    FAKE_DIRS,
    FRAMES_PER_VIDEO,
    TRAIN_FRAC,
    VAL_FRAC,
    SEED,
)


def collect_videos(data_root: Path, dir_names: list[str]) -> list[Path]:
    # find all video files inside the given folder names
    videos = []
    for name in dir_names:
        folder = data_root / name
        if not folder.exists():
            print(f"[WARN] folder not found: {folder}")
            continue
        found = (
            list(folder.glob("*.mp4"))
            + list(folder.glob("*.MOV"))
            + list(folder.glob("*.avi"))
        )
        videos.extend(found)
    return videos


def split_videos(videos: list[Path], train_frac: float, val_frac: float, seed: int):
    # split at video level so no video leaks between train and test
    rng = random.Random(seed)
    shuffled = videos[:]
    rng.shuffle(shuffled)

    n       = len(shuffled)
    n_train = max(1, int(n * train_frac))
    n_val   = max(1, int(n * val_frac))

    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


def extract_frames(video_path: Path, out_dir: Path, n_frames: int, prefix: str) -> int:
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        cap.release()
        print(f"[WARN] 0 frames in {video_path.name}, skipping")
        return 0

    # pick evenly-spaced frame indices across the video
    step    = max(1, total // n_frames)
    indices = set(range(0, total, step)[:n_frames])

    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    idx   = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            fname = out_dir / f"{prefix}_{video_path.stem}_f{idx:06d}.jpg"
            cv2.imwrite(str(fname), frame)
            saved += 1
        idx += 1

    cap.release()
    return saved


def main():
    random.seed(SEED)

    real_videos = collect_videos(DATA_ROOT, REAL_DIRS)
    fake_videos = collect_videos(DATA_ROOT, FAKE_DIRS)
    print(f"Found {len(real_videos)} real videos, {len(fake_videos)} fake videos")

    r_train, r_val, r_test = split_videos(real_videos, TRAIN_FRAC, VAL_FRAC, SEED)
    f_train, f_val, f_test = split_videos(fake_videos, TRAIN_FRAC, VAL_FRAC, SEED)

    splits = {
        "train": [(r_train, "real"), (f_train, "fake")],
        "val":   [(r_val,   "real"), (f_val,   "fake")],
        "test":  [(r_test,  "real"), (f_test,  "fake")],
    }

    total_saved = 0
    for split_name, groups in splits.items():
        for videos, label_name in groups:
            out_dir = FRAMES_ROOT / split_name / label_name
            for vid in videos:
                category = vid.parent.name.replace(" ", "_")
                n = extract_frames(vid, out_dir, FRAMES_PER_VIDEO, prefix=category)
                total_saved += n
                print(f"  [{split_name}/{label_name}] {vid.name} → {n} frames")

    print(f"\nDone. Total frames saved: {total_saved}")
    print(f"Output directory: {FRAMES_ROOT.resolve()}")


if __name__ == "__main__":
    main()
