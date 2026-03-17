import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


TIMESTAMP_PATTERNS = [
    (re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"), "%Y-%m-%d_%H-%M-%S"),
    (re.compile(r"(\d{8}_\d{6})"), "%Y%m%d_%H%M%S"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot temporal detection counts from a directory of label files."
    )
    parser.add_argument(
        "--labels-dir",
        required=True,
        help="Directory containing label files (e.g., detections_text).",
    )
    parser.add_argument(
        "--pattern",
        default="*_det.json",
        help="Glob pattern for label files (default: *_det.json).",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=50,
        help="Rolling average window size in number of files (default: 50).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Optional confidence threshold for JSON detections (default: 0.0).",
    )
    parser.add_argument(
        "--timestamp-source",
        choices=["filename", "json", "auto"],
        default="filename",
        help="Timestamp source for JSON labels: filename, json, or auto (default: filename).",
    )
    parser.add_argument(
        "--divider-hours",
        type=float,
        default=None,
        help="Optional vertical divider position in hours since first image.",
    )
    parser.add_argument(
        "--divider-time",
        default="2025-12-17_08-09-00",
        help="Optional absolute divider time (e.g. 2025-12-17_08-00-00).",
    )
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Output PNG path. Defaults to <labels-dir>/temporal_label_counts.png",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path. Defaults to <labels-dir>/temporal_label_counts.csv",
    )
    parser.add_argument(
        "--title",
        default="Dec, Lar01, Aken: Lights off to Lights on coral counts",
        help="Plot title.",
    )
    return parser.parse_args()


def parse_timestamp_from_name(name_stem):
    for pattern, dt_format in TIMESTAMP_PATTERNS:
        match = pattern.search(name_stem)
        if match:
            try:
                return datetime.strptime(match.group(1), dt_format)
            except ValueError:
                continue
    return None


def parse_json_label(file_path, confidence_threshold=0.7, timestamp_source="filename"):
    with open(file_path, "r") as file_handle:
        payload = json.load(file_handle)

    timestamp_from_filename = parse_timestamp_from_name(file_path.stem)

    timestamp_from_json = None
    raw_timestamp = payload.get("timestamp")
    if raw_timestamp:
        try:
            timestamp_from_json = datetime.fromisoformat(raw_timestamp)
        except ValueError:
            timestamp_from_json = None

    if timestamp_source == "filename":
        timestamp = timestamp_from_filename
    elif timestamp_source == "json":
        timestamp = timestamp_from_json
    else:
        timestamp = timestamp_from_filename or timestamp_from_json

    detections = payload.get("detections", [])
    if confidence_threshold > 0.0:
        count = sum(
            1
            for detection in detections
            if float(detection.get("confidence", 0.0)) >= confidence_threshold
        )
    else:
        count = int(payload.get("detection_count", len(detections)))

    return timestamp, count


def parse_txt_label(file_path):
    timestamp = parse_timestamp_from_name(file_path.stem)
    with open(file_path, "r") as file_handle:
        count = sum(1 for line in file_handle if line.strip())
    return timestamp, count


def read_labels(labels_dir, pattern, confidence_threshold, timestamp_source):
    labels_path = Path(labels_dir)
    files = sorted(labels_path.rglob(pattern))

    rows = []
    skipped = 0

    for file_path in files:
        try:
            if file_path.suffix.lower() == ".json":
                timestamp, count = parse_json_label(file_path, confidence_threshold, timestamp_source)
            elif file_path.suffix.lower() == ".txt":
                timestamp, count = parse_txt_label(file_path)
            else:
                skipped += 1
                continue

            if timestamp is None:
                skipped += 1
                continue

            rows.append({"timestamp": timestamp, "count": count, "file": str(file_path)})
        except Exception:
            skipped += 1

    if not rows:
        return pd.DataFrame(columns=["timestamp", "count", "file"]), len(files), skipped

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df, len(files), skipped


def build_temporal_series(df, rolling_window):
    if df.empty:
        return df

    first_time = df["timestamp"].iloc[0]
    df = df.copy()
    df["hours_since_start"] = (df["timestamp"] - first_time).dt.total_seconds() / 3600.0
    df["rolling_avg"] = df["count"].rolling(window=rolling_window, min_periods=1).mean()
    return df


def _parse_divider_time(divider_time_str):
    if not divider_time_str:
        return None

    for dt_format in ("%Y-%m-%d_%H-%M-%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(divider_time_str, dt_format)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(divider_time_str)
    except ValueError:
        return None


def _set_multicolour_title(fig, ax, title):
    """
    Render the axes title with 'Lights on' in green and 'Lights off' in purple.
    Falls back to a plain title if neither phrase is found.
    """
    import re

    keywords = [
        (re.compile(r"(lights on)", re.IGNORECASE), "green"),
        (re.compile(r"(lights off)", re.IGNORECASE), "purple"),
    ]

    # Tokenise the title into coloured segments
    events = []
    for pattern, colour in keywords:
        for m in pattern.finditer(title):
            events.append((m.start(), m.end(), colour))
    events.sort(key=lambda e: e[0])

    segments = []
    last = 0
    for start, end, colour in events:
        if start > last:
            segments.append((title[last:start], "black"))
        segments.append((title[start:end], colour))
        last = end
    if last < len(title):
        segments.append((title[last:], "black"))

    if not segments or all(c == "black" for _, c in segments):
        ax.set_title(title)
        return

    fontsize = plt.rcParams.get("axes.titlesize", 14)
    fontweight = plt.rcParams.get("axes.titleweight", "normal")
    title_y = 1.02  # axes-fraction coords just above the axes box
    ax.set_title("")  # suppress matplotlib's own title

    # Place all segments temporarily at x=0 so we can measure them
    texts = [
        ax.text(
            0.0, title_y, seg_text,
            ha="left", va="bottom",
            fontsize=fontsize, fontweight=fontweight,
            color=colour,
            transform=ax.transAxes,
        )
        for seg_text, colour in segments
    ]

    # Measure each segment's pixel width, then reposition centred on the axes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [t.get_window_extent(renderer) for t in texts]
    total_width_px = sum(bb.width for bb in bboxes)
    ax_win = ax.get_window_extent(renderer)
    x_px = ax_win.x0 + (ax_win.width - total_width_px) / 2
    for t, bb in zip(texts, bboxes):
        t.set_x((x_px - ax_win.x0) / ax_win.width)
        x_px += bb.width


def plot_temporal_counts(df, title, output_plot, divider_hours=None):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(df["timestamp"], df["count"], label="Detections per image", alpha=0.5)
    ax.plot(df["timestamp"], df["rolling_avg"], label="Rolling average", linewidth=2)

    before_avg = None
    after_avg = None

    if divider_hours is not None:
        divider_timestamp = df["timestamp"].iloc[0] + timedelta(hours=divider_hours)
        ax.axvline(
            x=divider_timestamp,
            color="purple",
            linestyle="--",
            linewidth=1.8,
            label=f"Lights off ({divider_hours:.2f}h)",
        )

        before_df = df[df["hours_since_start"] < divider_hours]
        after_df = df[df["hours_since_start"] >= divider_hours]

        if not before_df.empty:
            before_avg = before_df["count"].mean()
            ax.hlines(
                y=before_avg,
                xmin=before_df["timestamp"].min(),
                xmax=before_df["timestamp"].max(),
                colors="purple",
                linestyles="-.",
                linewidth=2,
                label=f"Avg before: {before_avg:.2f}",
            )

        if not after_df.empty:
            after_avg = after_df["count"].mean()
            ax.hlines(
                y=after_avg,
                xmin=after_df["timestamp"].min(),
                xmax=after_df["timestamp"].max(),
                colors="green",
                linestyles="-.",
                linewidth=2,
                label=f"Avg after: {after_avg:.2f}",
            )

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()

    first_time = df["timestamp"].iloc[0]
    last_time = df["timestamp"].iloc[-1]
    ax.set_xlabel(
        f"Time of image capture (first: {first_time.strftime('%Y-%m-%d %H:%M:%S')}, "
        f"last: {last_time.strftime('%Y-%m-%d %H:%M:%S')})"
    )
    ax.set_ylabel("Detection count")
    _set_multicolour_title(fig, ax, title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_plot, dpi=300)
    plt.close(fig)

    return before_avg, after_avg


def main():
    args = parse_args()

    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    output_plot = Path(args.output_plot) if args.output_plot else labels_dir / "temporal_label_counts.png"
    output_csv = Path(args.output_csv) if args.output_csv else labels_dir / "temporal_label_counts.csv"

    df, total_files, skipped_files = read_labels(
        labels_dir=labels_dir,
        pattern=args.pattern,
        confidence_threshold=0.7,
        timestamp_source=args.timestamp_source,
    )

    if df.empty:
        print(f"No parseable label data found. Files scanned: {total_files}, skipped: {skipped_files}")
        return

    df = build_temporal_series(df, args.rolling_window)

    divider_hours = args.divider_hours
    if divider_hours is None and args.divider_time:
        divider_dt = _parse_divider_time(args.divider_time)
        if divider_dt is None:
            raise ValueError(
                f"Could not parse --divider-time '{args.divider_time}'. "
                "Try formats like 2025-12-17_08-00-00 or 2025-12-17 08:00:00."
            )
        first_time = df["timestamp"].iloc[0]
        divider_hours = (divider_dt - first_time).total_seconds() / 3600.0

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    before_avg, after_avg = plot_temporal_counts(df, args.title, output_plot, divider_hours=divider_hours)
    df.to_csv(output_csv, index=False)

    print(f"Scanned files: {total_files}")
    print(f"Parsed points: {len(df)}")
    print(f"Skipped files: {skipped_files}")
    print(f"Timestamp source: {args.timestamp_source}")
    print(f"Rolling window: {args.rolling_window}")
    if divider_hours is not None:
        print(f"Divider line (hours since first image): {divider_hours:.4f}")
        if before_avg is not None:
            print(f"Average before divider: {before_avg:.4f}")
        if after_avg is not None:
            print(f"Average after divider: {after_avg:.4f}")
    print(f"Saved plot: {output_plot}")
    print(f"Saved csv: {output_csv}")


if __name__ == "__main__":
    main()
