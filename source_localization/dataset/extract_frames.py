#!/usr/bin/env python3
"""
Extract frames from plume videos and save them as individual images.

This script processes plume videos in the plume_video_dataset directory
and extracts frames only within the leak ranges specified in leak_range.csv,
saving them to plume_image_dataset/all_images.
"""

import cv2
import argparse
import csv
from pathlib import Path
from typing import Optional, Dict, Tuple


def parse_leak_ranges(csv_path: Path) -> Dict[int, Tuple[int, int]]:
    """
    Parse leak ranges from CSV file.

    Args:
        csv_path: Path to leak_range.csv file

    Returns:
        Dictionary mapping video numbers to (leak_start_seconds, leak_end_seconds) tuples
    """
    ranges = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_no = int(row['Video No.'])
            leak_start = int(row['Leak Range Start (s)'])
            leak_end = int(row['Leak Range End (s)'])
            ranges[video_no] = (leak_start, leak_end)
    return ranges


def seconds_to_frame_number(seconds: float, fps: float) -> int:
    """
    Convert time in seconds to frame number.

    Args:
        seconds: Time in seconds
        fps: Frames per second

    Returns:
        Frame number (0-indexed)
    """
    return int(seconds * fps)


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    leak_start_frame: Optional[int] = None,
    leak_end_frame: Optional[int] = None,
    frame_prefix: Optional[str] = None
) -> int:
    """
    Extract frames from a video and save them as images.
    Only extracts frames within the leak range if specified.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        leak_start_frame: First frame to extract (None = start from beginning)
        leak_end_frame: Last frame to extract (None = extract to end)
        frame_prefix: Optional prefix for frame filenames (default: video stem)

    Returns:
        Number of frames extracted
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine frame range to extract
    if leak_start_frame is not None and leak_end_frame is not None:
        start_frame = max(0, leak_start_frame)
        end_frame = min(total_frames - 1, leak_end_frame)
        frames_in_range = end_frame - start_frame + 1
    else:
        start_frame = 0
        end_frame = total_frames - 1

    # Determine frame prefix
    if frame_prefix is None:
        frame_prefix = video_path.stem

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    frames_saved = 0

    # Seek to start frame if leak range is specified
    if leak_start_frame is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = start_frame - 1  # Will be incremented to start_frame on first read

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Stop if we've passed the leak end frame
            if leak_end_frame is not None and frame_count > leak_end_frame:
                break

            # Skip frames before leak start
            if leak_start_frame is not None and frame_count < leak_start_frame:
                continue

            # Save frame as image
            # Format: {prefix}_frame_{frame_number:06d}.png
            # Use relative frame number within leak range for naming
            if leak_start_frame is not None:
                relative_frame_num = frame_count - leak_start_frame + 1
            else:
                relative_frame_num = frame_count
            
            frame_filename = output_dir / f"{frame_prefix}_frame_{relative_frame_num:06d}.png"
            cv2.imwrite(str(frame_filename), frame)
            frames_saved += 1

    finally:
        cap.release()

    return frames_saved


def extract_frames_from_dataset(
    plume_video_dir: Path,
    output_dir: Path,
    csv_path: Path,
    max_videos: Optional[int] = None
) -> None:
    """
    Extract frames from plume videos in the dataset, only extracting frames
    within the leak ranges specified in leak_range.csv.

    Args:
        plume_video_dir: Directory containing plume video files
        output_dir: Directory to save extracted frames
        csv_path: Path to leak_range.csv file
        max_videos: Maximum number of videos to process (None = all)
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse leak ranges from CSV
    print(f"Reading leak ranges from: {csv_path}")
    leak_ranges = parse_leak_ranges(csv_path)
    print(f"Found {len(leak_ranges)} videos in CSV")

    # Find all video files
    video_files = sorted(plume_video_dir.glob("MOV_*.mp4"))
    
    if not video_files:
        print(f"Warning: No MOV_*.mp4 video files found in {plume_video_dir}")
        return

    print(f"Found {len(video_files)} plume video files in directory")
    if max_videos:
        print(f"Processing up to {max_videos} videos")
    print(f"Output directory: {output_dir}\n")

    total_frames = 0
    processed_count = 0
    skipped_count = 0

    for video_path in video_files:
        # Stop if we've reached the max videos limit
        if max_videos and processed_count >= max_videos:
            print(f"\nReached maximum video limit ({max_videos}), stopping processing")
            break

        # Extract video number from filename (e.g., MOV_1237_plume.mp4 -> 1237)
        try:
            # Handle both MOV_1237_plume.mp4 and MOV_1237.mp4 formats
            parts = video_path.stem.split('_')
            if len(parts) >= 2:
                video_no = int(parts[1])
            else:
                skipped_count += 1
                continue
        except (IndexError, ValueError):
            skipped_count += 1
            continue

        # Check if video is in the leak ranges CSV
        if video_no not in leak_ranges:
            skipped_count += 1
            continue

        leak_start_sec, leak_end_sec = leak_ranges[video_no]
        print(f"Processing video {processed_count + 1}: {video_path.name} (leak: {leak_start_sec}s-{leak_end_sec}s)")

        # Get video FPS to convert seconds to frames
        cap_temp = cv2.VideoCapture(str(video_path))
        if not cap_temp.isOpened():
            print(f"  Warning: Could not open, skipping")
            skipped_count += 1
            continue

        fps = cap_temp.get(cv2.CAP_PROP_FPS)
        cap_temp.release()

        # Convert seconds to frame numbers
        leak_start_frame = seconds_to_frame_number(leak_start_sec, fps)
        leak_end_frame = seconds_to_frame_number(leak_end_sec, fps)

        try:
            frames_extracted = extract_frames_from_video(
                video_path=video_path,
                output_dir=output_dir,
                leak_start_frame=leak_start_frame,
                leak_end_frame=leak_end_frame,
                frame_prefix=None  # Will use video stem
            )
            total_frames += frames_extracted
            processed_count += 1
            print(f"  ✓ Extracted {frames_extracted} frames")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            skipped_count += 1

    print(f"\n{'='*60}")
    print(f"Frame extraction complete!")
    print(f"  Processed: {processed_count} videos")
    print(f"  Skipped: {skipped_count} videos")
    print(f"  Total frames extracted: {total_frames}")
    print(f"  Frames saved to: {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Extract frames from plume videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract leak frames from all plume videos (default paths)
  python extract_frames.py

  # Extract frames from first 5 videos only
  python extract_frames.py --max-videos 5

  # Extract frames with custom paths
  python extract_frames.py --plume-dir /path/to/plume_videos --output-dir /path/to/output --csv-path /path/to/leak_range.csv
        """
    )

    parser.add_argument('--plume-dir', type=str, default=None,
                        help='Directory containing plume video files (default: source_localization/dataset/plume_video_dataset)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save extracted frames (default: source_localization/dataset/plume_image_dataset/all_images)')
    parser.add_argument('--csv-path', type=str, default=None,
                        help='Path to leak_range.csv (default: source_localization/dataset/original_gasvid_dataset/leak_range.csv)')
    parser.add_argument('--max-videos', type=int, default=None,
                        help='Maximum number of videos to process (default: all)')

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent

    if args.plume_dir:
        plume_video_dir = Path(args.plume_dir)
    else:
        plume_video_dir = script_dir / 'plume_video_dataset'

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = script_dir / 'plume_image_dataset' / 'all_images'

    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        csv_path = script_dir / 'original_gasvid_dataset' / 'leak_range.csv'

    # Validate paths
    if not plume_video_dir.exists():
        print(f"Error: Plume video directory not found: {plume_video_dir}")
        return 1

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    # Extract frames
    try:
        extract_frames_from_dataset(
            plume_video_dir=plume_video_dir,
            output_dir=output_dir,
            csv_path=csv_path,
            max_videos=args.max_videos
        )
        return 0
    except Exception as e:
        print(f"Error extracting frames: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

