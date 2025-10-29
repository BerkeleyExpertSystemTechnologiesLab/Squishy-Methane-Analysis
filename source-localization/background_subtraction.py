#!/usr/bin/env python3
"""
Background Subtraction for Gas Leak Detection
Based on the LangGas paper (arXiv:2503.02910v1)

This script implements the background subtraction method described in Section 4.1
of the paper for detecting gas leaks in infrared video.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def enhance_difference(diff_image, alpha=15):
    """
    Enhance the difference image with adaptive enhancement factor.

    Based on Equation 1 from the paper:
    α = min(255 / (μ + σ), 15)

    Args:
        diff_image: Absolute difference between background and current frame
        alpha: Default enhancement factor

    Returns:
        Enhanced and clipped difference image
    """
    mean = np.mean(diff_image)
    std = np.std(diff_image)

    # Adaptive enhancement factor to prevent clipping
    if mean + std > 0:
        adaptive_alpha = min(255 / (mean + std), alpha)
    else:
        adaptive_alpha = alpha

    # Enhance and clip
    enhanced = np.clip(adaptive_alpha * diff_image, 0, 255).astype(np.uint8)

    return enhanced, adaptive_alpha


def apply_morphological_operations(mask, kernel_size=30):
    """
    Apply morphological operations to refine the mask.

    Args:
        mask: Binary mask from thresholding
        kernel_size: Size of the morphological kernel

    Returns:
        Refined mask
    """
    # Opening to remove salt noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Closing to connect separated regions
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


def process_video(video_path, output_path=None, output_subtracted=None, history=30,
                  threshold=40, morph_kernel_size=30, enhancement_factor=15, show_preview=True):
    """
    Process video using background subtraction for gas leak detection.

    Args:
        video_path: Path to input video file
        output_path: Path to save side-by-side output video (optional)
        output_subtracted: Path to save background-subtracted video only (optional)
        history: Number of frames for background model history
        threshold: Threshold for binary mask creation
        morph_kernel_size: Size of morphological closing kernel
        enhancement_factor: Default enhancement factor
        show_preview: Whether to show live preview
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"\nProcessing parameters:")
    print(f"  History: {history}")
    print(f"  Threshold: {threshold}")
    print(f"  Morphological kernel size: {morph_kernel_size}")
    print(f"  Enhancement factor: {enhancement_factor}")

    # Initialize MOG2 background subtractor
    # Using the parameters from the paper
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=16,  # Default value
        detectShadows=False
    )

    # Setup video writer if output path specified
    writer = None
    writer_subtracted = None

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps,
                                 (width * 3, height))
        print(f"\nSaving side-by-side output to: {output_path}")

    if output_subtracted:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer_subtracted = cv2.VideoWriter(str(output_subtracted), fourcc, fps,
                                            (width, height))
        print(f"Saving background-subtracted output to: {output_subtracted}")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Update background model
            back_sub.apply(frame)

            # Get background image from the model
            background = back_sub.getBackgroundImage()

            if background is None:
                # Background not ready yet
                continue

            # Compute absolute difference
            diff = cv2.absdiff(background, frame)

            # Convert to grayscale if needed
            if len(diff.shape) == 3:
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            else:
                diff_gray = diff

            # Enhance the difference
            enhanced, actual_alpha = enhance_difference(diff_gray, enhancement_factor)

            # Apply threshold
            _, mask = cv2.threshold(enhanced, threshold, 255, cv2.THRESH_BINARY)

            # Apply morphological operations
            mask_refined = apply_morphological_operations(mask, morph_kernel_size)

            # Create visualization
            # Convert mask to BGR for visualization
            mask_bgr = cv2.cvtColor(mask_refined, cv2.COLOR_GRAY2BGR)

            # Create overlay on original frame
            overlay = frame.copy()
            overlay[mask_refined > 0] = [0, 0, 255]  # Red color for detected regions
            result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            # Combine images for display
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([frame, enhanced_bgr, result])

            # Add text overlay with frame info
            text = f"Frame: {frame_count}/{total_frames} | Alpha: {actual_alpha:.2f}"
            cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            # Write frame if output specified
            if writer:
                writer.write(combined)

            # Write background-subtracted frame if specified
            if writer_subtracted:
                writer_subtracted.write(enhanced_bgr)

            # Show preview
            if show_preview:
                cv2.imshow('Gas Leak Detection | Original | Enhanced | Result', combined)

                # Press 'q' to quit, 'p' to pause
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nStopped by user")
                    break
                elif key == ord('p'):
                    print("Paused. Press any key to continue...")
                    cv2.waitKey(0)

            # Print progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")

    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if writer_subtracted:
            writer_subtracted.release()
        if show_preview:
            cv2.destroyAllWindows()

        print(f"\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Background subtraction for gas leak detection (LangGas method)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python background_subtraction.py test.mp4
  python background_subtraction.py test.mp4 -o output_sidebyside.mp4
  python background_subtraction.py test.mp4 -s output_subtracted.mp4
  python background_subtraction.py test.mp4 -o sidebyside.mp4 -s subtracted.mp4
  python background_subtraction.py test.mp4 --history 50 --threshold 30
  python background_subtraction.py test.mp4 --no-preview -o output.mp4
        """
    )

    parser.add_argument('video', type=str, help='Input video file path')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output video file path for side-by-side view (optional)')
    parser.add_argument('-s', '--subtracted', type=str, default=None,
                        help='Output video file path for background-subtracted view only (optional)')
    parser.add_argument('--history', type=int, default=30,
                        help='Background model history (default: 30)')
    parser.add_argument('--threshold', type=int, default=40,
                        help='Binary threshold value (default: 40)')
    parser.add_argument('--morph-kernel', type=int, default=30,
                        help='Morphological kernel size (default: 30)')
    parser.add_argument('--enhancement', type=float, default=15,
                        help='Enhancement factor (default: 15)')
    parser.add_argument('--no-preview', action='store_true',
                        help='Disable live preview window')

    args = parser.parse_args()

    # Check if input file exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1

    # Process video
    try:
        process_video(
            video_path=video_path,
            output_path=args.output,
            output_subtracted=args.subtracted,
            history=args.history,
            threshold=args.threshold,
            morph_kernel_size=args.morph_kernel,
            enhancement_factor=args.enhancement,
            show_preview=not args.no_preview
        )
        return 0
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())