# Gas Leak Detection - Background Subtraction

This script implements the background subtraction method from the paper:
**"LangGas: Introducing Language in Selective Zero-Shot Background Subtraction for Semi-Transparent Gas Leak Detection with a New Dataset"** (arXiv:2503.02910v1)

## Overview

The script performs background subtraction on infrared video to detect semi-transparent gas leaks. It implements Section 4.1 of the paper, which includes:

1. **Background Subtraction**: Using MOG2 algorithm with short history (30 frames)
2. **Adaptive Enhancement**: Enhancing the difference image using Equation 1 from the paper
3. **Thresholding**: Converting to binary mask
4. **Morphological Operations**: Refining the mask using opening and closing operations

## Requirements

```bash
pip install opencv-python numpy
```

## Usage

### Basic Usage

**NOTE:** Run from the ``source-localization`` folder.

```bash
python3 background_subtraction.py ./data/gasvid_test_0.mp4 -o ./data/side_by_side.mp4 -s ./data/output_0.mp4
```

This will process `test.mp4` and show a live preview with three panels:
- **Left**: Original frame
- **Middle**: Enhanced difference image
- **Right**: Detection result (red overlay on gas leak)

### Save Output to File

```bash
# Save side-by-side view
python background_subtraction.py test.mp4 -o output_sidebyside.mp4

# Save background-subtracted view only
python background_subtraction.py test.mp4 -s output_subtracted.mp4

# Save both versions
python background_subtraction.py test.mp4 -o sidebyside.mp4 -s subtracted.mp4
```

### Custom Parameters

```bash
python background_subtraction.py test.mp4 \
    --history 50 \
    --threshold 30 \
    --morph-kernel 40 \
    --enhancement 20
```

### No Preview Mode (Faster Processing)

```bash
python background_subtraction.py test.mp4 -o output.mp4 --no-preview
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-o, --output` | None | Path to save side-by-side output video |
| `-s, --subtracted` | None | Path to save background-subtracted video only |
| `--history` | 30 | Number of frames for background model (paper uses 30) |
| `--threshold` | 40 | Binary threshold value for mask creation |
| `--morph-kernel` | 30 | Size of morphological closing kernel |
| `--enhancement` | 15 | Default enhancement factor (adaptive) |
| `--no-preview` | False | Disable live preview window |

## Implementation Details

### Background Subtraction

The script uses OpenCV's MOG2 (Mixture of Gaussians) background subtractor, which was found to perform best in the paper's experiments (Table 2):
- MOG2 achieved **0.50 IoU** overall with morphological refinement
- Short history of 30 frames avoids false positives from slow-moving objects like clouds

### Adaptive Enhancement

Following Equation 1 from the paper:

```
α = min(255 / (μ + σ), 15)
I'' = clip(α * I', 0, 255)
```

Where:
- `μ` is the mean intensity of the difference image
- `σ` is the standard deviation
- `I'` is the absolute difference between background and current frame
- `I''` is the enhanced image

This prevents clipping when the difference intensity is large.

### Morphological Operations

1. **Opening** (5×5 kernel): Removes salt noise
2. **Closing** (30×30 kernel by default): Connects separated leak regions

The paper found that larger morphological kernel sizes tend to yield better results (Figure 3).

## Controls (Preview Mode)

- **'q'**: Quit processing
- **'p'**: Pause/resume
- **Any key** (when paused): Continue

## Performance Notes

From the paper (Table 2), the MOG2 baseline achieved:
- **Stationary Foreground**: 0.56 IoU, 0.67 Precision, 0.80 Recall
- **Moving Foreground**: 0.38 IoU, 0.56 Precision, 0.57 Recall
- **Overall**: 0.50 IoU, 0.63 Precision, 0.70 Recall

The full pipeline (with VLM filtering, temporal filtering, and SAM 2) achieves **0.69 IoU** overall.

## Example Output

The script can generate two types of output videos:

### 1. Side-by-Side View (`-o` option)
The output video shows three views side-by-side:
1. Original infrared frame
2. Enhanced difference (shows moving regions)
3. Detection result (red overlay on detected gas leaks)

### 2. Background-Subtracted View (`-s` option)
Shows only the enhanced difference image in grayscale, displaying the moving parts of the scene with gas leaks appearing as bright white regions.

You can generate both outputs simultaneously:
```bash
python background_subtraction.py test.mp4 -o sidebyside.mp4 -s subtracted.mp4
```

## Limitations

This script implements only the **first stage** of the full LangGas pipeline:
- ✅ Background subtraction
- ✅ Adaptive enhancement
- ✅ Morphological refinement
- ❌ VLM filtering (requires OWL-v2 model)
- ❌ Temporal filtering
- ❌ SAM 2 segmentation

For the complete pipeline with all components, additional dependencies and models would be needed.

## References

**Paper**: Wenqi Marshall Guo, Yiyang Du, Shan Du. "LangGas: Introducing Language in Selective Zero-Shot Background Subtraction for Semi-Transparent Gas Leak Detection with a New Dataset." arXiv:2503.02910v1 [cs.CV], March 2025.

## Troubleshooting

### Video file not found
```bash
# Make sure the video file exists
ls -lh test.mp4
```

### No preview window appears
- If running on a headless server, use `--no-preview` flag
- Make sure X11 forwarding is enabled if using SSH

### Performance issues
- Use `--no-preview` to speed up processing
- Reduce video resolution before processing
- Process every N frames instead of every frame
