from dataclasses import dataclass
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent


# Input paths
EXCEL_PATH = "source_data/GasVid_Dataset/GasVid Logging File.xlsx"
MOV_PATH = "source_data/GasVid_Dataset/Videos/"
PLUME_MODELING_PATH = "source_data/Metadata/Gasvid Plume Models.xlsx"
CONSOLIDATED_METADATA_PATH = "source_data/Metadata/consolidated_metadata.json"
CLASSES_JSON_PATH = "source_data/Metadata/classes.json"


# Background paths
BACKGROUNDS_PATH = "source_data/BackGrounds"
ARTIFICIAL_BACKGROUNDS_PATH = "source_data/BackGrounds/Artificial_Backgrounds"


# Output paths (base names)
PROCESSED_DATASET_BASE = "export_datasets/Processed_Dataset"
FINAL_DATASET_BASE = "export_datasets/Final_Dataset"


def get_processed_dataset_path(channels="double", include_artificial=False):
    """Get the processed dataset path for the given channel mode."""
    suffix = "_w_artif" if include_artificial else ""
    return f"{PROCESSED_DATASET_BASE}_{channels}_channel{suffix}"


def get_final_dataset_path(channels="double", include_artificial=False):
    """Get the final dataset path for the given channel mode."""
    suffix = "_w_artif" if include_artificial else ""
    return f"{FINAL_DATASET_BASE}_{channels}_channel{suffix}"


# Processing parameters
NUM_CLASSES = 8
CLASS_DURATION_MINUTES = 3
# Frames per class of 200 gives approx 44,000 numpy data points (200 x 8 classes x 28 Videos)
# Typically gives good results
FRAMES_PER_CLASS_DEFAULT = 200
VIDEO_MIN_DURATION_MINUTES = 24

# Background generation parameters
BACKGROUND_ALPHA = (
    0.1  # Used for generating a background, may not be necessary anymore?
)


@dataclass
class PipelineConfig:
    """Configuration for dataset generation pipeline."""

    # Channel configuration
    channels: str = "double"  # "single" or "double"

    # Background settings
    include_artificial: bool = False

    # Output settings
    frames_per_class: int = FRAMES_PER_CLASS_DEFAULT

    # Augmentation
    apply_augmentation: bool = False

    def validate(self):
        """Validate configuration settings."""
        if self.channels not in ["single", "double"]:
            raise ValueError(f"Invalid channels: {self.channels}")

    @property
    def processed_dataset_path(self):
        """Get the processed dataset path for this configuration."""
        return get_processed_dataset_path(self.channels, self.include_artificial)

    @property
    def final_dataset_path(self):
        """Get the final dataset path for this configuration."""
        return get_final_dataset_path(self.channels, self.include_artificial)


def create_config_from_args(args):
    """Create PipelineConfig from command-line arguments."""
    return PipelineConfig(
        channels=args.channels,
        include_artificial=args.include_artificial,
        frames_per_class=args.frames_per_class,
    )
