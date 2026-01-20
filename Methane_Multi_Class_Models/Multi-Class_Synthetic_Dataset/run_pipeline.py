import argparse
import sys

import config

# Import the three pipeline steps
from pipeline import (
    step_1_create_base_files,
    step_2_create_numpy_dataset,
    step_3_create_exportable_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run the Multi-Class Synthetic Dataset Generation Pipeline"
    )

    parser.add_argument(
        "--include-artificial",
        action="store_true",
        help="Include artificial background samples in dataset",
    )

    parser.add_argument(
        "--channels",
        type=str,
        choices=["single", "double"],
        default="double",
        help="Output format: 'single' (frames only) or 'double' (background + plume)",
    )

    parser.add_argument(
        "--step",
        type=str,
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which pipeline step to run (1=base, 2=numpy, 3=final, all=complete pipeline)",
    )

    parser.add_argument(
        "--test-videos",
        nargs="+",
        help="Test mode: only process specific videos (e.g., --test-videos 1237 1238)",
    )

    parser.add_argument(
        "--frames-per-class",
        type=int,
        default=config.FRAMES_PER_CLASS_DEFAULT,
        help=f"Number of frames per class (default: {config.FRAMES_PER_CLASS_DEFAULT})",
    )

    parser.add_argument(
        "--skip-step-1", action="store_true", help="Skip step 1 (base dataset creation)"
    )

    parser.add_argument(
        "--skip-step-2",
        action="store_true",
        help="Skip step 2 (numpy dataset creation)",
    )

    args = parser.parse_args()

    # Create configuration object
    pipeline_config = config.create_config_from_args(args)
    pipeline_config.validate()

    print("=" * 80)
    print("MULTI-CLASS SYNTHETIC DATASET GENERATION PIPELINE")
    print("=" * 80)
    print("Configuration:")
    print(f"  Step: {args.step}")
    print(f"  Channels: {pipeline_config.channels}")
    print(f"  Include artificial: {pipeline_config.include_artificial}")
    if args.test_videos:
        print(f"  Test videos: {args.test_videos}")
    print(f"  Frames per class: {pipeline_config.frames_per_class}")
    print("=" * 80)

    # Step 1: Base Dataset Creation
    if args.step in ["1", "all"] and not args.skip_step_1:
        print("\n" + ">" * 80)
        print(">>> RUNNING STEP 1: BASE DATASET CREATION")
        print(">" * 80)

        step_1_success = step_1_create_base_files.run(
            test_videos=args.test_videos, pipeline_config=pipeline_config
        )

        if not step_1_success:
            print("\nERROR: Step 1 failed!")
            return False

        print("\n" + "#" * 80)
        print("### STEP 1 COMPLETED SUCCESSFULLY")
        print("#" * 80)

    # Step 2: Numpy Dataset Creation
    if args.step in ["2", "all"] and not args.skip_step_2:
        print("\n" + ">" * 80)
        print(">>> RUNNING STEP 2: NUMPY DATASET CREATION")
        print(">" * 80)

        step_2_success = step_2_create_numpy_dataset.run(
            test_videos=args.test_videos, pipeline_config=pipeline_config
        )

        if not step_2_success:
            print("\nERROR: Step 2 failed!")
            return False

        print("\n" + "#" * 80)
        print("### STEP 2 COMPLETED SUCCESSFULLY")
        print("#" * 80)

    # Step 3: Final Dataset Assembly
    if args.step in ["3", "all"]:
        print("\n" + ">" * 80)
        print(">>> RUNNING STEP 3: FINAL DATASET ASSEMBLY")
        print(">" * 80)

        step_3_success = step_3_create_exportable_dataset.run(
            pipeline_config=pipeline_config
        )

        if not step_3_success:
            print("\nERROR: Step 3 failed!")
            return False

        print("\n" + "#" * 80)
        print("### STEP 3 COMPLETED SUCCESSFULLY")
        print("#" * 80)

    # Final summary
    print("\n" + "=" * 80)
    print("=" * 80)
    print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("=" * 80)

    print("\nYour dataset is ready at:")
    print(f"  - Processed data: {pipeline_config.processed_dataset_path}/")
    print(f"  - Final dataset: {pipeline_config.final_dataset_path}/")
    print("\nYou can now use this dataset with PyTorch DataLoaders!")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
