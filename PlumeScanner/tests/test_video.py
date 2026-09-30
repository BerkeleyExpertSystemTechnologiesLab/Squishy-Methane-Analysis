from plume_focus import video


def test_legacy_video_facade_reexports_supported_surface() -> None:
    supported_names = (
        "SyntheticMethaneLeakConfig",
        "SyntheticPlumeVideoConfig",
        "SyntheticVideoConfig",
        "RasterVideoScan",
        "VideoProofReport",
        "budget_column_sweep_mask",
        "column_sweep_scan_moving_frames",
        "estimate_opencv_motion",
        "foreground_mask",
        "generate_synthetic_methane_frames",
        "generate_synthetic_methane_values",
        "generate_synthetic_object_values",
        "generate_synthetic_plume_frames",
        "generate_synthetic_video_frames",
        "js_divergence",
        "kernel_plume_attention_field",
        "kl_divergence",
        "methane_scan_roi",
        "normalize_distribution",
        "raster_scan_moving_frames",
        "run_video_proof",
        "main",
    )

    assert all(hasattr(video, name) for name in supported_names)
    assert video.run_video_proof.__module__ == "plume_focus.video_workflow"
    assert video.main.__module__ == "plume_focus.video_cli"
