import asyncio
import pandas as pd
from os.path import join
from hooper.infer import process_video
from hooper.utils import comma_sep_str_to_list, get_config_path


async def main(args):
    r"""Performs video inference with Hooper v2."""
    output = await process_video(
        args.video_file,
        args.out_dir,
        paused_ts=comma_sep_str_to_list(args.paused_ts_str),
        frame_sample_rate=args.frame_sample_rate,
        context_window=args.context_window,
        min_detect_shot_score=args.min_detect_shot_score,
        min_shot_score=args.min_shot_score,
        min_hoop_fast_score=args.min_hoop_fast_score,
        min_hoop_slow_score=args.min_hoop_slow_score,
        min_ball_score=args.min_ball_score,
        min_person_box_score=args.min_person_box_score,
        min_person_keypoint_score=args.min_person_keypoint_score,
        min_segment_skeleton_iou=args.min_segment_skeleton_iou,
        hoop_fast_stride=args.hoop_fast_stride,
        hoop_slow_stride=args.hoop_slow_stride,
        dist_pixel_thres=args.dist_pixel_thres,
        max_ball_person_ratio=args.max_ball_person_ratio,
        visualize=args.visualize,
        use_replicate=args.use_replicate,
        replicate_api_token=args.replicate_api_token,
        replicate_hoop_fast_model=args.replicate_hoop_fast_model,
        replicate_hoop_slow_model=args.replicate_hoop_slow_model,
        replicate_clip_model=args.replicate_clip_model,
        replicate_timeout=args.replicate_timeout,
        save_artifacts=args.save_artifacts,
        upload_firebase=args.upload_firebase,
        firebase_bucket=args.firebase_bucket,
        session_id=args.session_id,
        skip_shot_idxs=args.skip_shot_idxs,
        run_shot_idxs=args.run_shot_idxs,
        annotated_hoop_points=(
            comma_sep_str_to_list(args.annotated_hoop_points_str)
            if args.annotated_hoop_points_str is not None
            else None
        ),
        override_hoop_boxes=(
            comma_sep_str_to_list(args.override_hoop_boxes_str)
            if args.override_hoop_boxes_str is not None
            else None
        ),
        device_name=args.device,
    )
    results = []
    for i in range(len(output.clips)):
        clip_i = output.clips[i]
        row = {
            "shot_idx": clip_i.shot_idx,
            "seconds": clip_i.keyframe_ball_ts,
            "outcome": clip_i.shot_outcome,
            "outcome_ball": clip_i.shot_outcome_by_ball,
            "outcome_model": clip_i.shot_outcome_by_model,
        }
        results.append(row)
    results = pd.DataFrame.from_records(results)
    result_file = join(args.out_dir, "results.csv")
    results.to_csv(result_file, index=False)
    print(f"Results saved to {result_file}")


if __name__ == "__main__":
    import torch
    from shutil import rmtree
    from os import makedirs
    from os.path import isdir
    from hooper.utils import from_yaml

    from shared_utils import Environment

    env = Environment()

    # Load the config file for defaults
    config = from_yaml(get_config_path())

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video_file", type=str, help="Path to the video file")
    parser.add_argument(
        "--paused-ts-str",
        type=str,
        default="",
        help='Paused timestamps, comma seperated (default: "")',
    )
    parser.add_argument(
        "--out-dir", type=str, default="./out", help="Path to output directory"
    )
    parser.add_argument(
        "--frame-sample-rate",
        type=int,
        default=config.get("frame_sample_rate", 2),
        help="Sample rate do inference with (default: 2)",
    )
    parser.add_argument(
        "--min-detect-shot-score",
        type=float,
        default=config.get("min_detect_shot_score", 0.5),
        help="Minimum confidence to detect a shot candidate (default: 0.5)",
    )
    parser.add_argument(
        "--min-shot-score",
        type=float,
        default=config.get("min_shot_score", 0.5),
        help="Minimum confidence to classify as a make (default: 0.5)",
    )
    parser.add_argument(
        "--min-person-box-score",
        type=float,
        default=config.get("min_person_box_score", 0.1),
        help="Minimum score to keep a person box (default: 0.1)",
    )
    parser.add_argument(
        "--min-person-keypoint-score",
        type=float,
        default=config.get("min_person_keypoint_score", 0.1),
        help="Minimum score to keep a person keypoint (default: 0.1)",
    )
    parser.add_argument(
        "--hoop-fast-stride",
        type=int,
        default=config.get("hoop_fast_stride", 15),
        help="Sample rate for video inference (default: 15)",
    )
    parser.add_argument(
        "--hoop-slow-stride",
        type=int,
        default=config.get("hoop_slow_stride", 300),
        help="Sample rate for video inference (default: 300)",
    )
    parser.add_argument(
        "--dist-pixel-thres",
        type=int,
        default=config.get("dist_pixel_thres", 10),
        help="Maximum distance between hand and ball to count possession (default: 10)",
    )
    parser.add_argument(
        "--max-ball-person-ratio",
        type=int,
        default=config.get("max_ball_person_ratio", 30),
        help="Max ratio between ball and person bbox to count possession (default: 30)",
    )
    parser.add_argument(
        "--min-hoop-fast-score",
        type=float,
        default=config.get("min_hoop_fast_score", 0.3),
        help="Minimum confidence to detect a hoop (default: 0.3)",
    )
    parser.add_argument(
        "--min-hoop-slow-score",
        type=float,
        default=config.get("min_hoop_slow_score", 0.8),
        help="Minimum confidence to detect a hoop (default: 0.8)",
    )
    parser.add_argument(
        "--min-ball-score",
        type=float,
        default=config.get("min_ball_score", 0.2),
        help="Minimum confidence to detect a hoop (default: 0.2)",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=config.get("context_window", 4),
        help="Number of seconds to sample around the window (default: 4)",
    )
    parser.add_argument(
        "--min-segment-skeleton-iou",
        type=float,
        default=config.get("min_segment_skeleton_iou", 0.6),
        help="Minimum IoU to match segment to person skeleton (default: 0.6)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Whether to visualize or not (default: False)",
    )
    parser.add_argument(
        "--use-replicate",
        action="store_true",
        default=False,
        help="Whether to do inference with replicate (default: False)",
    )
    parser.add_argument(
        "--replicate-api-token",
        type=str,
        default=env.replicate_api_token,
        help="Replicate API token (default: REPLICATE_API_TOKEN)",
    )
    parser.add_argument(
        "--replicate-hoop-fast-model",
        type=str,
        default=env.replicate_hoop_fast_model,
        help="Replicate model for hoop model (default: REPLICATE_HOOP_FAST_MODEL)",
    )
    parser.add_argument(
        "--replicate-hoop-slow-model",
        type=str,
        default=env.replicate_hoop_slow_model,
        help="Replicate model for hoop model (default: REPLICATE_HOOP_SLOW_MODEL)",
    )
    parser.add_argument(
        "--replicate-clip-model",
        dest="replicate_clip_model",
        type=str,
        default=env.replicate_clip_model,
        help="Replicate model for clip model (default: REPLICATE_CLIP_MODEL)",
    )
    parser.add_argument(
        "--replicate-timeout",
        type=int,
        default=10,
        help="Number of seconds before timeout (default: 10)",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        default=False,
        help="Whether to save predictions (default: False)",
    )
    parser.add_argument(
        "--upload-firebase",
        action="store_true",
        default=False,
        help="Whether to upload to firebase (default: False)",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session identifier (default: None)",
    )
    parser.add_argument(
        "--firebase-bucket",
        type=str,
        default=None,
        help="Firebase bucket to upload artifacts for (default: None)",
    )
    parser.add_argument(
        "--skip-shot-idxs",
        type=int,
        nargs="+",
        default=[],
        help="List of indices to skip (default: [])",
    )
    parser.add_argument(
        "--run-shot-idxs",
        type=int,
        nargs="+",
        default=[],
        help="List of indices to run (default: [])",
    )
    parser.add_argument(
        "--annotated-hoop-points-str",
        type=str,
        default=None,
        help="Annotated hoop points in the format of t1,x1,y1,t2,x2,y2,... (default: None)",
    )
    parser.add_argument(
        "--override-hoop-boxes-str",
        type=str,
        default=None,
        help="Override hoop boxes in the format of t1,x1,y1,x2,y2,t2,x3,y3,x4,y4,... (default: None)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="CUDA device (default: cuda)",
    )
    args = parser.parse_args()
    if isdir(args.out_dir):
        rmtree(args.out_dir)
    makedirs(args.out_dir, exist_ok=True)

    if (args.device == "cuda") and (not torch.cuda.is_available()):
        print("No GPU found. Setting device to cpu.")
        args.device = "cpu"

    asyncio.run(main(args))
