import cv2
import ffmpeg
import shutil
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image
from os import makedirs, remove
from os.path import join, exists, isfile
from typing import List, Optional
from shared_utils.types import SAM2SegmentsType
from .utils import (
    extract_audio_ffmpeg,
    stitch_video_and_audio_ffmpeg,
    reencode_video_mp4v_to_h264,
    check_if_video_has_audio_ffmpeg,
    binary_mask_to_bounding_box,
    get_box_center,
    tick,
)


def create_thumbnail(video_file: str, out_file: str, quiet: bool = False) -> str:
    r"""Create a preview thumbnail for a video.
    :param video_file: Path to video path
    :param out_file: Path to save the preview image
    """
    if exists(out_file):
        remove(out_file)
    # Generate a preview frame
    ffmpeg.input(video_file, ss=0).output(out_file, vframes=1).run(
        overwrite_output=True,
        quiet=quiet,
    )
    return out_file


def create_highlight(
    video_file: str,
    video_width: int,
    video_height: int,
    frame_dir: str,
    frame_names: List[str],
    video_segments: SAM2SegmentsType,
    follow_id: int,
    out_dir: str,
    logger: logging.Logger,
    out_video_name: str = "highlight.mp4",
    out_preview_name: str = "preview.jpg",
    out_loading_name: str = "loading.jpg",
    preview_frame_idx: int = 0,
    loading_frame_idx: int = 0,
    smoothing_window: int = 5,
    make_preview: bool = False,
    sample_rate: int = 1,
    fps: int = 30,
) -> List[int]:
    """Create a highlight around a single shot / clip
    @param video_file: Video file (for a single clip)
    @param frame_dir: Directory holding the frame images
    @param frame_names: Filenames for all frame images
    @param video_segments: Map from frame index to identified objects
    @param follow_id: Identifier to follow across the video
    @param preview_frame_idx: Which frame to use as a preview
    @param loading_frame_idx: Which frame to use as a loading image
    @param smoothing_window: Smoothing window
    @param make_preview: if true, we make a preview image of the video
    @param sample_rate: the `frame_dir` and `frame_names` passed to us may not match with |video_segments|
                        the sample rate tells us the interval from video segment keys to frames
    @param out_video_file: The generated highlight will be saved to this file
    @param out_preview_file: A preview of the highlight video will be saved here
    @param out_loading_file: A loading image for the highlight video will be saved here
    @param logger: Logger to use for logging
    """
    makedirs(out_dir, exist_ok=True)

    # To prevent super skinny videos, take a 9:16 ratio (instagram)
    highlight_width = int(video_height * 9 / 16)
    has_audio = check_if_video_has_audio_ffmpeg(video_file)
    num_frames = len(frame_names)

    if (preview_frame_idx < 0) or (preview_frame_idx > (len(frame_names) - 1)):
        # block any bad choices
        preview_frame_idx = 0

    audio_file = None
    if has_audio:
        # Extract audio for subclip
        audio_file = join(out_dir, "audio.mp3")
        if not exists(audio_file):
            start_time = tick()
            extract_audio_ffmpeg(video_file, audio_file)
            end_time = tick()
            logger.info(
                f"> extracted audio from video - {end_time - start_time}s elapsed"
            )

    # Loop through segments to find which center point to follow
    raw_centers: List[Optional[int]] = []

    start_time = tick()
    # Do a loop to find all the boxes
    for i in range(num_frames):
        if i % sample_rate != 0:  # if we didn't process this frame, skip
            raw_centers.append(None)
        else:  # find the right segment
            segments = video_segments[i // sample_rate]
            if follow_id in segments:
                mask = segments[follow_id]
                bbox = binary_mask_to_bounding_box(mask[0])
                if bbox is None:
                    raw_centers.append(None)
                else:
                    point = get_box_center(*bbox[:4])
                    x = int(point[0])
                    # Make sure not out of bounds
                    if x < (highlight_width // 2):
                        x = highlight_width // 2
                    elif (x + (highlight_width // 2)) > video_width:
                        x = video_width - highlight_width // 2
                    raw_centers.append(x)
            else:
                raw_centers.append(None)
    end_time = tick()
    logger.info(f"> determined camera focal points - {end_time - start_time}s elapsed")

    assert len(raw_centers) == len(frame_names), (
        "Expected |raw_centers| = |frame_names|"
    )

    # interpolate and smooth the focus point
    start_time = tick()
    centers = interpolate_centers(raw_centers)
    centers = moving_average_smoothing(centers, window_size=smoothing_window)
    end_time = tick()
    logger.info(f"> smoothing focal points - {end_time - start_time}s elapsed")

    noaudio_video_file: str = join(out_dir, "noaudio.mp4")
    # Directly write images as a sequence to video
    # https://fourcc.org/codecs.php
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        noaudio_video_file, fourcc, fps, (highlight_width, video_height)
    )
    if not writer.isOpened():
        logger.error("> failed to initialize VideoWriter with 'mp4v' codec")
        raise RuntimeError(
            "> failed to initialize VideoWriter with 'mp4v' codec. Quitting early..."
        )

    start_time = tick()
    for i in tqdm(range(num_frames), desc="writing frames"):
        im = Image.open(join(frame_dir, frame_names[i]))

        # Compute boundaries
        center = centers[i]
        top = video_height
        bottom = 0
        left = center - (highlight_width // 2)
        right = center + (highlight_width // 2)

        # Make a crop centered at this point
        im = im.crop((left, bottom, right, top))
        im_bgr_i = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        writer.write(im_bgr_i)

        # Make preview image of middle
        if make_preview and (i == preview_frame_idx):
            im.save(join(out_dir, out_preview_name))
        # Make loading image of middle
        if make_preview and (i == loading_frame_idx):
            im.save(join(out_dir, out_loading_name))

    writer.release()
    end_time = tick()
    logger.info(f"> compiled highlight frames - {end_time - start_time}s elapsed")

    # Combine the video and audio
    mp4v_video_file: str = join(out_dir, "highlight.raw.mp4")
    if has_audio and (audio_file is not None):
        start_time = tick()
        stitch_video_and_audio_ffmpeg(noaudio_video_file, audio_file, mp4v_video_file)
        end_time = tick()
        logger.info(
            f"> added audio to the highlight video - {end_time - start_time}s elapsed"
        )
    else:
        shutil.copyfile(noaudio_video_file, mp4v_video_file)

    # Reencode the video to h264
    out_video_file = join(out_dir, out_video_name)
    start_time = tick()
    reencode_video_mp4v_to_h264(mp4v_video_file, out_video_file)
    end_time = tick()
    logger.info(f"> converted mp4v to h.264 - {end_time - start_time}s elapsed")

    # Clean up and delete the temporary artifacts made
    if isfile(noaudio_video_file):
        remove(noaudio_video_file)
    if has_audio and audio_file and isfile(audio_file):
        remove(audio_file)
    if isfile(mp4v_video_file):
        remove(mp4v_video_file)

    return centers


def interpolate_centers(raw_centers: List[Optional[int]]) -> List[int]:
    """Interpolate points.
    @param raw_centers: List of center points but with missing entries.
    @return: List of center points with missing entries interpolated.
    """
    # Find the indices with non-None values
    valid_indices = [i for i, val in enumerate(raw_centers) if val is not None]
    # If there are no valid indices, return the original list as it can't be interpolated
    if not valid_indices:
        return []
    centers = []
    # Interpolate values between valid indices
    for i in range(len(raw_centers)):
        if raw_centers[i] is None:
            # Find the nearest valid indices to the left and right
            left = max((index for index in valid_indices if index < i), default=None)
            right = min((index for index in valid_indices if index > i), default=None)
            if left is not None and right is not None:
                raw_center_left, raw_center_right = (
                    raw_centers[left],
                    raw_centers[right],
                )
                assert raw_center_left is not None and raw_center_right is not None
                # Linear interpolation between the two nearest valid points
                center_i = raw_center_left + (raw_center_right - raw_center_left) * (
                    i - left
                ) // (right - left)
            elif left is not None:
                raw_center_left = raw_centers[left]
                assert raw_center_left is not None
                # Fill with the nearest left value
                center_i = raw_center_left
            elif right is not None:
                raw_center_right = raw_centers[right]
                assert raw_center_right is not None
                # Fill with the nearest right value
                center_i = raw_center_right
            else:
                raise Exception("this should not be possible")
            centers.append(center_i)
        else:
            centers.append(raw_centers[i])
    return centers


def moving_average_smoothing(centers: List[int], window_size: int = 3) -> List[int]:
    """Smooth the centers.
    @param centers: List of coordinates
    @param window_size: Window for smoothing
    """
    smoothed_centers = []
    n = len(centers)
    # Handle the case where the window size is greater than the number of centers
    if window_size > n:
        window_size = n
    # Calculate the moving average
    for i in range(n):
        # Define the window range
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        # Collect non-None values within the window range
        window_values = [
            centers[j] for j in range(start, end) if centers[j] is not None
        ]
        # Calculate the average of the window values
        if window_values:
            smoothed_value = sum(window_values) // len(window_values)
            smoothed_centers.append(smoothed_value)
        else:
            smoothed_centers.append(centers[i])
    return smoothed_centers
