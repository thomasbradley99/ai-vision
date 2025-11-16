import torch
import numpy.typing as npt
from pydantic import BaseModel
from typing import Callable, Optional, List, Tuple, Dict, Any, Union

# Helper shortcuts for types
BoxType = Union[List[float], List[int]]
PointType = Union[List[float], List[int]]
BatchSelectorFunctionType = Callable[[torch.Tensor, int], int]
SelectorFunctionType = Callable[[torch.Tensor], int]
CheckFunctionType = Callable[[Any], bool]
PromptType = Tuple[
    int, int, Optional[npt.NDArray], Optional[npt.NDArray], Optional[BoxType]
]
SAM2SegmentsType = Dict[int, Dict[int, npt.NDArray]]
SAM2BoxesType = Dict[int, Dict[int, Optional[BoxType]]]
SAM2SkeletonsType = Dict[int, Dict[int, npt.NDArray]]
CompressedSegmentsType = Dict[int, Dict[int, List[List[int]]]]


class HoopOutput(BaseModel):
    ts: float
    box: BoxType
    prob: Optional[float] = None


class TagArtifact(BaseModel):
    id: int  # object id in clip
    name: str  # filename that was uploaded
    url: Optional[str] = None  # URL after uploading to firebase


class ClipArtifacts(BaseModel):
    trace_log: Optional[str] = None
    labels_json: Optional[str] = None
    embeddings_pt: Optional[str] = None
    segments_pt: Optional[str] = None
    skeletons_pt: Optional[str] = None
    viz_hoop_jpg: Optional[str] = None
    viz_paint_jpg: Optional[str] = None
    viz_ball_jpg: Optional[str] = None
    viz_person_jpg: Optional[str] = None
    median_background_jpg: Optional[str] = None
    crops_zip: Optional[str] = None
    preview_jpg: Optional[str] = None
    loading_jpg: Optional[str] = None
    highlight_mp4: Optional[str] = None


class PlayerEmbedding(BaseModel):
    person_id: int
    action_id: Optional[int] = 1
    frame_idxs: List[int]
    image_names: List[str]
    feat_name: str
    embs: List[List[float]]
    feat: List[float]
    tag_artifact: Optional[TagArtifact] = None


class ClipOutput(BaseModel):
    shot_idx: Optional[int] = None
    ball_id: int
    hoop_id: int
    start_ts: float
    end_ts: float
    clip_length: float
    keyframe_ball_ts: float
    keyframe_ball_idx: int
    keyframe_person_ts: float
    keyframe_person_idx: int
    scorer_id: Optional[int] = None
    scorer_possession_first_idx: Optional[int] = None
    scorer_possession_first_ts: Optional[float] = None
    scorer_possession_first_bbox: Optional[List[int]] = None
    scorer_possession_last_idx: Optional[int] = None
    scorer_possession_last_ts: Optional[float] = None
    scorer_possession_last_bbox: Optional[List[int]] = None
    scorer_image_url: Optional[str] = None
    shot_outcome: bool
    shot_outcome_by_ball: bool
    shot_outcome_by_model: bool
    shot_above_idx: int
    shot_below_idx: int
    shot_above_ts: float
    shot_below_ts: float
    assister_id: Optional[int] = None
    assister_possession_last_idx: Optional[int] = None
    assister_possession_last_ts: Optional[float] = None
    assister_possession_last_bbox: Optional[List[int]] = None
    assister_image_url: Optional[str] = None
    rebounder_id: Optional[int] = None
    rebounder_possession_first_idx: Optional[int] = None
    rebounder_possession_first_ts: Optional[float] = None
    rebounder_possession_first_bbox: Optional[List[int]] = None
    rebounder_image_url: Optional[str] = None
    paint_keypoints: Optional[List[int]] = None
    embeddings: List[PlayerEmbedding]
    artifacts: ClipArtifacts


class ClusterOutput(BaseModel):
    shot_idx: int
    person_id: Optional[int] = None
    cluster_id: int
    action_id: int  # 1: shot, 2: rebound, 3: assist
    image_name: Optional[str] = None
    image_url: Optional[str] = None


class KeyframeArtifact(BaseModel):
    idx: int
    ts: float
    start_ts: float
    end_ts: float
    width: int
    height: int
    url: Optional[str] = None


class VideoArtifacts(BaseModel):
    trace_log: Optional[str] = None
    fenceposts_log: Optional[str] = None
    metadata_json: Optional[str] = None
    fenceposts_pt: Optional[str] = None
    clusters_json: Optional[str] = None
    preview_jpg: Optional[str] = None


class VideoOutput(BaseModel):
    fps: float
    length: float
    width: int
    height: int
    firebase_bucket: Optional[str] = None
    clips: List[ClipOutput]
    clusters: List[ClusterOutput]
    cluster_classes: List[int]
    artifacts: VideoArtifacts


class RawClipOutput(BaseModel):
    """Used for Replicate output."""

    shot_idx: int
    output: Optional[ClipOutput] = None


class RawHoopOutput(BaseModel):
    """Used for Replicate output."""

    hoops: Optional[List[HoopOutput]] = None
    frac_missing: Optional[float] = None
    frac_moving: Optional[float] = None


class WrappedHoopOutput(BaseModel):
    hoops: Optional[List[HoopOutput]] = None
    frac_missing: Optional[float] = None
    frac_moving: Optional[float] = None
    success: Optional[bool] = True
    error: Optional[str] = None
    elapsed_sec: int


class WrappedClipOutput(BaseModel):
    shot_idx: int
    output: Optional[ClipOutput] = None
    success: Optional[bool] = True
    error: Optional[str] = None
    elapsed_sec: int
