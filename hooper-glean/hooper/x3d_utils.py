import math
import torch
import torch.nn as nn
from typing import Callable, Dict
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo


def get_x3d_shot_model(
    model_name: str = "x3d_m", weight_file: str = "SHOT-Classification/x3d_m.pt"
) -> torch.nn.Module:
    """Return predictor model for video shot classification."""
    model = torch.hub.load("facebookresearch/pytorchvideo", model_name, pretrained=True)
    model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, 1)  # type: ignore
    model.blocks[-1].activation = nn.Identity()  # type: ignore
    # Load trained parameters
    state_dict = torch.load(weight_file)
    model.load_state_dict(state_dict)  # type: ignore
    model.eval()  # type: ignore
    return model  # type: ignore


@torch.no_grad()
def x3d_preprocess_video(
    video: torch.Tensor, model_name: str = "x3d_m"
) -> torch.Tensor:
    """Preprocess video input.
    :param model_name: x3d_xs | x3d_s | x3d_m
    """
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    model_transform_params = {
        "x3d_xs": {
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 4,
            "sampling_rate": 12,
        },
        "x3d_s": {
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 13,
            "sampling_rate": 6,
        },
        "x3d_m": {
            "side_size": 256,
            "crop_size": 256,
            "num_frames": 16,
            "sampling_rate": 5,
        },
    }
    params = model_transform_params[model_name]
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(params["num_frames"]),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=params["side_size"]),
                CenterCropVideo(crop_size=(params["crop_size"], params["crop_size"])),
            ]
        ),
    )
    # Preprocess the video
    video = transform({"video": video})  # type: ignore
    video = video["video"]  # type: ignore
    return video


@torch.no_grad()
def x3d_infer_video(model, video: torch.Tensor) -> float:
    """Classify shot video to be a make or a miss.
    :return number between 0 and 1
    """
    logits = model(video[None, ...])
    probs = torch.sigmoid(logits)
    probs = probs.cpu().item()
    return probs


# --- my own implementations of pytorchvideo ---


class ApplyTransformToKey:
    """
    Applies transform to key of dictionary input.

    Args:
        key (str): the dictionary key the transform is applied to
        transform (callable): the transform that is applied

    Example:
        >>>   transforms.ApplyTransformToKey(
        >>>       key='video',
        >>>       transform=UniformTemporalSubsample(num_video_samples),
        >>>   )
    """

    def __init__(self, key: str, transform: Callable):
        self._key = key
        self._transform = transform

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x[self._key] = self._transform(x[self._key])
        return x


class ShortSideScale(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.short_side_scale``.
    """

    def __init__(
        self, size: int, interpolation: str = "bilinear", backend: str = "pytorch"
    ):
        super().__init__()
        self._size = size
        self._interpolation = interpolation
        self._backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return short_side_scale(x, self._size, self._interpolation, self._backend)


def short_side_scale(
    x: torch.Tensor,
    size: int,
    interpolation: str = "bilinear",
    backend: str = "pytorch",
) -> torch.Tensor:
    """
    Determines the shorter spatial dim of the video (i.e. width or height) and scales
    it to the given size. To maintain aspect ratio, the longer side is then scaled
    accordingly.
    Args:
        x (torch.Tensor): A video tensor of shape (C, T, H, W) and type torch.float32.
        size (int): The size the shorter side is scaled to.
        interpolation (str): Algorithm used for upsampling,
            options: nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'
        backend (str): backend used to perform interpolation. Options includes
            `pytorch` as default, and `opencv`. Note that opencv and pytorch behave
            differently on linear interpolation on some versions.
            https://discuss.pytorch.org/t/pytorch-linear-interpolation-is-different-from-pil-opencv/71181
    Returns:
        An x-like Tensor with scaled spatial dims.
    """  # noqa
    assert len(x.shape) == 4
    assert x.dtype == torch.float32
    assert backend in ("pytorch")
    c, t, h, w = x.shape
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))
    if backend == "pytorch":
        return torch.nn.functional.interpolate(
            x, size=(new_h, new_w), mode=interpolation, align_corners=False
        )
    else:
        raise NotImplementedError(f"{backend} backend not supported.")


class UniformTemporalSubsample(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.uniform_temporal_subsample``.
    """

    def __init__(self, num_samples: int, temporal_dim: int = -3):
        """
        Args:
            num_samples (int): The number of equispaced samples to be selected
            temporal_dim (int): dimension of temporal to perform temporal subsample.
        """
        super().__init__()
        self._num_samples = num_samples
        self._temporal_dim = temporal_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return uniform_temporal_subsample(x, self._num_samples, self._temporal_dim)


def uniform_temporal_subsample(
    x: torch.Tensor, num_samples: int, temporal_dim: int = -3
) -> torch.Tensor:
    """
    Uniformly subsamples num_samples indices from the temporal dimension of the video.
    When num_samples is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.

    Args:
        x (torch.Tensor): A video tensor with dimension larger than one with torch
            tensor type includes int, long, float, complex, etc.
        num_samples (int): The number of equispaced samples to be selected
        temporal_dim (int): dimension of temporal to perform temporal subsample.

    Returns:
        An x-like Tensor with subsampled temporal dimension.
    """
    t = x.shape[temporal_dim]
    assert num_samples > 0 and t > 0
    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, temporal_dim, indices)
