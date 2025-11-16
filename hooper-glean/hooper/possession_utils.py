import json
from os.path import join
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from transformers import CLIPVisionModel
from torchvision import transforms

from shared_utils.types import BoxType

from .utils import get_checkpoints_dir



def get_possession_model(
    weight_file: str = "POSSESSION-Classification/emb.200epochs.20candidates.pt",
    params_file: str = "POSSESSION-Classification/hparams.json",
) -> "HolderFrameClassifier":
    """Loads a pretrained possession model.
    @param weight_file: Path to the weight file.
    @param params_file: Path to the parameter file.
    """
    checkpoint_path = join(get_checkpoints_dir(), weight_file)
    hparams_path = join(get_checkpoints_dir(), params_file)
    with open(hparams_path, "r") as f:
        hparams = json.load(f)
    max_candidates = hparams["max_candidates"]
    state_dict = torch.load(checkpoint_path)
    model = HolderFrameClassifier(in_dim=768, dim=256, dropout_prob=0.2, pooling=None)
    model.set_max_candidates(max_candidates)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_clip_vision_model(
    model: str = "openai/clip-vit-base-patch32",
) -> torch.nn.Module:
    """Loads a pretrained CLIP vision model.
    @param model: Name of the CLIP model to load.
    """
    image_encoder = CLIPVisionModel.from_pretrained(model)
    image_encoder.eval()
    return image_encoder


@torch.no_grad()
def infer_possessor(
    possession_model: "HolderFrameClassifier",
    clip_model: "CLIPVisionModel",
    img_transform: "transforms.Compose",
    img_path: str,
    box_candidates: List[BoxType],
    box_ball: Optional[BoxType] = None,
    max_candidates: int = 10,
) -> Tuple[bool, Optional[float], Optional[int]]:
    """Infer the possessor from an image frame.
    :note: If no ball is detected, then we return None for the possessor.
    :return: A dictionary with keys -
        pred_has_holder: Bool
        prob_has_holder: Optional[Float]
        pred_holder_idx: Optional[Int]
    """
    if box_ball is None:
        return False, None, None

    device = clip_model.device
    num_candidates = len(box_candidates)

    if num_candidates == 0:
        return False, None, None

    # Read the image
    img = Image.open(img_path)
    width, height = img.size

    # Format the inputs
    box_ball_t = preprocess_ball(box_ball, width, height)
    box_candidates_t = preprocess_candidates(
        box_candidates, width, height, max_candidates=max_candidates
    )
    box_candidates_t = box_candidates_t.to(device)

    # img - 1 x c x h x w
    # box_ball_t - 1 x 4
    # box_candidates_t - 1 x max_candidates x 4
    img = img_transform(img).unsqueeze(0).to(device)  # type: ignore
    box_ball_t = box_ball_t.unsqueeze(0).to(device)
    box_candidates_t = box_candidates_t.unsqueeze(0).to(device)

    # emb_img - 1 x 768
    emb_img = clip_model(pixel_values=img).pooler_output
    # emb_ball - 1 x 256
    # emb_candidates - 1 x max_candidates x 256
    # logits_has_holder - 1 x 1
    emb_ball, emb_candidates, logits_has_holder, _ = possession_model(
        emb_img, box_ball, box_candidates
    )
    prob_has_holder: float = torch.sigmoid(logits_has_holder).item()
    pred_has_holder: bool = prob_has_holder > 0.5

    if pred_has_holder:
        # valid_candidates = 1 x num_candidates x 256
        valid_candidates = emb_candidates[:, :num_candidates, :]
        pred_holder_idx = find_closest_candidate_euclidean(emb_ball, valid_candidates)
    else:
        pred_holder_idx = None

    return pred_has_holder, prob_has_holder, pred_holder_idx


@torch.no_grad()
def infer_possessor_batch(
    possession_model: "HolderFrameClassifier",
    clip_model: "CLIPVisionModel",
    img_transform: "transforms.Compose",
    batch_img_path: List[str],
    batch_box_candidates: List[List[BoxType]],
    batch_box_ball: List[Optional[BoxType]],
    max_candidates: int = 10,
    conf_threshold: float = 0.5,
) -> Tuple[List[bool], List[Optional[float]], List[Optional[int]]]:
    """Infer the possessor from a batch of image frames."""
    assert len(batch_img_path) == len(batch_box_candidates)
    assert len(batch_img_path) == len(batch_box_ball)

    batch_size = len(batch_img_path)
    device = clip_model.device

    imgs: List[torch.Tensor] = []
    balls: List[torch.Tensor] = []
    candidates: List[torch.Tensor] = []
    num_candidates: List[int] = []
    for img_path, box_ball, box_candidates in zip(
        batch_img_path, batch_box_ball, batch_box_candidates
    ):
        img = Image.open(img_path)
        width, height = img.size
        num_candidate = len(box_candidates)

        img_t = img_transform(img)  # type: ignore
        if box_ball is None:
            box_ball_t = torch.Tensor([0.0, 0.0, 0.0, 0.0]).float()
        else:
            box_ball_t = preprocess_ball(box_ball, width, height)
        box_candidates_t = preprocess_candidates(
            box_candidates, width, height, max_candidates=max_candidates
        )
        box_candidates_t = box_candidates_t.to(device)

        imgs.append(img_t)  # type: ignore
        balls.append(box_ball_t)
        candidates.append(box_candidates_t)
        num_candidates.append(num_candidate)

    # imgs - |batch| x 3 x h x w
    # balls - |batch| x 4
    # candidates - |batch| x max_candidates x 4
    imgs = torch.stack(imgs).to(device)  # type: ignore
    balls = torch.stack(balls).to(device)  # type: ignore
    candidates = torch.stack(candidates).to(device)  # type: ignore

    # emb_img - |batch| x 768
    emb_imgs = clip_model(pixel_values=imgs).pooler_output
    # emb_ball - |batch| x 256
    # emb_candidates - |batch| x max_candidates x 256
    # logits_has_holder - |batch| x 1
    emb_balls, emb_candidates, logits_has_holders, _ = possession_model(
        emb_imgs, balls, candidates
    )

    prob_has_holders = (
        torch.sigmoid(logits_has_holders).squeeze(1).cpu().numpy().tolist()
    )
    pred_has_holders = [p > conf_threshold for p in prob_has_holders]

    pred_holder_idxs = []
    for i in range(batch_size):
        if num_candidates[i] == 0:
            pred_holder_idx = None
            pred_has_holders[i] = False  # override with false
            prob_has_holders[i] = None
        else:
            # We do the inference regardless of `pred_has_holders[i]`. Outside logic can filter.
            # valid_candidates = 1 x |<max_candidates| x 256
            valid_candidates = emb_candidates[i, : num_candidates[i], :].unsqueeze(0)
            pred_holder_idx = find_closest_candidate_euclidean(
                emb_balls[i].unsqueeze(0), valid_candidates
            )
        pred_holder_idxs.append(pred_holder_idx)

    return pred_has_holders, prob_has_holders, pred_holder_idxs


def get_possession_model_transforms() -> "transforms.Compose":
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    eval_transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return eval_transform


def normalize_box(box: BoxType, width: int, height: int) -> BoxType:
    x0, y0, x1, y1 = box
    x0 = x0 / float(width)
    x1 = x1 / float(width)
    y0 = y0 / float(height)
    y1 = y1 / float(height)
    return [x0, y0, x1, y1]


def normalize_boxes(boxes: List[BoxType], width: int, height: int) -> List[BoxType]:
    return [normalize_box(box, width, height) for box in boxes]


def pad_with_zeros(box_candidates: torch.Tensor, max_size: int = 10) -> torch.Tensor:
    if len(box_candidates) > max_size:
        box_candidates = box_candidates[:max_size, :]
    elif len(box_candidates) < max_size:
        dims = box_candidates.size(1)
        device = box_candidates.device
        pad_size = max_size - len(box_candidates)
        padding = torch.zeros(pad_size, dims, device=device)
        box_candidates = torch.cat((box_candidates, padding), dim=0)
    return box_candidates


def preprocess_ball(box_ball: BoxType, img_width: int, img_height: int) -> torch.Tensor:
    box = normalize_box(box_ball, img_width, img_height)
    box = torch.Tensor(box).float()
    return box


def preprocess_candidates(
    box_candidates: List[BoxType],
    img_width: int,
    img_height: int,
    max_candidates: int = 10,
) -> torch.Tensor:
    """Pad candidate embeddings to an expected number."""
    if len(box_candidates) == 0:
        boxes = torch.zeros(max_candidates, 4).float()
    else:
        boxes = normalize_boxes(box_candidates, img_width, img_height)
        boxes = torch.Tensor(boxes).float()
        boxes = pad_with_zeros(boxes, max_size=max_candidates)
    return boxes


def find_closest_candidate_euclidean(
    emb_ball: torch.Tensor, emb_candidates: torch.Tensor
) -> int:
    r"""Find the closest candidate to the ball using Euclidean distance.
    Args:
        emb_ball (torch.Tensor, shape: [1, 256]): Embedding of the ball.
        emb_candidates (torch.Tensor, shape: [10, 256]): Embeddings of the candidates.
    Returns:
        int: Index of the closest candidate.
    """
    # Reshape emb_ball to [1, 1, 256] to enable broadcasting
    emb_ball_expanded = emb_ball.unsqueeze(1)  # Shape: [1, 1, 256]
    # Calculate squared Euclidean distance between ball and each candidate
    # This will have shape [1, 10]
    distances = torch.sum((emb_ball_expanded - emb_candidates) ** 2, dim=2)
    # Find the index of the candidate with minimum distance
    min_distance, min_idx = torch.min(distances, dim=1)
    return int(min_idx.item())


# --- model definitions ---


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_prob):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.layers(x)  # Residual connection


class AttentionPool(nn.Module):
    def __init__(self, in_dim=256):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, 1, kernel_size=1)

    def forward(self, x):
        # x shape: [batch, channels, height, width]
        attention = torch.softmax(self.conv(x).view(x.size(0), -1), dim=1)
        attention = attention.view(x.size(0), 1, x.size(2), x.size(3))
        weighted = x * attention
        return weighted.sum(dim=[2, 3])


class PyramidPooling(nn.Module):
    def __init__(self, in_dim=256, out_dim=512):
        super().__init__()
        self.levels = [1, 2, 4, 8]
        self.projection = nn.Linear(
            in_dim * sum([level**2 for level in self.levels]), out_dim
        )

    def forward(self, x):
        features = []
        for level in self.levels:
            pool = nn.AdaptiveMaxPool2d(output_size=(level, level))
            y = pool(x)
            y = y.view(x.size(0), x.size(1), -1)
            features.append(y)
        concat = torch.cat([f for f in features], dim=2)
        return self.projection(concat.flatten(1))


class HolderFrameClassifier(nn.Module):
    """Model on individual frames to classify which person is holding the ball in a sequence of frames.

    :param dim: Hidden dimension - used whenever possible
    :param dropout_prob: Probability of dropout across models
    :param pooling: attention | pyramid | average | None
    :note: We assume to use intermediate SAM2 image features - (batch_size, 256, height/16, width/16)
    """

    def __init__(
        self,
        in_dim: int,
        dim: int = 512,
        dropout_prob: float = 0.2,
        pooling: Optional[str] = "pyramid",
    ):
        super().__init__()
        # [batch_size, 256, height/16, width/16] -> [batch_size, 256]
        if pooling == "attention":
            self.pooler = AttentionPool(in_dim=in_dim)
        elif pooling == "pyramid":
            self.pooler = PyramidPooling(in_dim=in_dim, out_dim=dim)
        elif pooling == "average":  # just default average
            self.pooler = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        elif pooling is None:
            self.pooler = nn.Linear(in_dim, dim)
        else:
            raise Exception(f"Pooling {pooling} not supported")
        # Model to embed the bounding box of a ball
        # f(ball | image, bbox)
        self.ball_embedder = nn.Sequential(
            nn.Linear(4 + dim, dim),
            ResidualBlock(dim, dropout_prob=dropout_prob),
            ResidualBlock(dim, dropout_prob=dropout_prob),
        )
        # Model to embed the bounding box of a person
        # f(person | image, bbox)
        self.person_embedder = nn.Sequential(
            nn.Linear(4 + 2 + dim, dim),
            ResidualBlock(dim, dropout_prob=dropout_prob),
            ResidualBlock(dim, dropout_prob=dropout_prob),
        )
        # Predict whether the ball in an image has a holder
        self.has_holder_classifier = nn.Linear(dim, 1)
        # Predict whether the ball in an image is visible
        self.ball_visible_classifier = nn.Linear(dim, 1)
        self.pooling = pooling
        self.dim = dim
        # Save the maximum candidates
        self.max_candidates = None

    def set_max_candidates(self, max_candidates: int):
        self.max_candidates = max_candidates

    def forward(
        self,
        emb_image: torch.Tensor,
        box_ball: torch.Tensor,
        box_candidates: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param emb_image      : batch_size x 256 x H/16 x W/16
        :param box_ball       : batch_size x 4
                                Expected to be normalized relative to the image frame
                                Defaults to [0,0,0,0] if not present.
                                FORMAT: XYWH
        :param box_candidates : batch_size x candidate_size x 4
                                Expected to be normalized relative to the image frame
                                Defaults to [0,0,0,0] if not present.
                                FORMAT: XYWH
        :return emb_ball      : batch_size x dim
                                Contextual embedding of a ball bounding box in an image
        :return emb_candidates: batch_size x candidate_size x dim
                                Contextual embedding the holder bounding box in an image
        :return logits_has_holder : batch_size
                                    Probability that the frame has a holder of the ball
        :return logits_has_ball   : batch_size
                                    Probability that the frame has a visible ball
        """
        batch_size, candidate_size, _ = box_candidates.size()
        # Format image features
        emb_image = self.pooler(emb_image)
        box_candidates_flat = box_candidates.view(batch_size * candidate_size, 4)
        # Compute position features
        rel_candidates = get_relative_position_features(box_ball, box_candidates)
        rel_candidates_flat = rel_candidates.view(batch_size * candidate_size, 2)
        # Embed bounding box of ball
        # emb_ball - batch_size x dim
        emb_ball = self.ball_embedder(torch.concat([box_ball, emb_image], dim=-1))
        # Embed bounding box of persons
        # emb_images_dup - batch_size*candidate_size x 256
        # emb_candidates_flat - batch_size*candidate_size x dim
        emb_images_dup = (
            emb_image.unsqueeze(1)
            .repeat(1, candidate_size, 1)
            .view(batch_size * candidate_size, self.dim)
        )
        emb_candidates_flat = self.person_embedder(
            torch.concat(
                [box_candidates_flat, rel_candidates_flat, emb_images_dup], dim=-1
            )
        )
        emb_candidates = emb_candidates_flat.view(batch_size, candidate_size, self.dim)
        # Predict if there is a holder
        logits_has_holder = self.has_holder_classifier(emb_ball)
        # Predict if the ball is visible
        logits_has_ball = self.ball_visible_classifier(emb_ball)
        return emb_ball, emb_candidates, logits_has_holder, logits_has_ball

    def loss(
        self,
        emb_ball: torch.Tensor,  # batch_size x dim
        emb_candidates: torch.Tensor,  # batch_size x candidate_size x dim
        logits_has_holder: torch.Tensor,  # batch_size x 1
        logits_has_ball: torch.Tensor,  # batch_size x 1
        label_holder_idx: torch.Tensor,  # batch_size
        label_has_holder: torch.Tensor,  # batch_size
        label_has_ball: torch.Tensor,  # batch_size
        label_num_candidates: torch.Tensor,  # batch_size (0 -> candidate_size)
        temperature: float = 0.07,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss function:
        (1) Cross-batch CLIP on top of ball/person embeddings
        (2) In-frame CLIP on top of ball/person embeddings
        (3) Predicting if the ball has a holder
        (4) Predicting if the ball is visible
        """
        emb_holder = select_holder_indices_3d(emb_candidates, label_holder_idx)
        loss1 = cross_batch_clip_loss(
            emb_ball, emb_holder, label_has_holder, temperature=temperature
        )
        loss2 = in_frame_clip_loss(
            emb_ball,
            emb_candidates,
            label_holder_idx,
            label_has_holder,
            label_num_candidates,
            temperature=temperature,
        )
        loss3 = F.binary_cross_entropy_with_logits(
            logits_has_holder, label_has_holder.float().unsqueeze(1)
        )
        loss4 = F.binary_cross_entropy_with_logits(
            logits_has_ball, label_has_ball.float().unsqueeze(1)
        )
        return loss1, loss2, loss3, loss4


def in_frame_clip_loss(
    emb_ball: torch.Tensor,  # batch_size x dim
    emb_candidates: torch.Tensor,  # batch_size x candidate_size x dim
    label_holder_idx: torch.Tensor,  # batch_size
    label_has_holder: torch.Tensor,  # batch_size
    label_num_candidates: torch.Tensor,  # batch_size
    temperature: float = 0.07,
) -> torch.Tensor:
    """Calculate the CLIP loss between holder and other players in frame.
    :param emb_ball: Ball embeddings
    :param emb_candidates: All candidate person embeddings
    :param label_holder_idx: Index of the true holder among candidates
    :param label_has_holder: Whether each example has a holder
    :param label_num_candidates: Number of candidates for each element in batch
    :param temperature: Temperature for similarity scaling
    """
    # Check if there are any holders in this batch
    if label_has_holder.sum() == 0 or label_num_candidates.sum() == 0:
        # Return zero loss if there are no holders
        return torch.tensor(0.0, device=emb_ball.device)
    emb_ball = F.normalize(emb_ball, dim=-1)  # batch_size x dim
    emb_candidates = F.normalize(
        emb_candidates, dim=-1
    )  # batch_size x candidate_size x dim
    candidate_size = emb_candidates.size(1)
    # Similarities between ball and all other candidates
    logits = torch.bmm(emb_candidates, emb_ball.unsqueeze(-1)).squeeze(-1) / temperature
    # Create a mask to factor in the # of candidates
    # shape: batch_size x 1
    mask = torch.arange(candidate_size, device=logits.device).unsqueeze(
        0
    ) < label_num_candidates.unsqueeze(1)
    logits = logits.masked_fill(~mask, -1e9)  # apply a loarge number
    loss = F.cross_entropy(logits, label_holder_idx, reduction="none")
    # Only compute loss for examples with holders
    loss = (loss * label_has_holder).sum() / label_has_holder.sum()
    return loss


def cross_batch_clip_loss(
    emb_ball: torch.Tensor,
    emb_holder: torch.Tensor,
    label_has_holder: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Calculate the CLIP loss between a ball and its holder.
    This ends up comparing holders of other examples and is not effective.
    :param emb_ball         : batch_size x dim
    :param emb_holder       : batch_size x dim
    :param label_has_holder : batch_size
    :param temperature      : Smoothing hyperparameter
    :return loss            : CLIP loss
    """
    # Check if there are any holders in this batch
    if label_has_holder.sum() == 0:
        # Return zero loss if there are no holders
        return torch.tensor(0.0, device=emb_ball.device)
    emb_ball = F.normalize(emb_ball, dim=-1)  # batch_size x dim
    emb_holder = F.normalize(emb_holder, dim=-1)  # batch_size x dim
    # Compute similarities via dot product        # batch_size x batch_size
    logits = (emb_ball @ emb_holder.t()) / temperature
    # Make index labels
    batch_size = emb_ball.shape[0]
    labels = torch.arange(batch_size, device=emb_ball.device)
    # For ball->holder direction
    loss1 = F.cross_entropy(logits, labels, reduction="none")
    loss1 = (loss1 * label_has_holder).sum() / label_has_holder.sum()
    # For holder->ball direction
    loss2 = F.cross_entropy(logits.t(), labels, reduction="none")
    loss2 = (loss2 * label_has_holder).sum() / label_has_holder.sum()
    # Compute a mix of the two
    loss = (loss1 + loss2) / 2
    return loss


def get_relative_position_features(
    box_ball: torch.Tensor, box_candidates: torch.Tensor
) -> torch.Tensor:
    """Compute relative positioning.

    :param box_ball: batch_size x 4 (XYWH)
    :param box_candidates: batch_size x candidate_size x 4 (XYWH)
    """
    center_ball = box_ball[..., :2].unsqueeze(1)  # batch_size x 1 x 2
    center_candidates = box_candidates[..., :2]  # batch_size x candidate_size x 2
    rel_pos = center_candidates - center_ball  # batch_size x candidate_size x 2
    return rel_pos


def get_area_features(box: torch.Tensor) -> torch.Tensor:
    """Compute area features.

    :param box: batch_size x 4 (XYWH)
    """
    area = box[..., 2] * box[:, 3]  # batch_size
    return area.unsqueeze(-1)


def select_holder_indices_3d(
    candidates: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """
    candidates: batch_size x candidate_size x dim
    indices   : batch_size
    """
    gather_idx = indices.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
    gather_idx = gather_idx.expand(-1, -1, candidates.size(-1))  # [batch_size, 1, dim]
    selected_embeddings = torch.gather(
        candidates, 1, gather_idx
    )  # [batch_size, 1, dim]
    selected_embeddings = selected_embeddings.squeeze(1)  # [batch_size, dim]
    return selected_embeddings
