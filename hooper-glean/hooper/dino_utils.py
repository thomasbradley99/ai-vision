import torch
from PIL import Image
from typing import Tuple, List
from transformers.image_processing_utils import BaseImageProcessor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from shared_utils.types import BoxType
from .utils import format_bbox



def load_dino_model(
    model_name: str = "IDEA-Research/grounding-dino-base",
    device_name: str = "cuda:0",
) -> Tuple[torch.nn.Module, BaseImageProcessor]:
    device = torch.device(device_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
    model = model.to(device)
    model = model.eval()
    return model, processor


def detect_balls_with_dino(
    model: torch.nn.Module,
    processor: BaseImageProcessor,
    image_paths: List[str],
    device_name: str = "cuda",
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
) -> List[List[BoxType]]:
    """Batched ball detection using a pretrained Grounding DINO
    @param image_paths: List of paths to frames
    @param box_threshold: Score threshold for detecting a bounding box
    @return boxes: [[x1, y1, x2, y2, score]...]
    """
    device = torch.device(device_name)
    pil_images = [Image.open(path) for path in image_paths]
    texts = ["basketball." for _ in range(len(image_paths))]
    inputs = processor(images=pil_images, text=texts, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # compute sizes - important to reverse
    target_sizes = torch.tensor([img.size[::-1] for img in pil_images], device=device)
    results = processor.post_process_grounded_object_detection(  # type: ignore
        outputs,
        inputs["input_ids"],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )
    preds = []
    for i in range(len(results)):
        result = results[i]
        preds_i = []
        for score, box in zip(result["scores"], result["boxes"]):
            score = score.item()
            pred = box.tolist() + [round(score, 3)]
            preds_i.append(format_bbox(pred))
        preds.append(preds_i)
    return preds
