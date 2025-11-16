import torch
from os.path import join
from typing import Tuple, Dict, Any
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from .utils import get_checkpoints_dir


def get_detectron2_skeleton_model(
    device: str = "cpu",
    batch_mode: bool = False,
    config_file: str = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
    weight_file: str = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
) -> Tuple[DefaultPredictor, Dict[str, Any]]:
    """Return predictor model for COCO person keypoint detection.
    @param device (str, default='cpu'): Device to run the model on
    @param batch_mode (bool, default=False): Determines the predictor to use
    Notes:
    --
    See https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md for different model versions.
    We are going to use a very large model.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weight_file)
    cfg.MODEL.DEVICE = device
    predictor = get_predictor(cfg, batch_mode)
    return predictor, cfg


def get_detectron2_hoop_model(
    device: str = "cpu",
    batch_mode: bool = False,
    config_file: str = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    weight_file: str = "HOOP-Detection/faster_rcnn_R_101_FPN_3x.pth",
) -> Tuple[DefaultPredictor, Dict[str, Any]]:
    """Return predictor model for hoop detection.
    @param device (str, default='cpu'): Device to run the model on
    @param batch_mode (bool, default=False): Determines the predictor to use
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Use a model we pretrained on a basketball dataset
    cfg.MODEL.WEIGHTS = join(get_checkpoints_dir(), weight_file)
    cfg.MODEL.DEVICE = device
    predictor = get_predictor(cfg, batch_mode)
    return predictor, cfg


def get_predictor(cfg, batch_mode: bool = False):
    r"""Get a predictor class to do inference using Detectron2 models.
    @note The `DefaultPredictor` is used for single image inference.
    """
    return BatchPredictor(cfg) if batch_mode else DefaultPredictor(cfg)


class BatchPredictor(DefaultPredictor):
    r"""End-to-end predictory with a given config that runs on
    single device for a batch of input images.
    """

    def __call__(self, batch_images):
        r"""
        @param batch_images (List[np.ndarray]): a list of images of shape (H, W, C) (in BGR order).
        @return predictions (List[dict]): outputs for each images in order
        @note based on https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/defaults.py
        """
        with torch.no_grad():
            batch_inputs = self.preprocess_images(batch_images)
            predictions = self.infer(batch_inputs)
            return predictions

    def infer(self, batch_inputs):
        r"""Alternative to __call__ that does not preprocess images.
        @param batch_inputs (List[dict]): a list of prepped images inputs
            - image (torch.Tensor): image tensor of shape (C, H, W). Fields:
            - height (int): original image height
            - width (int): original image width
        """
        with torch.no_grad():
            predictions = self.model(batch_inputs)
            return predictions

    def preprocess_images(self, batch_images):
        r"""One of the expensive parts of inference is image i/o.
        @note Expose this function so we may be able to share images across Detectron2 models.
        @param batch_images (List[np.ndarray]): a list of images of shape (H, W, C) (in BGR order).
        @return batch_inputs (List[dict]): a list of prepped images inputs
            - image (torch.Tensor): image tensor of shape (C, H, W). Fields:
            - height (int): original image height
            - width (int): original image width
        """
        # @note https://github.com/facebookresearch/detectron2/issues/282
        batch_inputs = []
        for original_image in batch_images:
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            # Move to device for faster inference
            image = image.to(self.cfg.MODEL.DEVICE)
            inputs = {"image": image, "height": height, "width": width}
            batch_inputs.append(inputs)

        return batch_inputs


def filter_low_confidence_hoops(preds, min_score: float = 0.8):
    """Remove hoops lower than 80% confidence."""
    hoops = preds["instances"]
    hoops = hoops[hoops.scores >= min_score]
    return {"instances": hoops}
