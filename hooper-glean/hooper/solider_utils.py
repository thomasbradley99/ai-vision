import torch
import numpy as np
from typing import Optional
import torchvision.transforms as T
from os.path import exists, join, dirname, realpath, isfile
from PIL import Image
from .reid.config import cfg
from .reid.model.make_model import make_model

CUR_DIR = dirname(__file__)
CONFIG_DIR = realpath(join(CUR_DIR, "reid/configs/msmt17"))

__model_types = ["swin_base_msmt17", "swin_small_msmt17", "swin_tiny_msmt17"]


class SoliderFeatureExtractor(object):
    r"""Helper for feature extraction for Solider ReID.
    @param model_weights (str): The path to the model weights.
    @param device (str, default='cuda'): The device to use.
    """

    def __init__(
        self,
        model_name: str,
        model_weights: str,
        device_name: str = "cuda",
        verbose: bool = True,
    ):
        preprocess, model = get_solider_reid_model(model_name, model_weights)
        model.eval()

        if verbose:
            print("Model: {}".format(model_name))

        device = torch.device(device_name)
        model.to(device)

        to_pil = T.ToPILImage()

        # Class attributes
        self.model = model
        self.preprocess = preprocess
        self.to_pil = to_pil
        self.device = device

    def __call__(self, input):
        if isinstance(input, list):
            images = []
            for element in input:
                if isinstance(element, str):
                    image = read_image(element)
                elif isinstance(element, np.ndarray):
                    image = self.to_pil(element)
                else:
                    raise NotImplementedError(
                        f"Input type {type(element)} not supported."
                    )
                image = self.preprocess(image)
                images.append(image)
            images = torch.stack(images, dim=0)
            images = images.to(self.device)
        elif isinstance(input, str):
            image = read_image(input)
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)  # type: ignore
        elif isinstance(input, np.ndarray):
            image = self.to_pil(input)
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)
        else:
            raise NotImplementedError(f"Input type {type(input)} not supported.")

        with torch.no_grad():
            features, _ = self.model(images)

        return features


def get_solider_reid_model(model_name: str, model_weights: str):
    r"""Returns a model and transform for use for ReID.
    @param model_name (str): The model to use. Options are 'base', 'small', and 'tiny'.
    @param model_weights (str): Weights for the model
    @return transforms (torchvision.transforms.Compose): A transform to apply to the image.
    @return model (nn.Module): The model to use for ReID (set to eval)
    """
    transforms = T.Compose(
        [
            T.Resize([384, 128]),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    config_path = join(CONFIG_DIR, f"swin_{model_name}.yml")
    assert isfile(config_path), f"Config file {config_path} does not exist."
    cfg.merge_from_file(config_path)
    cfg.freeze()

    assert isfile(model_weights), f"Checkpoint file {model_weights} does not exist."
    model = make_model(cfg, 10, 1, 1, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(model_weights)
    model = model.eval()

    return transforms, model


def get_solider_feature_extractor(
    model_weights: str, device: str
) -> SoliderFeatureExtractor:
    r"""Return a pre-trained Solider feature extractor."""
    model_name = get_model_name(model_weights)
    assert model_name in ["base", "tiny", "small"], (
        f"Model name {model_name} not supported"
    )
    extractor = SoliderFeatureExtractor(model_name, model_weights, device_name=device)
    return extractor


def read_image(img_path: str) -> Optional[Image.Image]:
    """Read image for solider prep.
    Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process.
    """
    got_img = False
    img = None
    if not exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert("RGB")
            got_img = True
        except IOError:
            print(
                "IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(
                    img_path
                )
            )
            pass
    if got_img:
        return img
    return None


def get_model_name(model: str) -> Optional[str]:
    model = str(model).rsplit("/", 1)[-1].split(".")[0]
    for x in __model_types:
        if x in model:
            return x.split("_")[1]
    return None
