import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.models.base_model import BaseModel
from gluefactory.utils.tools import Timer
from gluefactory.models.extractors.jpldd.backbone_encoder import AlikedEncoder, aliked_cfgs
from gluefactory.models.extractors.jpldd.descriptor_head import SDDH
from gluefactory.models.extractors.jpldd.keypoint_decoder import SMH
from gluefactory.models.extractors.jpldd.keypoint_detection import DKD
from gluefactory.models.extractors.jpldd.utils import InputPadder, change_dict_key

to_ctr = OmegaConf.to_container  # convert DictConfig to dict
aliked_checkpoint_url = "https://github.com/Shiaoming/ALIKED/raw/main/models/{}.pth"
logger = logging.getLogger(__file__)


def renormalize_keypoints(keypoints, img_wh):
    if isinstance(keypoints, torch.Tensor):
        return img_wh * (keypoints + 1.0) / 2.0
    elif isinstance(keypoints, list):
        for i in range(len(keypoints)):
            keypoints[i] = img_wh * (keypoints[i] + 1.0) / 2.0
        return keypoints


class JointPointLineDetectorDescriptor(BaseModel):
    # currently contains only ALIKED
    default_conf = {
        # ToDo: create default conf once everything is running -> default conf is merged with input conf to the init method!
        "model_name": "aliked-n16",
        "max_num_keypoints": -1,
        "detection_threshold": 0.2,
        "force_num_keypoints": False,
        "pretrained": True,
        "nms_radius": 2,
        "timeit": True,  # override timeit: False from BaseModel
        "train_descriptors": {
            "do": True,  # if train is True, initialize ALIKED Light model form OTF Descriptor GT
            "device": None  # device to house the lightweight ALIKED model
        }
    }

    n_limit_max = 20000  # taken from ALIKED which gives max num keypoints to detect!

    required_data_keys = ["image"]

    def _init(self, conf):
        logger.debug(f"final config dict(type={type(conf)}): {conf}")
        # c1-c4 -> output dimensions of encoder blocks, dim -> dimension of hidden feature map
        # K=Kernel-Size, M=num sampling pos
        aliked_model_cfg = aliked_cfgs[conf.model_name]
        dim = aliked_model_cfg["dim"]
        K = aliked_model_cfg["K"]
        M = aliked_model_cfg["M"]
        # Load Network Components
        self.encoder_backbone = AlikedEncoder(aliked_model_cfg)
        self.keypoint_and_junction_branch = SMH(dim)  # using SMH from ALIKE here
        self.dkd = DKD(radius=conf.nms_radius,
                       top_k=-1 if conf.detection_threshold > 0 else conf.max_num_keypoints,
                       scores_th=conf.detection_threshold,
                       n_limit=(
                           conf.max_num_keypoints
                           if conf.max_num_keypoints > 0
                           else self.n_limit_max
                       ), )  # Differentiable Keypoint Detection from ALIKE
        # Keypoint and line descriptors
        self.descriptor_branch = SDDH(dim, K, M, gate=nn.SELU(inplace=True), conv2D=False, mask=False)
        self.line_descriptor = torch.lerp  # we take the endpoints of lines and interpolate to get the descriptor
        # Line Attraction Field information (Line Distance Field and Angle Field)
        self.distance_field_branch = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(),
        )
        self.angle_field_branch = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        # ToDo Figure out heuristics
        # self.line_extractor = LineExtractor(torch.device("cpu"), self.line_extractor_cfg)

        # load pretrained_elements if wanted (for now that only the ALIKED parts of the network)
        if conf.pretrained:
            logger.info("Load pretrained weights for aliked parts...")
            old_test_val1 = self.encoder_backbone.conv1.weight.data.clone()
            self.load_pretrained_elements()
            assert not torch.all(torch.eq(self.encoder_backbone.conv1.weight.data.clone(),
                                          old_test_val1)).item()  # test if weights really loaded!

        # Initialize Lightweight ALIKED model to perform OTF GT generation for descriptors if training
        if conf.train_descriptors.do:
            logger.info("Load ALiked Lightweight model for descriptor training...")
            device = conf.train_descriptors.device if conf.train_descriptors.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
            self.aliked_lw = get_model("jpldd.aliked_light")(aliked_model_cfg).eval().to(device) # use same config than for our network parts

    # Utility methods for line df and af with deepLSD
    def normalize_df(self, df):
        return -torch.log(df / self.conf.line_neighborhood + 1e-6)

    def denormalize_df(self, df_norm):
        return torch.exp(-df_norm) * self.conf.line_neighborhood

    def _forward(self, data):
        """
        Perform a forward pass. Certain things are only executed NOT in training mode.
        """
        # output container definition
        output = {}

        # load image and padder
        image = data["image"]
        div_by = 2 ** 5
        padder = InputPadder(image.shape[-2], image.shape[-1], div_by)

        # Get Hidden Feature Map and Keypoint/junction scoring
        padded_img = padder.pad(image)
        feature_map_padded = self.encoder_backbone(padded_img)
        score_map_padded = self.keypoint_and_junction_branch(feature_map_padded)
        feature_map_padded_normalized = torch.nn.functional.normalize(feature_map_padded, p=2, dim=1)
        feature_map = padder.unpad(feature_map_padded_normalized)
        logger.debug(
            f"Image size: {image.shape}\nFeatureMap-unpadded: {feature_map.shape}\nFeatureMap-padded: {feature_map_padded.shape}")
        assert (feature_map.shape[2], feature_map.shape[3]) == (image.shape[2], image.shape[3])
        keypoint_and_junction_score_map = padder.unpad(score_map_padded)
        output["keypoint_and_junction_score_map"] = keypoint_and_junction_score_map  # B x 1 x H x W

        # Line Elements
        line_angle_field = self.angle_field_branch(feature_map)
        line_distance_field = self.distance_field_branch(feature_map)
        output["deeplsd_line_anglefield"] = line_angle_field
        output["deeplsd_line_distancefield"] = line_distance_field

        keypoints, kptscores, scoredispersitys = self.dkd(
            keypoint_and_junction_score_map, #image_size=data.get("image_size")
        )
        _, _, h, w = image.shape
        wh = torch.tensor([w, h], device=image.device)
        # no padding required,
        # we can set detection_threshold=-1 and conf.max_num_keypoints
        # todo: figure out whether there are issues with the list representation -> cannot expect same num of keypoints
        output["keypoints"] = renormalize_keypoints(keypoints, wh)  # B N 2 (list of B tensors having N by 2)
        output["keypoint_scores"] = kptscores  #torch.stack(kptscores),  # B N
        output["keypoint_score_dispersity"] = scoredispersitys  #torch.stack(scoredispersitys),

        # Keypoint descriptors
        # todo: figure out whether there are issues with the list representation -> cannot expect same num of keypoints
        keypoint_descriptors, offsets = self.descriptor_branch(feature_map, keypoints)
        output["keypoint_descriptors"] = keypoint_descriptors # torch.stack(keypoint_descriptors)  # B N D

        # Extract Lines from Learned Part of the Network
        # Only Perform line detection when NOT in training mode
        if not self.training:
            line_segments = None  # as endpoints
            output["line_segments"] = line_segments
            # Use aliked points sampled from inbetween Line endpoints?
            line_descriptors = None
            output["line_descriptors"] = line_descriptors

        return output

    def loss(self, pred, data):
        """
        perform loss calculation based on prediction and data(=groundtruth)
        1. On Keypoint-ScoreMap:        L1/L2 Loss / FC-Softmax?
        2. On Keypoint-Descriptors:     L1/L2 loss
        3. On Line-Angle Field:         L1/L2 Loss / FC-Softmax?
        4. On Line-Distance Field:      L1/L2 Loss / FC-Softmax?
        """
        keypoint_scoremap_loss = F.l1_loss(pred["keypoint_and_junction_score_map"],
                                           data["superpoint_heatmap"], reduction='mean')
        # Descriptor Loss: expect aliked descriptors as GT
        keypoint_descriptor_loss = F.l1_loss(pred["keypoint_descriptors"], data["aliked_descriptors"], reduction='mean')
        line_af_loss = F.l1_loss(pred["deeplsd_line_anglefield"], data["deeplsd_angle_field"], reduction='mean')
        line_df_loss = F.l1_loss(pred["deeplsd_line_distancefield"], data["deeplsd_distance_field"], reduction='mean')
        overall_loss = keypoint_scoremap_loss + keypoint_descriptor_loss + line_af_loss + line_df_loss
        return overall_loss

    def get_groundtruth_descriptors(self, pred: dict):
        """
        Takes keypoints from predictions (best 100 + 100 random) + computes groundtruth descriptors for it.
        """
        assert pred.get('image', None) is not None and pred.get('keypoints', None) is not None # todo: check dims
        with torch.no_grad():
            descriptors = self.aliked_lw(pred)
        return descriptors

    def load_pretrained_elements(self):
        """
        Loads ALIKED weights for backbone encoder, score_head(SMH) and SDDH
        """
        # Load state-dict of wanted aliked-model
        aliked_state_url = aliked_checkpoint_url.format(self.conf.model_name)
        aliked_state_dict = torch.hub.load_state_dict_from_url(aliked_state_url, map_location="cpu")
        # change keys
        for k, v in list(aliked_state_dict.items()):
            if k.startswith("block") or k.startswith("conv"):
                change_dict_key(aliked_state_dict, k, f"encoder_backbone.{k}")
            elif k.startswith("score_head"):
                change_dict_key(aliked_state_dict, k, f"keypoint_and_junction_branch.{k}")
            elif k.startswith("desc_head"):
                change_dict_key(aliked_state_dict, k, f"descriptor_branch.{k[10:]}")
            else:
                continue
        # load values
        self.load_state_dict(aliked_state_dict, strict=False)

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
