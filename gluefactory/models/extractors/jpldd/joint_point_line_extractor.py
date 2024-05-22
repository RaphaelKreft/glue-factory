import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.models.base_model import BaseModel
from gluefactory.models.extractors.jpldd.backbone_encoder import AlikedEncoder, aliked_cfgs
from gluefactory.models.extractors.jpldd.descriptor_head import SDDH
from gluefactory.models.extractors.jpldd.keypoint_decoder import SMH
from gluefactory.models.extractors.jpldd.keypoint_detection import SimpleDetector, DKD
from gluefactory.models.extractors.jpldd.utils import InputPadder, change_dict_key
from gluefactory.models.extractors.jpldd.metrics import compute_pr, compute_loc_error, compute_repeatability

to_ctr = OmegaConf.to_container  # convert DictConfig to dict
aliked_checkpoint_url = "https://github.com/Shiaoming/ALIKED/raw/main/models/{}.pth"
logger = logging.getLogger(__file__)


class JointPointLineDetectorDescriptor(BaseModel):
    # currently contains only ALIKED
    default_conf = {
        "aliked_model_name": "aliked-n16",
        "max_num_keypoints": 1000,  # setting for training, for eval: -1
        "detection_threshold": -1,  # setting for training, for eval: 0.2
        "force_num_keypoints": False,
        "pretrained": True,
        "pretrain_kp_decoder": True,
        "nms_radius": 2,
        "line_neighborhood": 5,
        "timeit": True,  # override timeit: False from BaseModel
        "train_descriptors": {
            "do": True,  # if train is True, initialize ALIKED Light model form OTF Descriptor GT
            "gt_aliked_model": "aliked-n32"
        },
        "lambda_weighted_bce": 200,
        "loss_weights": {
            "line_af_weight": 10,
            "line_df_weight": 10,
            "keypoint_weight": 1,
            "descriptor_weight": 1
        }
    }

    n_limit_max = 20000  # taken from ALIKED which gives max num keypoints to detect!

    required_data_keys = ["image"]

    def _init(self, conf):
        logger.debug(f"final config dict(type={type(conf)}): {conf}")
        # c1-c4 -> output dimensions of encoder blocks, dim -> dimension of hidden feature map
        # K=Kernel-Size, M=num sampling pos
        aliked_model_cfg = aliked_cfgs[conf.aliked_model_name]
        dim = aliked_model_cfg["dim"]
        K = aliked_model_cfg["K"]
        M = aliked_model_cfg["M"]
        self.lambda_valid_kp = conf.lambda_weighted_bce
        # Load Network Components
        self.encoder_backbone = AlikedEncoder(aliked_model_cfg)
        self.keypoint_and_junction_branch = SMH(dim)  # using SMH from ALIKE here
        #self.detector = SimpleDetector(nms_radius=conf.nms_radius,
        #                               num_keypoints=-1 if conf.detection_threshold > 0 else conf.max_num_keypoints,
        #                               threshold=conf.detection_threshold)
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

        if conf.timeit:
            self.timings = {
                "total-makespan": [],
                "encoder": [],
                "keypoint-and-junction-heatmap": [],
                "line-af": [],
                "line-df": [],
                "descriptor-branch": [],
                "keypoint-detection": []
            }

        # load pretrained_elements if wanted (for now that only the ALIKED parts of the network)
        if conf.pretrained:
            logger.warning("Load pretrained weights for aliked parts...")
            old_test_val1 = self.encoder_backbone.conv1.weight.data.clone()
            self.load_pretrained_elements()
            assert not torch.all(torch.eq(self.encoder_backbone.conv1.weight.data.clone(),
                                          old_test_val1)).item()  # test if weights really loaded!

        # Initialize Lightweight ALIKED model to perform OTF GT generation for descriptors if training
        if conf.train_descriptors.do:
            logger.warning("Load ALiked Lightweight model for descriptor training...")
            aliked_gt_cfg = {
                "model_name": self.conf.train_descriptors.gt_aliked_model,
                "max_num_keypoints": self.conf.max_num_keypoints,
                "detection_threshold": self.conf.detection_threshold,
                "force_num_keypoints": False,
                "pretrained": True,
                "nms_radius": self.conf.nms_radius,
            }
            self.aliked_lw = get_model("jpldd.aliked_light")(
                aliked_gt_cfg).eval()

    # Utility methods for line df and af with deepLSD
    def normalize_df(self, df):
        return -torch.log(df / self.conf.line_neighborhood + 1e-6)

    def denormalize_df(self, df_norm):
        return torch.exp(-df_norm) * self.conf.line_neighborhood

    def _forward(self, data):
        """
        Perform a forward pass. Certain things are only executed NOT in training mode.
        """
        if self.conf.timeit:
            total_start = time.time()
        # output container definition
        output = {}

        # load image and padder
        image = data["image"]
        div_by = 2 ** 5
        padder = InputPadder(image.shape[-2], image.shape[-1], div_by)

        # Get Hidden Feature Map and Keypoint/junction scoring
        padded_img = padder.pad(image)

        # pass through encoder
        if self.conf.timeit:
            start_encoder = time.time()
            feature_map_padded = self.encoder_backbone(padded_img)
            self.timings["encoder"].append(time.time() - start_encoder)
        else:
            feature_map_padded = self.encoder_backbone(padded_img)

        # pass through keypoint & junction decoder
        if self.conf.timeit:
            start_keypoints = time.time()
            score_map_padded = self.keypoint_and_junction_branch(feature_map_padded)
            self.timings["keypoint-and-junction-heatmap"].append(time.time() - start_keypoints)
        else:
            score_map_padded = self.keypoint_and_junction_branch(feature_map_padded)

        # normalize and remove padding and format dimensions
        feature_map_padded_normalized = torch.nn.functional.normalize(feature_map_padded, p=2, dim=1)
        feature_map = padder.unpad(feature_map_padded_normalized)
        logger.debug(
            f"Image size: {image.shape}\nFeatureMap-unpadded: {feature_map.shape}\nFeatureMap-padded: {feature_map_padded.shape}")
        assert (feature_map.shape[2], feature_map.shape[3]) == (image.shape[2], image.shape[3])
        keypoint_and_junction_score_map = padder.unpad(score_map_padded)  # B x 1 x H x W

        # For storing, remove additional dimension but keep batch dimension even if its 1
        # but keep additional dimension for variable -> needed by dkd
        if keypoint_and_junction_score_map.shape[0] == 1:
            output["keypoint_and_junction_score_map"] = keypoint_and_junction_score_map[:, 0, :, :]  # B x H x W
        else:
            output["keypoint_and_junction_score_map"] = keypoint_and_junction_score_map.squeeze()  # B x H x W

        # Line AF Decoder
        if self.conf.timeit:
            start_line_af = time.time()
            line_angle_field = self.angle_field_branch(
                feature_map) * torch.pi  # multipy with pi as output is in [0, 1] and we want to get angle
            self.timings["line-af"].append(time.time() - start_line_af)
        else:
            line_angle_field = self.angle_field_branch(feature_map) * torch.pi

        # Line DF Decoder
        if self.conf.timeit:
            start_line_df = time.time()
            line_distance_field = self.denormalize_df(self.distance_field_branch(
                feature_map))  # denormalize as NN outputs normalized version which is focused on line neighborhood
            self.timings["line-df"].append(time.time() - start_line_df)
        else:
            line_distance_field = self.denormalize_df(self.distance_field_branch(feature_map))

        # remove additional dimensions of size 1 if not having batchsize one
        assert line_angle_field.shape == line_distance_field.shape
        if line_angle_field.shape[0] == 1:
            line_angle_field = line_angle_field[:, 0, :, :]
            line_distance_field = line_distance_field[:, 0, :, :]
        else:
            line_angle_field = line_angle_field.squeeze()  # squeeze to remove size 1 dim to match groundtruth
            line_distance_field = line_distance_field.squeeze()

        output["line_anglefield"] = line_angle_field
        output["line_distancefield"] = line_distance_field

        # Keypoint detection
        if self.conf.timeit:
            start_keypoints = time.time()
            keypoints, kptscores, _ = self.dkd(
                keypoint_and_junction_score_map,
            )
            self.timings["keypoint-detection"].append(time.time() - start_keypoints)
        else:
            keypoints, kptscores, _ = self.dkd(
                keypoint_and_junction_score_map,
            )

        # raw output of DKD needed to generate GT-Descriptors
        if self.conf.train_descriptors.do:
            output["keypoints_raw"] = keypoints

        _, _, h, w = image.shape
        wh = torch.tensor([w, h], device=image.device)
        # no padding required,
        # can set detection_threshold=-1 and conf.max_num_keypoints -> HERE WE SET THESE VALUES SO WE CAN EXPECT SAME NUM!
        output["keypoints"] = wh * (torch.stack(keypoints) + 1.) / 2.0
        output["keypoint_scores"] = torch.stack(kptscores)

        # Keypoint descriptors
        if self.conf.timeit:
            start_desc = time.time()
            keypoint_descriptors, _ = self.descriptor_branch(feature_map, keypoints)
            self.timings["descriptor-branch"].append(time.time() - start_desc)
        else:
            keypoint_descriptors, _ = self.descriptor_branch(feature_map, keypoints)
        output["keypoint_descriptors"] = torch.stack(keypoint_descriptors)  # B N D

        # Extract Lines from Learned Part of the Network
        # Only Perform line detection when NOT in training mode
        if not self.training:
            line_segments = None  # as endpoints
            output["line_segments"] = line_segments
            # Use aliked points sampled from inbetween Line endpoints?
            line_descriptors = None
            output["line_descriptors"] = line_descriptors

        if self.conf.timeit:
            self.timings["total-makespan"].append(time.time() - total_start)
        return output

    def loss(self, pred, data):
        """
        format of data: B x H x W
        perform loss calculation based on prediction and data(=groundtruth) for a batch
        1. On Keypoint-ScoreMap:        weighted BCE Loss
        2. On Keypoint-Descriptors:     L1 loss
        3. On Line-Angle Field:         use angle loss from deepLSD paper
        4. On Line-Distance Field:      use L1 loss on normalized versions of Distance field (as in deepLSD paper)
        """

        def weighted_bce_loss(pred, target):
            return -self.lambda_valid_kp * target * torch.log(pred) - (1 - target) * torch.log(1 - pred)

        losses = {}
        metrics = {}

        assert (0 <= pred["keypoint_and_junction_score_map"].min() and pred[
            "keypoint_and_junction_score_map"].max() <= 1)
        assert (0 <= data["superpoint_heatmap"].min() and data["superpoint_heatmap"].max() <= 1)
        # Use Weighted BCE Loss for Point Heatmap
        keypoint_scoremap_loss = weighted_bce_loss(pred["keypoint_and_junction_score_map"],
                                                   data["superpoint_heatmap"]).mean(dim=(1, 2))

        losses["keypoint_and_junction_score_map"] = keypoint_scoremap_loss
        # Descriptor Loss: expect aliked descriptors as GT
        if self.conf.train_descriptors.do:
            data = {**data,
                    **self.get_groundtruth_descriptors({"keypoints": pred["keypoints_raw"], "image": data["image"]})}
            keypoint_descriptor_loss = F.l1_loss(pred["keypoint_descriptors"], data["aliked_descriptors"],
                                                 reduction='none').mean(dim=(1, 2))
            losses["keypoint_descriptors"] = keypoint_descriptor_loss

        # use angular loss for distance field
        af_diff = (data["deeplsd_angle_field"] - pred["line_anglefield"])
        line_af_loss = torch.minimum(af_diff ** 2, (torch.pi - af_diff.abs()) ** 2).mean(
            dim=(1, 2))  # pixelwise minimum
        losses["line_anglefield"] = line_af_loss

        # use normalized versions for loss
        gt_mask = data["deeplsd_distance_field"] < self.conf.line_neighborhood
        line_df_loss = F.l1_loss(self.normalize_df(pred["line_distancefield"]) * gt_mask,
                                 self.normalize_df(data["deeplsd_distance_field"]) * gt_mask,
                                 # only supervise in line neighborhood
                                 reduction='none').mean(dim=(1, 2))
        losses["line_distancefield"] = line_df_loss

        # Compute overall loss
        overall_loss = (self.conf.loss_weights.keypoint_weight * keypoint_scoremap_loss
                        + self.conf.loss_weights.line_af_weight * line_af_loss
                        + self.conf.loss_weights.line_df_weight * line_df_loss)
        if self.conf.train_descriptors.do:
            overall_loss += self.conf.loss_weights.descriptor_weight * keypoint_descriptor_loss
        losses["total"] = overall_loss

        # add metrics if not in training mode
        if not self.training:
            metrics = self.metrics(pred, data)
        return losses, metrics

    def get_groundtruth_descriptors(self, pred: dict):
        """
        Takes keypoints from predictions (best 100 + 100 random) + computes ground-truth descriptors for it.
        """
        assert pred.get('image', None) is not None and pred.get('keypoints', None) is not None
        with torch.no_grad():
            descriptors = self.aliked_lw(pred)
        return descriptors

    def load_pretrained_elements(self):
        """
        Loads ALIKED weights for backbone encoder, score_head(SMH) and SDDH
        """
        # Load state-dict of wanted aliked-model
        aliked_state_url = aliked_checkpoint_url.format(self.conf.aliked_model_name)
        aliked_state_dict = torch.hub.load_state_dict_from_url(aliked_state_url, map_location="cpu")
        # change keys
        for k, v in list(aliked_state_dict.items()):
            if k.startswith("block") or k.startswith("conv"):
                change_dict_key(aliked_state_dict, k, f"encoder_backbone.{k}")
            elif k.startswith("score_head"):
                if not self.conf.pretrain_kp_decoder:
                    del aliked_state_dict[k]
                else:
                    change_dict_key(aliked_state_dict, k, f"keypoint_and_junction_branch.{k}")
            elif k.startswith("desc_head"):
                change_dict_key(aliked_state_dict, k, f"descriptor_branch.{k[10:]}")
            else:
                continue

        # load values
        self.load_state_dict(aliked_state_dict, strict=False)

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def check_loss_keys_in_dict(self, data_keys):
        for required_loss_key in self.conf.required_loss_keys:
            if required_loss_key not in data_keys:
                return False
        return True

    def state_dict(self, *args, **kwargs):
        """
        Custom state dict to exclude aliked_lw module from checkpoint.
        """
        sd = super().state_dict(*args, **kwargs)
        # don't store lightweight aliked model for descriptor gt computation
        if self.conf.train_descriptors.do:
            for k in list(sd.keys()):
                if k.startswith("aliked_lw"):
                    del sd[k]
        return sd

    def get_current_timings(self, reset=False):
        """
        ONLY USE IF TIMEIT ACTIVATED. It returns the average of the current times in a dictionary for
        all the single network parts.

        reset: if True deletes all collected times until now
        """
        results = {}
        for k, v in self.timings.items():
            results[k] = np.mean(v)
            if reset:
                self.timings[k] = []
        return results

    def get_pr(self, pred_kp: torch.Tensor, gt_kp: torch.Tensor, tol=3):
        """ Compute the precision and recall, based on GT KP. """
        if len(gt_kp) == 0:
            precision = float(len(pred_kp) == 0)
            recall = 1.
        elif len(pred_kp) == 0:
            precision = 1.
            recall = float(len(gt_kp) == 0)
        else:
            dist = torch.norm(pred_kp[:, None] - gt_kp[None], dim=2)
            close = (dist < tol).float()
            precision = close.max(dim=1)[0].mean()
            recall = close.max(dim=0)[0].mean()
        return precision, recall

    def metrics(self, pred, data):
        device = pred['keypoint_and_junction_score_map'].device
        gt = data["superpoint_heatmap"].cpu().numpy()
        predictions = pred["keypoint_and_junction_score_map"].cpu().numpy()
        imgs = data["image"].cpu().numpy()
        # Compute the precision and recall
        precision, recall, _ = compute_pr(gt, predictions)
        loc_error = compute_loc_error(gt, predictions)
        rep = compute_repeatability(gt, predictions, imgs, self, device)

        out = {
            'precision': torch.tensor(precision.copy(), dtype=torch.float, device=device),
            'recall': torch.tensor(recall.copy(), dtype=torch.float, device=device),
            'repeatability': torch.tensor([rep], dtype=torch.float, device=device),
            'loc_error': torch.tensor([loc_error], dtype=torch.float, device=device)
        }
        return out
