#!/usr/bin/env python3

import contextlib
import copy
import os
import unittest
from PIL import Image

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.utils import bundled_inputs
from torch.utils.bundled_inputs import augment_model_with_bundled_inputs

from d2go.export.api import convert_and_export_predictor
from d2go.export.d2_meta_arch import patch_d2_meta_arch
from d2go.runner import create_runner, GeneralizedRCNNRunner
from d2go.model_zoo import model_zoo

from mobile_cv.common.misc.file_utils import make_temp_directory

patch_d2_meta_arch()


def test_export_torchvision_format():
    cfg_name = 'faster_rcnn_fbnetv3a_dsmask_C4.yaml'
    pytorch_model = model_zoo.get(cfg_name, trained=True)

    from typing import List, Dict
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            coco_idx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                             27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
                             52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
                             78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91]

            self.coco_idx = torch.tensor(coco_idx_list)

        def forward(self, inputs: List[torch.Tensor]):
            x = inputs[0].unsqueeze(0) * 255
            scale = 320.0 / min(x.shape[-2], x.shape[-1])
            x = torch.nn.functional.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=True, recompute_scale_factor=True)
            out = self.model(x[0])
            res : Dict[str, torch.Tensor] = {}
            res["boxes"] = out[0] / scale
            res["labels"] = torch.index_select(self.coco_idx, 0, out[1])
            res["scores"] = out[2]
            return inputs, [res]

    size_divisibility = max(pytorch_model.backbone.size_divisibility, 10)
    h, w = size_divisibility, size_divisibility * 2

    runner = create_runner("d2go.runner.GeneralizedRCNNRunner")
    cfg = model_zoo.get_config(cfg_name)
    datasets = list(cfg.DATASETS.TRAIN)

    data_loader = runner.build_detection_test_loader(cfg, datasets)

    predictor_path = convert_and_export_predictor(
        cfg,
        copy.deepcopy(pytorch_model),
        "torchscript_int8@tracing",
        './',
        data_loader,
    )

    orig_model = torch.jit.load(os.path.join(predictor_path, "model.jit"))
    scripted_model = torch.jit.script(orig_model)
    optimized_model = optimize_for_mobile(scripted_model)

    inflatable_arg_predict_net = bundled_inputs.bundle_randn(3,600,600)
    inputs_predict_net = [
        (inflatable_arg_predict_net,),
        (inflatable_arg_predict_net,),
    ]

    augment_model_with_bundled_inputs(optimized_model, inputs_predict_net)
    optimized_model._save_for_lite_interpreter("../PerfBenchMarkModels/object_detection_1.ptl")


if __name__ == '__main__':
    test_export_torchvision_format()
