import torch
from pytorchvideo.accelerator.deployment.mobile_cpu.utils.model_conversion import (
    convert_to_deployable_form,
)
from pytorchvideo.models.accelerator.mobile_cpu.efficient_x3d import EfficientX3d
from torch.hub import load_state_dict_from_url
from torch.utils.mobile_optimizer import (
    optimize_for_mobile,
)
from torch.utils import bundled_inputs
from torch.utils.bundled_inputs import augment_model_with_bundled_inputs

model_efficient_x3d_xs = EfficientX3d(expansion='XS', head_act='identity')

checkpoint_path = 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/efficient_x3d_xs_original_form.pyth'
checkpoint = load_state_dict_from_url(checkpoint_path)

model_efficient_x3d_xs.load_state_dict(checkpoint)
input_blob_size = (1, 3, 4, 160, 160)
input_tensor = torch.randn(input_blob_size)
model_efficient_x3d_xs_deploy = convert_to_deployable_form(model_efficient_x3d_xs, input_tensor)
traced_model = torch.jit.trace(model_efficient_x3d_xs_deploy, input_tensor, strict=False)
optimized_traced__model = optimize_for_mobile(traced_model)


inflatable_arg_predict_net = bundled_inputs.bundle_randn(1, 3, 4, 160, 160 )
inputs_predict_net = [
    (inflatable_arg_predict_net,),
    (inflatable_arg_predict_net,),
]

augment_model_with_bundled_inputs( optimized_traced__model , inputs_predict_net)
optimized_traced__model._save_for_lite_interpreter("../PerfBenchMarkModels/video_classification_1.ptl")


