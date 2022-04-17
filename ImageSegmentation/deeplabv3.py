import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.utils import bundled_inputs
from torch.utils.bundled_inputs import augment_model_with_bundled_inputs

model = torch.hub.load('pytorch/vision:v0.11.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()

scripted_module = torch.jit.script(model)
optimized_model = optimize_for_mobile(scripted_module)

inflatable_arg_predict_net = bundled_inputs.bundle_randn(1,3,224,224)
inputs_predict_net = [
    (inflatable_arg_predict_net,),
    (inflatable_arg_predict_net,),
]

augment_model_with_bundled_inputs( optimized_model , inputs_predict_net)
optimized_model.save("ImageSegmentation/deeplabv3_scripted.pt")
optimized_model._save_for_lite_interpreter("../PerfBenchMarkModels/image_segmentation_1.ptl")
