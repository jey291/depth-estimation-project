

# openvino ir转换脚本
import sys
from pathlib import Path

import torch
import openvino as ov

root = Path(
    r"C:\Users\11650\PycharmProjects\PythonProject1\Depth-Anything-V2"
)

sys.path.insert(0, str(root))
print(sys.path[0])
from depth_anything_v2.dpt import DepthAnythingV2
print("import success")
model_configs = {
    'vits': {
        'encoder': 'vits',
        'features': 64,
        'out_channels': [48, 96, 192, 384]
    }
}

model = DepthAnythingV2(**model_configs['vits'])

ckpt = root / "checkpoints" / "depth_anything_v2_vits.pth"

model.load_state_dict(
    torch.load(ckpt, map_location="cpu")
)

model.eval()

example_input = torch.randn(1, 3, 518, 518)

ov_model = ov.convert_model(
    model,
    example_input=example_input
)

ov.save_model(
    ov_model,
    "depth_anything_v2_vits.xml"
)

print("转换成功")