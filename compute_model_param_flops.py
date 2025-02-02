import torch

from model_analysis.get_model_info import get_model_stat
from LMV.models.module import LMV_seg, LMV_cls

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_info_path = 'LMV_seg_para.txt'
    model = LMV_cls().to(device)

    stats = {}

    input_tensor = torch.randn(1, 3, 640, 640).to(device)
    flops, params = get_model_stat(model, input_tensor, output_path=model_info_path, device=str(device))

    stats["Parameters(M)"] = params / 1e6
    stats["FLOPs(G)"] = flops / 1e9

    for k, v in stats.items():
        print(f'{k}: {v}')

