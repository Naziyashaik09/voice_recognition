import argparse
import functools

import numpy as np
import torch

from utils.reader import load_audio
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('threshold',        float,   0.71,                    'Threshold for judging whether it is the same person')
add_arg('input_shape',      str,    '(1, 257, 257)',          'shape of data entry')
add_arg('model_path',       str,    'models_large/resnet34.pth',    'Predictive model path')
# args = parser.parse_args()
args =parser.parse_known_args()[0]

print_arguments(args)
print(torch.cuda.is_available())
device = torch.device("cpu")

# 加载模型
# model = torch.jit.load(args.model_path)
model = torch.jit.load(args.model_path,map_location="cpu")
# model.to(device)
model.eval()


# 预测音频
def infer(audio_path):
    input_shape = eval(args.input_shape)
    data = load_audio(audio_path, mode='infer', spec_len=input_shape[2])
    data = data[np.newaxis, :]
    data = torch.tensor(data, dtype=torch.float32)
    # 执行预测
    feature = model(data)
    return feature.data.cpu().numpy()


def run(audio1,audio2):
    # 要预测的两个人的音频文件
    feature1 = infer(audio1)[0]
    feature2 = infer(audio2)[0]
    # 对角余弦值
    dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    
    return dist