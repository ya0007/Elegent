import torch
import numpy as np
import os
import json
import time 
import onnx
import onnxruntime
from PIL import Image
import torchvision.transforms as transforms


# Input to the model
batch_size = 1
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

onnx_model = onnx.load("resnet50.onnx")
onnx.checker.check_model(onnx_model)

# read class_indict
json_path = './class_indices.json'
def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)
labels = load_labels(json_path)
with open(json_path, "r") as f:
    class_indict = json.load(f)


ort_session = onnxruntime.InferenceSession("resnet50.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

img = Image.open("2.jpg")

data_transform = transforms.Compose(
    [transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


img_y = data_transform(img)
img_y = torch.unsqueeze(img_y, dim=0)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}

start = time.time()
raw_result = ort_session.run(None, ort_inputs)
end = time.time()
inference_time = np.round((end - start) * 1000, 2)

print('========================================')
print('Inference time: ' + str(inference_time) + " ms")
print('========================================')


def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
def postprocess(result):
    return softmax(np.array(result)).tolist()
res = postprocess(raw_result)
idx = np.argmax(res)


print('========================================')
print('Final top prediction is: ' + labels[idx])





