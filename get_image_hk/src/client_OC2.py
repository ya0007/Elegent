import torch
from torchvision import models, transforms
from PIL import Image

def classify_cat_image(image_path):
    # 加载预训练的ResNet模型
    model = models.resnet18(pretrained=True)
    model.eval()  # 设置为评估模式

    # 定义图像的预处理
    preprocess = transforms.Compose([
        transforms.Resize(256),  # 将图像大小调整为256x256
        transforms.CenterCrop(224),  # 从图像中心裁剪224x224大小
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])

    # 加载图像并进行预处理
    img = Image.open(image_path)
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)  # 添加一个批次维度

    # 使用模型进行预测
    with torch.no_grad():
        output = model(batch_t)

    # 获取预测结果
    _, predicted = torch.max(output, 1)

    # 加载ImageNet类别标签
    with open("imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    # 输出预测结果
    predicted_class = classes[predicted[0]]
    print(f"Predicted class: {predicted_class}")

# 使用示例
# 确保你有一张名为"cat.jpg"的猫的图片，以及一个包含ImageNet类别标签的"imagenet_classes.txt"文件
classify_cat_image("cat.jpg")