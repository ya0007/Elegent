import numpy as np
import cv2
import onnxruntime as ort

class YOLOv5cls:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = [output.name for output in self.session.get_outputs()]
        self.inpHeight = self.session.get_inputs()[0].shape[2]
        self.inpWidth = self.session.get_inputs()[0].shape[3]

    def resize_image(self, srcimg):
        newh, neww = self.inpHeight, self.inpWidth
        dstimg = cv2.resize(srcimg, (neww, newh))
        return dstimg

    def normalize_(self, img, imgmean=[0.485, 0.456, 0.406], imgstd=[0.229, 0.224, 0.225]):
        img = img.astype(np.float32) / 255.0
        img = (img - imgmean) / imgstd
        return img

    def detect(self, frame):
        if frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        dstimg = self.resize_image(frame)
        dstimg = self.normalize_(dstimg)
        dstimg = np.transpose(dstimg, (2, 0, 1))
        dstimg = np.expand_dims(dstimg, axis=0)

        outputs = self.session.run(self.output_name, {self.input_name: dstimg})
        output = outputs[0]
        # 这里的处理取决于你的模型输出
        result = np.argmax(output)  # 假设输出是一个概率数组
        return result

# 示例使用
if __name__ == "__main__":
    model_path = "path_to_your_model.onnx"
    yolo_model = YOLOv5cls(model_path)
    img_path = "path_to_your_image.jpg"
    srcimg = cv2.imread(img_path)
    result = yolo_model.detect(srcimg)
    print("Detection result:", result)

    