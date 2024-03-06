## 给图片添加中文字体
import cv2
import numpy as np


imgBGR = cv2.imread("C://Users//TQ//img//KeJiGan3.jpg")  # 读取彩色图像(BGR)

from PIL import Image, ImageDraw, ImageFont

if (isinstance(imgBGR, np.ndarray)):  # 判断是否 OpenCV 图片类型
    imgPIL = Image.fromarray(cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB))


# text1 = "力矩器表面质量视觉检测系统"
# pos1 = (80, 180)  # (left, top)，字符串左上角坐标
# color = (255, 255, 255)  # 字体颜色
# textSize1 = 65
# drawPIL = ImageDraw.Draw(imgPIL)
# fontText1 = ImageFont.truetype("font/simsun.ttc", textSize1, encoding="utf-8")
# drawPIL.text(pos1, text1, color, font=fontText1)

text2 = "北京航天控制仪器研究所"
pos2 = (300, 680)  # (left, top)，字符串左上角坐标
color = (255, 255, 255)  # 字体颜色
textSize2 = 30
drawPIL = ImageDraw.Draw(imgPIL)
fontText2 = ImageFont.truetype("font/simsun.ttc", textSize2, encoding="utf-8")
drawPIL.text(pos2, text2, color, font=fontText2)

imgPutText = cv2.cvtColor(np.asarray(imgPIL), cv2.COLOR_RGB2BGR)

cv2.imshow("imgPutText", imgPutText)  # 显示叠加图像 imgAdd
key = cv2.waitKey(0)  # 等待按键命令

cv2.imwrite('./KeJiGan3g.jpg',imgPutText)
cv2.destroyAllWindows()


