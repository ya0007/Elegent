# 圆心坐标探测

import cv2
import numpy as np


#smarties = cv2.imread(r"C:\\Users\\TQ\\img\\20230426_152102.bmp")
smarties = cv2.imread(r"C:\\Users\\TQ\\img\\20230426_152102.bmp")

ret, smarties_inv = cv2.threshold(smarties, 74, 255, cv2.THRESH_BINARY_INV)
img_gray = cv2.cvtColor(smarties_inv, cv2.COLOR_BGR2GRAY)

# # 进行中值滤波
# img = cv2.medianBlur(img_gray, 5)
#
# circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 40, param1=10, param2=15, minRadius=30, maxRadius=110)
#
# # cv2.namedWindow("Circle2",0);
# # cv2.resizeWindow("Circle2", 800, 600);
# # cv2.imshow("Circle2",img_gray)
#
#
# # 对数据进行四舍五入变为整数
# # circles = pd.DataFrame(circles, dtype=float)
# circles = np.uint16(np.around(circles))
#
# # font for the text being specified
# font1 = cv2.FONT_HERSHEY_SIMPLEX
#
# # font scale for the text being specified
# fontScale1 = 1
# # Blue color for the text being specified from BGR
# color1 = (255, 255, 255)
# # Line thickness for the
# thickness1 = 2
#
# for i in circles[0, :]:
#     # 画出来圆的边界
#     cv2.circle(smarties, (i[0], i[1]), i[2], (0, 0, 255), 2)
#     # 画出来圆心
#     cv2.circle(smarties, (i[0], i[1]), 2, (0, 0, 255), 3)
#     # org for the text being specified
#     # 获取圆心坐标
#     b = circles[:][0]
#     org1 = b[:, 0:2]
#     # 读出来圆心坐标
#     # print(i[0],i[1],i[2])
#     # cv2.putText(smarties, "position: "+str(i[0]), org1, font1, fontScale1, color1, thickness1, cv2.LINE_AA)
#     # cv2.putText(smarties,"position: "+str(i[0])+", "+str(i[1]),org1,cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,255),2)
#
# # 获取圆心坐标
# b = circles[:][0]
# c = b[:, 0:2]
# c = c / 6
# a = c.tolist()
#
# print(a)

cv2.namedWindow("Circle", 0);
cv2.resizeWindow("Circle", 800, 600);
cv2.imshow("Circle", img_gray)
cv2.waitKey()
cv2.destroyAllWindows()
#!/user/bin/env python3
# -*- coding: utf-8 -*-
