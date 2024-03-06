#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rospy
import cv2
import time
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
#from learn_service.srv import img, imgRequest
from sineva_vision.srv import GetImage, GetImageRequest
from pyzbar.pyzbar import decode
#import zbar


if __name__ == "__main__":

    # 2.初始化 ROS 节点
    rospy.init_node('client_img_com111') 

    # 3.创建请求对象
    client = rospy.ServiceProxy("/hikrobot/getImage",GetImage)

    # 请求前，等待服务已经就绪
    rospy.wait_for_service("/hikrobot/getImage")

    # 4.发送请求,接收并处理响应
    # resp = client(AddIntsRequest(1,5))
    getCassetSrv = GetImageRequest(1)


    # resp = client.call(req)


    # bridge = CvBridge()
    # cv_image = bridge.imgmsg_to_cv2(resp.imgs, "mono8")
    # cv2.imshow("view", cv_image)
    # cv2.waitKey(0)

#client_srv_get_image  client
#getCassetSrv req

    for i in range(3):
        resp = client(getCassetSrv)
        # print(resp)

        if resp.state == 1:
            break
        print("Reconnect camera……")
        time.sleep(1)

    if resp.state == 0:
        camera_state = False
        # print("11111111111")
        state = 1005
     
    # print("22222222222")
    if resp.state == 1:
        camera_state = True
        for image_msg in resp.imgs:
            print("Get casst_code image ......")
            cv_bridge = CvBridge()
            try:
                cv_ptr = cv_bridge.imgmsg_to_cv2(image_msg, "mono8")
            except CvBridgeError as e:
                print("CvBridge Error: ", e)
            
           
            # cv2.imshow("view", cv_ptr)
            # cv2.waitKey(0)


            # cv2.namedWindow("enhanced",0)

            # cv2.resizeWindow("enhanced", 640, 480)
            cv2.imwrite('/opt/SinevaAGV/SinevaCodeAMR/src/vision/get_image_hk/src/img/saved_222.png', cv_ptr)

            # cv2.imshow("enhanced",cv_ptr)         

            # cv2.waitKey(0)


            barcodes = decode(cv_ptr)
             
            # print("1111111111")
            for barcode in barcodes:
                # print("3333")
                print(barcode.data.decode('utf-8'))


 

            imgCasst = cv_ptr
            if imgCasst.size == 0:
                print("the image is empty")
             

    rospy.loginfo("响应结果")








