#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#调试成功
import sys
import rospy
import cv2
import time
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
#from learn_service.srv import img, imgRequest
from sineva_vision.srv import ScanID, ScanIDRequest, ScanIDResponse,  GetImage, GetImageRequest
from pyzbar.pyzbar import decode
#import zbar


class VisualPose:
    def __init__(self):
        self.bridge = CvBridge()
        # 假设已经有了一个ROS服务客户端来获取图像
        self.client_srv_get_image = rospy.ServiceProxy("/hikrobot/getImage", GetImage)
        # print("44444444444")

    def read_casset_code(self, req):
        # print("55555555555555555")
        if req.commandId == 999:
            if req.projectNum == 1:
                count_no_code = 0
                no_code = True
                while no_code:
                    try:
                        get_casset_srv = self.client_srv_get_image(1)  # 假设服务请求需要图像数量
                        if get_casset_srv.state == 0:
                            return ScanIDResponse(state=1005)
                        elif get_casset_srv.state == 1:
                            for image_msg in get_casset_srv.imgs:
                                try:
                                    cv_image = self.bridge.imgmsg_to_cv2(image_msg, "mono8")
                                except CvBridgeError as e:
                                    rospy.logwarn("CvBridge Error: {0}".format(e))
                                    return ScanIDResponse(state=1005)
                                
                                # 使用pyzbar解码
                                barcodes = decode(cv_image)
                                if not barcodes:
                                    count_no_code += 1
                                    if count_no_code >= 5:
                                        return ScanIDResponse(state=1001)
                                    continue
                                
                                for barcode in barcodes:
                                    rospy.loginfo("Code type: {}".format(barcode.type))
                                    rospy.loginfo("Decode string: {}".format(barcode.data.decode("utf-8")))
                                    # 假设ScanIDResponse有一个scanID列表来存储解码的数据
                                    res = ScanIDResponse(state=1100, scanID=[ord(c) for c in barcode.data.decode("utf-8")])
                                    return res
                    except rospy.ServiceException as exc:
                        rospy.loginfo("Service did not process request: " + str(exc))
                        return ScanIDResponse(state=1005)
        return ScanIDResponse(state=1001)  # 默认返回状态

if __name__ == "__main__":
    rospy.init_node('visual_pose_node')
    # print("111")
    visual_pose = VisualPose()
    # print("222")
    s = rospy.Service("/visual/scan_id", ScanID, visual_pose.read_casset_code)
    # print("33333")
    rospy.spin()




    









