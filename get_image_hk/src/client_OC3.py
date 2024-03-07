import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
# from some_yolo_package import YOLOv5cls  # 假设的YOLO模型Python包
import onnx2predic3NUC
from your_service_package.srv import JudgeOC, JudgeOCRequest, JudgeOCResponse  # 根据实际服务定义调整
import os
import time

class OcJudge:
    def __init__(self):
        self.bridge = CvBridge()
        # self.yolo_model = YOLOv5cls(os.path.join(roslib.packages.get_pkg_dir("visual_oc"), "model", "best.onnx"))

        
        # self.image_topic_name = rospy.get_param("/topic/image")
        self.client_srv_get_image = rospy.ServiceProxy("/hikrobot/getImage", GetImage)  # 根据实际服务类型调整
        self.server = rospy.Service("/visual/judge_oc", JudgeOC, self.trigger)

    def visual_compute(self, result):
        getImageSrv = GetImageRequest(num_image=1)  # 根据实际服务请求类型调整
        for i in range(3):
            if self.client_srv_get_image.call(getImageSrv) and getImageSrv.response.state == 1:
                break
            rospy.loginfo("Reconnect camera…")
            time.sleep(1)
        if getImageSrv.response.state != 1:
            return False

        for image_msg in getImageSrv.response.imgs:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "mono8")
            if cv_image is None:
                rospy.logwarn("The image is empty")
                return False
            # result = self.yolo_model.detect(cv_image)
            result = self.onnx2predic3NUC(cv_image)
            
            # 保存图像等操作
        return True

    def trigger(self, req, res):
        rospy.logwarn("Receive Trigger msg")
        projectNum = req.projectNum
        if projectNum == 6:
            ocresult = -1
            result = self.visual_compute(ocresult)
            if result:
                res.ocstate = 1200 if ocresult == 0 else 1201
                res.state = 1100
            else:
                res.state = res.ocstate = 1001
        return True

if __name__ == "__main__":
    rospy.init_node('oc_judge_node')
    oc_judge = OcJudge()
    rospy.spin()