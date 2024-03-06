import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sineva_vision.srv import *

class OcJudge:
    def __init__(self):
        self.image_topic_name = rospy.get_param("/topic/image")
        self.yolo_model = YOLOv5cls(ros.package.get_path("visual_oc") + "/model/best.onnx")
        self.client_srv_get_image = rospy.ServiceProxy('/hikrobot/getImage', sineva_vision.GetImage)
        self.img_current_vec = []
        self.camera_state = False

    def start(self):
        pass

    def run(self):
        self.trigger_service = rospy.Service('/visual/judge_oc', JudgeOC, self.trigger)
        rospy.Subscriber('/hikrobot_camera/mono8', Image, self.image_callback)
        rospy.spin()

    def visual_compute(self, result):
        getImageSrv = sineva_vision.GetImage()
        getImageSrv.num_image = 1
        for i in range(3):
            response = self.client_srv_get_image(getImageSrv)
            if response.state == 1:
                break
            rospy.loginfo("Reconnect camera...")
            rospy.sleep(1)
        
        if response.state == 0:
            self.camera_state = False
            return False
        
        if response.state == 1:
            rospy.loginfo("Camera is success!")
            num = 0
            self.camera_state = True
            for image_msg in response.imgs:
                cv_ptr = CvBridge().imgmsg_to_cv2(image_msg, desired_encoding="mono8")
                image = cv2.cvtColor(cv_ptr, cv2.COLOR_BGR2GRAY) if cv_ptr.shape[2] == 3 else cv_ptr
                
                if image is None:
                    rospy.logwarn("The image is empty")
                    return False
                
                self.img_current_vec.append(image.copy())
                num += 1
                if len(self.img_current_vec) > calculation_image_num_:
                    self.img_current_vec.pop()
        
        rospy.loginfo("Image num: {}".format(len(self.img_current_vec)))
        
        while self.img_current_vec:
            img_ori = self.img_current_vec.pop(0)
            result = self.yolo_model.detect(img_ori)
            timeStr = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            name_temp = ros.package.get_path("visual_oc") + "/tempfile/" + timeStr + "_" + str(result) + ".png"
            cv2.imwrite(name_temp, img_ori)
        
        return True

    def trigger(self, req):
        projectNum = req.projectNum
        rospy.logwarn("Receive Trigger msg")
        
        if projectNum == 6:
            ocresult = -1
            result = self.visual_compute(ocresult)
            
            if result:
                if ocresult == 0:
                    ocstate = 1200
                elif ocresult == 1:
                    ocstate = 1201
                else:
                    rospy.logwarn("YOLO classify result is -1")
                    ocstate = 1001
                
                state = 1100
            else:
                rospy.logwarn("No result")
                state = 1001
                ocstate = 1001
            
            rospy.loginfo("res.state: {}, res.ocstate: {}".format(state, ocstate))
            return state, ocstate

if __name__ == '__main__':
    rospy.init_node('code128')
    oc_judge = OcJudge()
    oc_judge.run()