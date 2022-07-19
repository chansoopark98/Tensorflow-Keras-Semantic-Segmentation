#!/usr/bin/env python3
import os
import sys
from PIL import Image
from numpy.core.fromnumeric import nonzero
import rospy
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import ros_numpy

from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.point_cloud2 import PointCloud2
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Bool
import tensorflow as tf
import numpy as np
import cv2
import random
import vision_utils as putils
from vision.srv import srv_grasp, srv_graspResponse

from visualization_msgs.msg import Marker

global center_x
global center_y 
global marker_id
marker_id = 0
rospy.init_node('vision', anonymous=True)

bridge = CvBridge()

class CameraManager:
    COLORSTRAM = 0b0001
    def __init__(self, params={}):
        self.camerainfos_topic = params.get('cameraInfos', None)
        self.colorstream_topic = params.get('colorStream', None)
        self.pointclouds_topic = params.get('pointclouds', None)
        self.depthstream_topic = params.get('depthStream', None)
        self.depthstream_topic = None
        calib_config = params.get('calibration', None)
        self.color = None
        self.points = None
        self.depth = None
        self._info = {
                      'name': params.get('name', None),
                      'width': 0,
                      'height': 0,
                      'K': 0,
                      'D': 0,
                      'Extrinsic': calib_config["concent_matrix"],
                      'type': calib_config["name"]}
        self._is_registered = False
        self._registered_mode = 0b0000

        self._get_info()
        self.depth = np.empty((0, self._info['height'], self._info['width']), dtype=np.uint16)

        self.update_rate = 10.
        self.msg_t0 = -1.
        self.msg_tn = 0
        self.status_timer = rospy.Timer(rospy.Duration(1.0 / self.update_rate), self._status_callback)
        self.status_pub = rospy.Publisher(f'{self._get_name()}_status_msg', Bool, queue_size=1)

        self.n_frame = 10
    
    def _get_name(self):
        return self._info['name']

    def _get_info(self):
        if self.camerainfos_topic is not None:
            cam_info = rospy.wait_for_message(self.camerainfos_topic, CameraInfo, timeout=None)
            self._info['width'] = cam_info.width
            self._info['height'] = cam_info.height
            self._info['K'] = np.reshape(cam_info.K, (-1,3))
            self._info['D'] = np.array(cam_info.D)
        else:
            rospy.logwarn("CameraManager:: camera info isn\'t given.")

    def _status_callback(self, timer):
        if self._is_registered:
            status_msg = Bool()
            if self.msg_t0 == self.msg_tn:
                rospy.logwarn("CameraManager:: no new messages")
                # self._clear_images(h=self.get_height(), w=self.get_width())
                status_msg = False
            else:
                self.msg_t0 = self.msg_tn
                status_msg = True
            self.status_pub.publish(status_msg)

    def _cam_callback(self, colorStream_data, pointcloud_data):
        self.msg_tn = rospy.get_rostime().to_sec()
        try:
            bgr = bridge.imgmsg_to_cv2(colorStream_data, desired_encoding='bgr8')

            tmp_pc = ros_numpy.numpify(pointcloud_data)
            # pc = np.zeros((tmp_pc.shape[0], tmp_pc.shape[1], 3))

            tmp_pc = np.reshape(tmp_pc, (bgr.shape[0], bgr.shape[1]))
            pc = np.zeros((bgr.shape[0], bgr.shape[1], 3))

            pc[:,:,0] = tmp_pc['x']
            pc[:,:,1] = tmp_pc['y']
            pc[:,:,2] = tmp_pc['z']

            self.color = bgr
            self.points = pc

        except CvBridgeError as e:
            print(e)

    def _color_depth_callback(self, colorStream_data, depthStream_data):
        self.msg_tn = rospy.get_rostime().to_sec()
        try:
            bgr = bridge.imgmsg_to_cv2(colorStream_data, desired_encoding='bgr8')
            dep = bridge.imgmsg_to_cv2(depthStream_data, desired_encoding='passthrough')

            __depth = np.concatenate((self.depth, np.expand_dims(dep, 0)), 0)
            if len(__depth) > self.n_frame:
                __depth = __depth[-self.n_frame:]

            self.color = bgr
            self.depth = __depth
            
        except CvBridgeError as e:
            print(e)

    def _color_callback(self, colorStream_data):
        self.msg_tn = rospy.get_rostime().to_sec()
        try:
            bgr = bridge.imgmsg_to_cv2(colorStream_data, desired_encoding='bgr8')
            self.color = bgr
            
        except CvBridgeError as e:
            print(e)
    
    def _points_callback(self, pointcloud_data):
        self.msg_tn = rospy.get_rostime().to_sec()
        try:
            tmp_pc = ros_numpy.numpify(pointcloud_data)
            pc = np.zeros((tmp_pc.shape[0], tmp_pc.shape[1], 3))
            pc[:,:,0] = tmp_pc['x']
            pc[:,:,1] = tmp_pc['y']
            pc[:,:,2] = tmp_pc['z']

            self.points = pc
        
        except CvBridgeError as e:
            print(e)

    def _clear_images(self, h, w):
        self.depth = np.empty((0, h, w), dtype=np.uint16)

    def get_width(self):
        return self._info["width"]

    def get_height(self):
        return self._info["height"]

    def get_size(self):
        return (self._info['width'], self._info['height'])

    def get_intrinsic(self):
        return self._info["K"]

    def get_extrinsic(self):
        return self._info["Extrinsic"]
    
    def get_extrinsic_type(self):
        return self._info["type"]

    def register_cb(self):
        color_sub = message_filters.Subscriber(self.colorstream_topic, Image)
        point_sub = message_filters.Subscriber(self.pointclouds_topic, PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, point_sub], 2, 0.1)
        ts.registerCallback(self._cam_callback)
        self._is_registered = True


class DLManager():
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../src/Concent-pose-detection'))  
    
    CONCENT = 'concent'

    def __init__(self, model=CONCENT, model_path=None, num_classes=0, detection_min_confidence=0.9):
        from models.model_builder import semantic_model
        from data_labeling.utils import find_contours

        # model pretrained weight 경로 불러오기
        self.seg_path = model_path
        
        # semantic segmnetation 모델 입력 해상도 설정 (default : RGB이미지 512x512)
        self.IMAGE_SIZE = (512, 512)

        if model is DLManager.CONCENT:
            hex_color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            self.rgb_color_list = [tuple(int(ci[i:i+2], 16) for i in (5, 3, 1)) for ci in hex_color_list]
            
            if os.path.exists(model_path):
                # Semantic segmentation 모델 불러오기
                self.model = semantic_model(image_size=self.IMAGE_SIZE)
                
                # 불러온 모델에 pretrained weight 설정
                self.model.load_weights(self.seg_path)

                # 첫 추론 시 time delay를 줄이기 위해 dummy data를 만들어 inference 수행
                img = tf.zeros([self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], 3])
                img = tf.cast(img, dtype=tf.float32)
                img = tf.expand_dims(img, axis=0)
                _ = self.model.predict_on_batch(img)

                rospy.logdebug(f"DLManager:: Loading weights {model_path} for grasping.")
            else:
                rospy.logerr(f"DLManager:: {model_path} doesn\'t exist. Fail to load weights.")
    
    def inference(self, img):
        # Concent hole segmentation 결과 확인을 위한 BGR image copy
        original_img = img.copy()

        # DL network는 RGB image를 사용하기 때문에 BGR 이미지를 RGB 이미지로 컨버팅
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Pretrained된 DL network의 입력 해상도는 3채널 RGB image 512x512 (H*W)
        img = tf.image.resize(img, size=(self.IMAGE_SIZE[0], self.IMAGE_SIZE[1]),
                method=tf.image.ResizeMethod.BILINEAR)
        # 8비트 Int type의 이미지를 32비트 float형으로 변경
        img = tf.cast(img, tf.float32)
        # 0~255까지의 픽셀 값 범위를 0~1사잇값으로 정규화
        img /= 255.
        # 딥러닝 추론 시 4차원 배치 처리를 위해 차원 확장 (H, W, C) -> (B, H, W, C)
        img = tf.expand_dims(img, axis=0)
        # 모델 추론
        model_output = self.model.predict_on_batch(img)

        """
            !!  << model_output = self.model.predict_on_batch(img) >> !! 
        기존 pretrained된 semantic segmentation 모델의 경우 출력 형태가 logits이다.
        분류할 클래스 수 만큼 출력 채널이 결정된다. (512x512@4)
        0 - background
        1 - concent
        2 - plug
        3 - hole
        
        각 채널별로 픽셀값이 구분되어 있어서 argmax 함수를 사용하여 하나의 채널로 압축한다
        """
        # 각 채널별로 분류된 픽셀을 하나의 채널로 합친다 (H, W, Classes) -> (H, W, 1)
        pred = tf.argmax(model_output, axis=-1)
        # 원본 해상도 (1024, 1024)로 복원하기 위해 resize를 수행해야 함, 그러나, tf.image.resize 함수는 4차원 형태의 데이터를 취급하여 차원 확장 후 크기 조정
        pred = tf.expand_dims(pred, axis=-1)
        pred = tf.image.resize(pred, size=(self.IMAGE_SIZE[0] * 2, self.IMAGE_SIZE[1] * 2),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # 4차원의 semantic segmentation map을 2차원의 행렬로 변환 (B, H, W, 1) -> (H, W)
        pred = pred[0, :, :, 0]
         
        # 예측한 segmentation map을 사용하여 mask를 생성
        mask = pred.numpy().astype(np.uint8)

        """
        생성된 mask 이미지에서 가장 외곽의 contour를 추출 
        cv2.RETR_EXTERNAL -> 가장 외부의 contours만 추출하는 파라미터
        mask 이미지에서 가장 외부 contours는 콘센트를 의미한다
        """
        # 콘센트 contours 추출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 검출에 사용 할 외부 contours list 선언
        contour_list = []
        
        # 추론 결과에서 오분할된 작은 noise contours를 무시하기 위해 contour의 내부 영역 크기를 비교하여 일정 크기 이상의 contours만 사용
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= 1000: # contour 크기 (1000) 값은 이미지 해상도, object의 물체 크기에 따라 수동으로 조정해야 함
                contour_list.append(contour)


        # 각 콘센트를 구분하기 위한 zero mask
        zero_mask = np.zeros(original_img.shape)[:, :, 0].astype(original_img.dtype)

        # 각 콘센트의 hole contour를 그리기 위한 zero mask -> 시각화 용
        hole_zero_mask = np.zeros(original_img.shape)[:, :, 0].astype(np.uint8)
        
        """
        하나의 콘센트에서 hole의 갯수는 8개이다
        이 중 랜덤으로 하나의 홀을 지정 (select_list)한다.
        지정한 홀에서 가장 근접한 홀 (near_list)을 찾는다.
        """
        select_list = []
        near_list = []

        # Concent의 갯수만큼 반복
        for idx in range(len(contour_list)):
            
            # 콘센트를 구분하기 위한 zero mask를 복사 (Concnet 갯수만큼 반복하여 그리기 때문에 덮어쓰는 문제를 방지)
            draw_zero_mask = zero_mask.copy()

            # mask는 semantic segmentation argMax 결과값 (Concnet 갯수만큼 반복하여 그리기 때문에 덮어쓰는 문제를 방지)
            draw_mask = mask.copy()

            # drawContours를 이용하여 하나의 concent에 대한 binary mask 이미지를 생성
            cv2.drawContours(draw_zero_mask, contour_list, idx, 1, -1)
            
            # 생성된 binary mask를 segmentation 결과에 곱연산하여 concent 영역 외의 결과는 0값 처리한다
            draw_mask *= draw_zero_mask

            # class pixel 값이 3 (hole을 의미)인 경우 픽셀값을 255, 그 외에는 0값 처리
            draw_plug = np.where(draw_mask==3, 255, 0)

            # uint 8비트 타입으로 변환
            draw_plug = draw_plug.astype(np.uint8)

            # Hole class만 존재하는 segmentation image에서 hole contours를 찾는다
            hole_contours, _ = cv2.findContours(draw_plug, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)       

            # Hole contours들의 중심점 (cx, cy)를 반환하기 위한 list 선언
            output_coords = []

            # Hole 개수만큼 반복
            for hole_idx in range(len(hole_contours)):
                cv2.drawContours(hole_zero_mask, hole_contours, hole_idx, 127, -1)
                M = cv2.moments(hole_contours[hole_idx])
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.circle(original_img, (cx, cy), 2,(255, 0, 0), 1, cv2.LINE_AA)
                
                # 각 홀의 cx, cy 반환
                output_coords.append([cx, cy])
            
            # Python list를 numpy array로 변환
            output_coords = np.array(output_coords)
            

            # 랜덤으로 지정한 hole에서 가까운 거리에 위치해있는 hole을 찾기 위한 temp 변수
            close_coord = 50000

            try:
                # Hole list의 길이만큼 랜덤값으로 index를 지정
                random_idx = random.randint(0, len(output_coords)-1)
                # 랜덤으로 지정한 hole의 중심점 (cx, cy)과 가까운 hole의 index를 가져오기 위한 temp 변수
                close_idx = 0

                # Hole 개수만큼 반복
                for coord_idx in range(len(output_coords)):

                    # 처음에 Random으로 지정한 index는 고려하지 않음 (같은 index를 비교하는 경우 거리값이 0이기 때문에 무효)
                    if coord_idx != random_idx:
                        
                        # 랜덤으로 지정한 홀의 중심점과 각 hole의 거리 벡터를 계산 (벡터 정규화를 사용하여 계산)
                        abs_coord = np.linalg.norm(output_coords[coord_idx] - output_coords[random_idx])
                        
                        # 기존에 저장된 temp 변수보다 벡터값이 작은 경우 가장 가까운 hole index라고 할 수 있음
                        if close_coord > abs_coord:
                            close_coord = abs_coord
                            close_idx = coord_idx
                                
                # 출력 결과 시각화
                cv2.circle(original_img, (output_coords[random_idx][0], output_coords[random_idx][1]), 3, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.circle(original_img, (int(output_coords[close_idx][0]), int(output_coords[close_idx][1])), 5, (0,0,255), 1, cv2.LINE_AA)

                # 랜덤으로 지정한 hole의 중심점 (cx, cy)와 가까운 hole의 중심점 (cx, cy)를 반환
                select_list.append([output_coords[random_idx][0], output_coords[random_idx][1]])
                near_list.append([output_coords[close_idx][0], output_coords[close_idx][1]])
                
            except:
                print('no detected')
            
        return select_list, near_list, mask



class ProcessManager():
    # import json
    class VisionProcess():
        ANGLE_FUNC = {
            "rbb": putils.get_angle_rbb,
            "pcl": putils.get_angle_pca
        }
        def __init__(self, service_name='/camera01_result0a_srv', camera=None, network=None, cat_config=None, pre_config=None, post_config=None):
            self.service = None

            self.service_name = service_name
            self.camera = camera
            self.network = network

            self.roi_flag = False
            self.roi_xs = -1
            self.roi_ys = -1
            self.roi_xe = -1
            self.roi_ye = -1
            self.resize_flag = False
            self.resize_w = -1
            self.resize_h = -1
            self._preprocess_config(pre_config=pre_config)

            self.categories_remap = {}
            self.categories_name = {}
            self.categories_color = {}
            self._categories_config(cat_config=cat_config)

            self.categories_postfunc = {}
            self._postprocess_config(post_config=post_config)

            if self.roi_flag:
                self.network.cam_data.set_crop_attrs(1280, 720, self.roi_xs, self.roi_ys, self.roi_xe, self.roi_ye)
                self.scale = self.network.cam_data.zoom
            self.result_img_pub = rospy.Publisher(service_name.replace('_srv', '_img'), Image, queue_size=1)
            self.result_seg_pub = rospy.Publisher('/segmentation_results', Image, queue_size=1)
            self.result_quality_pub = rospy.Publisher(service_name.replace('_srv', '_quality'), Image, queue_size=1)
            self.marker_pub = rospy.Publisher("/visualization_marker/pub", Marker, queue_size = 1)

        def _preprocess_config(self, pre_config=None):
            keys = pre_config.keys()
            
            if 'roi' in keys:
                roi_config = pre_config['roi']
                if (roi_config['x0'] > -1) & (roi_config['y0'] > -1) & (roi_config['w'] > 0) & (roi_config['h'] > 0):
                    self.roi_xs = max(roi_config["x0"], 0)
                    self.roi_ys = max(roi_config["y0"], 0)
                    self.roi_xe = self.roi_xs + min(roi_config["w"], self.camera.get_width())
                    self.roi_ye = self.roi_ys + min(roi_config["h"], self.camera.get_height())
                    self.roi_flag = True
            
            if 'resize' in keys:
                resize_config = pre_config['resize']
                if (resize_config["w"] > 0) & (resize_config["h"] > 0):
                    self.resize_w = resize_config["w"]
                    self.resize_h = resize_config["h"]
                    self.resize_flag = True
            
        def _categories_config(self, cat_config=None):
            for _, category in enumerate(cat_config):
                _id = category["id"]
                self.categories_remap[_id] = category["remap"]
                self.categories_name[_id] = category["name"]
                self.categories_color[_id] = category["color"]
        
        def _postprocess_config(self, post_config=None):
            for _, proc in enumerate(post_config):
                _id = proc["category_id"]
                angle_method = proc["angle"]
                self.categories_postfunc[_id] = self.ANGLE_FUNC[angle_method] if angle_method is not None else None
        

        def _pub_result_image(self, pub_img):
            try:
                img_msg = bridge.cv2_to_imgmsg(pub_img, encoding='bgr8')
                # img_msg = bridge.cv2_to_imgmsg(pub_img)
                self.result_img_pub.publish(img_msg)
            except CvBridgeError as e:
                print(e)

        def _pub_test_image(self, pub_img):
            try:
                img_msg = bridge.cv2_to_imgmsg(pub_img)
                # img_msg = bridge.cv2_to_imgmsg(pub_img)
                self.result_seg_pub.publish(img_msg)
            except CvBridgeError as e:
                print(e)
        
        def _pub_result_pose_array(self, poses):
            global marker_id
            
            try:
                marker = Marker()
                marker.header.frame_id = "/rgb_camera_link"
                marker.header.stamp = rospy.Time.now()
                
                marker.type = marker.CYLINDER
                marker.id = marker_id
                # marker.frame_locked = True
                marker.action = marker.ADD
                marker.ns = "vision"

                # Set the scale of the marker
                marker.scale.x = 0.02
                marker.scale.y = 0.02
                marker.scale.z = 0.02

                # Set the color
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0


                rotation_mat = poses[:3,:3]# rotation
                roll, pitch, yaw = putils.euler_from_matrix(rotation_mat)
                quaternion = putils.euler_to_quaternion(roll, pitch, yaw)
                qx, qy, qz, qw = quaternion

                dx, dy, dz = poses[:3,3] # transrations

                # Set the pose of the marker
                marker.pose.position.x = dx
                marker.pose.position.y = dy
                marker.pose.position.z = dz
                marker.pose.orientation.x = qx
                marker.pose.orientation.y = qy
                marker.pose.orientation.z = qz
                marker.pose.orientation.w = qw

                self.marker_pub.publish(marker)

                marker_id +=1

            except CvBridgeError as e:
                print(e)


        def _calib_poses(self, obj_poses, calib_mat):
            """
            예측한 object pose (4x4)와 Auto calibration을 통해 얻은 camera extrinsic matrix (4x4)를 이용하여 캘리브레이션 수행
            Args:
                obj_poses (List) : 각 콘센트 당 하나의 pose (4x4 matrix)를 가지고 있으며 N개의 출력을 가집니다
                calib_mat (Numpy array) : Auto calibration을 통해 얻은 4x4 numpy array (matrix)

            Return:
                rs_poses : 캘리브레이션된 오브젝트 pose
            """

            stamp = rospy.Time.now()
            rs_poses = PoseArray()
            rs_poses.header.stamp = stamp

            # 여러 개의 pose를 return하기 위해 빈 배열을 선언
            calib_poses = np.empty((0,4,4))

            # 예측한 N개의 concent pose만큼 반복하여 캘리브레이션을 수행
            for i in range(len(obj_poses)):
                obj_pose = obj_poses[i]
                calib_pose = np.dot(calib_mat, obj_pose)

                # 계산된 pose는 0번째 축으로 이어붙임
                calib_poses = np.append(calib_poses, calib_pose.reshape( 1, 4, 4 ), axis=0 )
                        
            print('pose array shape: ')
            print(calib_poses.shape[0])

            # 계산한 N개의 pose만큼 반복하여 연산
            for idx in range(calib_poses.shape[0]):
                pose = calib_poses[idx,:,:]

                # 3x3 rotation matrix를 오일러 각으로 변환
                r, p, y = putils.euler_from_matrix(pose[:3,:3])

                # 오일러 각을 쿼터니언각으로 변환 (x, y, z, w)
                q = putils.euler_to_quaternion(r, p, y)
                
                # ROS messeage type인 Pose 인스턴스 생성
                new_pose = Pose()

                # Pose 인스턴스 속성에 transformation x, y, z와 rotation x, y, z, w를 설정
                new_pose.position.x = pose[0,3]
                new_pose.position.y = pose[1,3]
                new_pose.position.z = pose[2,3]
                new_pose.orientation.x = q[0]
                new_pose.orientation.y = q[1]
                new_pose.orientation.z = q[2]
                new_pose.orientation.w = q[3]
                
                rs_poses.poses.append(new_pose)
                # print("Object #{} position(base_coord)".format(idx))
                print(pose[:3,3])
                print('orientation(euler)')
                print((np.degrees(r), np.degrees(p), np.degrees(y)))

            return rs_poses
    

        def register_cb(self):
            self.camera.register_cb()
            self.service = rospy.Service(self.service_name, srv_grasp, self.service_callback)

        def _draw_transform(self, img, trans):
            arrow = np.array([[0,0,0],[0.05,0,0],[0,0.05,0],[0,0,0.05]])
            arrow = np.dot(arrow, trans[:3, :3].T) + trans[:3, 3]
            arrow_p2ds = putils.project_p3d(arrow, 1.0, self.camera.get_intrinsic())
            img = putils.draw_arrow(img, arrow_p2ds, thickness=2)
            return img


        def calc_pca(self, rgb, pcd, select_list, random_list):
            output_list = []
            x, y, w, h = 411, 201, 1024, 1024
            
            for i in range(len(select_list)):
                pointCloud_area = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
                select_x, select_y = select_list[i]
                select_x += x
                select_y += y
                random_x, random_y = random_list[i]
                random_x += x
                random_y += y
                pointCloud_area = cv2.line(pointCloud_area, (select_x, select_y), (random_x, random_y), 255, 10, cv2.LINE_AA)
                
                self._pub_test_image(pointCloud_area)

                pointCloud = pcd.copy()
                
                choose_pc = pointCloud[np.where(pointCloud_area[:, :]==255)]
                
                choose_pc = choose_pc[~np.isnan(choose_pc[:,2])]
                
                
                # PCA 통해 주성분 벡터, 평균, 분산
                meanarr, comparr, vararr = cv2.PCACompute2(choose_pc, mean=None)

                vararr = vararr.reshape(-1)

                comparr = -comparr
                if comparr[2, 2] < 0:
                    comparr[2, :3] = -comparr[2, :3]
                
                # Target Pose 생성
                target_pose = np.identity(4)
                target_pose[:3,:3] = comparr.T # rotation
                target_pose[:3,3] = meanarr # transration
                
                output_list.append(target_pose)

            return output_list

        def service_callback(self, req):
            """
            Service call message가 들어오면 callback function을 수행
            """
            finish = False
            msg_posearray = PoseArray()
            msg_posearray.header.stamp = rospy.Time.now()
            msg_widtharray = []

            # CameraManager를 통해 rgb image topic을 불러오고, numpy array (cv)로 변환한 값을 복사하여 가져옴
            _color = self.camera.color.copy()

            # 원본 이미지에서 콘센트를 찾을 테이블 영역만 crop하여 사용 (해상도 크기 : 1024x1024)
            x, y, w, h = 411, 201, 1024, 1024
            roi_input_img = _color.copy()[y:y+h, x:x+w]

            # Inference: Hole에 해당하는 x, y값들을 반환한다
            select_list, random_list, mask = self.network.inference(roi_input_img)

            # Hole의 x, y좌표와 포인트 클라우드를 이용하여 pose를 계산한다
            target_pose_list = self.calc_pca(_color, self.camera.points, select_list, random_list)   
            
            # 예측한 pose 수만큼 반복하여 연산
            for i in range(len(target_pose_list)):
                # 예측한 포즈를 이용하여 rgb image에 pose Arrow를 그림
                _color = self._draw_transform(img=_color, trans=target_pose_list[i])
                # Rviz에 예측한 pose를 marker를 이용하여 visualize 
                self._pub_result_pose_array(target_pose_list[i])
            
            # RQT에 얘측한 pose를 visualize
            self._pub_result_image(_color)

            # 예측한 포즈를 카메라 외부 파라메타 (calibration matrix)를 이용하여 calibration 수행
            calib_poses = self._calib_poses(target_pose_list, self.camera.get_extrinsic())
            msg_posearray.poses = calib_poses.poses

            if len(msg_posearray.poses) == 0:
                msg_widtharray = []
            else:
                msg_widtharray = []
        
            finish =True           
            
            return srv_graspResponse(msg_widtharray, msg_posearray, finish)

    def __init__(self, config_path="/home/park/catkin_ws/camera_infos.json"):
        self.config_data = None
        self._read_config(config_path)
        self._processes = []
        self._configuration(self.config_data)

    def _read_config(self, config_path):
        import json
        if os.path.exists(config_path):
            rospy.logdebug(f"ProcessManager:: process configuration. config_file: {config_path}")
            with open(config_path, 'r') as f_config:
                self.config_data = json.load(f_config)
        else:
            rospy.logerr(f"ProcessManager:: can\'t open config file. {config_path}")
    
    def _configuration(self, config_data):
        camera_config_data = config_data["cameras"]
        visionprocs_config_data = config_data["vision_procs"]

        # camera configuration
        cameras = []
        for idx, cam in enumerate(camera_config_data):
            topic_info = cam["topics"]
            camera = CameraManager({'name':cam["name"],'cameraInfos':topic_info["info"],'colorStream':topic_info["color"],'depthStream':topic_info["depth"], 'pointclouds':topic_info["points"], 'calibration':cam["calibration"]})
            cameras.append(camera)
        
        processes = []
        for idx, proc in enumerate(visionprocs_config_data):
            dl = DLManager(model=DLManager.CONCENT, model_path=(proc["seg_path"]))
            processes.append(self.VisionProcess(service_name=proc["service"], camera=cameras[proc["camera_id"]], network=dl, cat_config=proc["categories"], pre_config=proc["preprocessing"], post_config=proc["postprocessing"]))
        self._processes = processes
        

    def register_cb(self):
        for idx, process in enumerate(self._processes):
            process.register_cb()

def main():
    pm = ProcessManager()
    pm.register_cb()
    print("==================================================================")
    print('|                                                                |')
    print('|                         Start!!                                |')
    print('|                                                                |')
    print("==================================================================")
    rospy.spin() 



if __name__ == "__main__":
    main()