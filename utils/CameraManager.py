#!/usr/bin/env python3
from PIL import Image
from numpy.core.fromnumeric import nonzero
import rospy
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import ros_numpy

from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.point_cloud2 import PointCloud2
from std_msgs.msg import Bool
import numpy as np

bridge = CvBridge()
rospy.init_node('test_init', anonymous=True)
class CameraBuilder:
    COLORSTRAM = 0b0001
    DEPTHSTREAM = 0b0010
    POINTCLOUD = 0b0100
    COLORPOINT = 0b0101
    COLORDEPTH = 0b0011

    def __init__(self, params={}):
        
        # self.camerainfos_topic = params.get('cameraInfos', None)
        self.colorstream_topic = params.get('colorStream', None)
        
        # self.pointclouds_topic = params.get('pointclouds', None)
        self.depthstream_topic = params.get('depthStream', None)
        
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
                      'Extrinsic': calib_config["matrix"],
                      'type': calib_config["name"]}
        self._is_registered = False
        self._registered_mode = 0b0000
        # self._registered_mode = 0b0001

        # self._get_info()
        # self.depth = np.empty((0, self._info['height'], self._info['width']), dtype=np.uint16)

        self.update_rate = 1.
        self.msg_t0 = -1.
        self.msg_tn = 0
        self.status_timer = rospy.Timer(rospy.Duration(1.0 / self.update_rate), self._status_callback)
        self.status_pub = rospy.Publisher(f'{self._get_name()}_status_msg', Bool, queue_size=1)

        self.n_frame = 1
        print('CameraManager Init clear')
    
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
            pc = np.zeros((tmp_pc.shape[0], tmp_pc.shape[1], 3))
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
            
            # __depth = np.concatenate((self.depth, np.expand_dims(dep, 0)), 0)
            # if len(__depth) > self.n_frame:
            #     __depth = __depth[-self.n_frame:]

            self.color = bgr
            self.depth = dep
            
        except CvBridgeError as e:
            print(e)

    def _color_callback(self, colorStream_data):
        print('color_callback')
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
        # rgb + depth = 0b0011
        # rgb = 0b0001
        cam_mode = 0b0011 
        
        if cam_mode is self.COLORSTRAM:
            
            color_sub = message_filters.Subscriber(self.colorstream_topic, Image)
            
            color_sub.registerCallback(self._color_callback)
            
            self._is_registered = True
        # elif cam_mode is self.POINTCLOUD:
        #     point_sub = message_filters.Subscriber(self.pointclouds_topic, PointCloud2)
        #     point_sub.registerCallback(self._points_callback)
        #     self._is_registered = True
        # elif cam_mode is self.COLORPOINT:
        #     color_sub = message_filters.Subscriber(self.colorstream_topic, Image)
        #     point_sub = message_filters.Subscriber(self.pointclouds_topic, PointCloud2)
        #     ts = message_filters.ApproximateTimeSynchronizer([color_sub, point_sub], 2, 0.1)
        #     ts.registerCallback(self._cam_callback)
        #     self._is_registered = True
        elif cam_mode is self.COLORDEPTH:
            print('color depth mode ')
            color_sub = message_filters.Subscriber(self.colorstream_topic, Image)
            depth_sub = message_filters.Subscriber(self.depthstream_topic, Image)
            ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.1)
            ts.registerCallback(self._color_depth_callback)
            self._is_registered = True
        else:
            print("any topic")

        self._registered_mode = cam_mode
