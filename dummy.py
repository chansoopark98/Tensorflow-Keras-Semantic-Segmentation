TEST_IMG = rospy.Publisher('JH_IMG_TEST', Image, queue_size=1)

jh_img = 0

def jh_img_callback(img):
    rospy.get_rostime().to_sec()
    global jh_img
    bgr = bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')
    TEST_IMG.publish(img)
    jh_img = bgr

color_sub = message_filters.Subscriber('mv_cam', Image) 
color_sub.registerCallback(jh_img_callback)
