import numpy as np
from math import *
import cv2
import open3d as o3

def project_p3d(p3d, cam_scale, K):
    p3d = p3d * cam_scale
    p2d = np.dot(p3d, K.T)
    p2d_3 = p2d[:, 2]
    p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
    p2d[:, 2] = p2d_3
    p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
    return p2d

def draw_arrow(img, p2ds, thickness=1):
    h, w = img.shape[0], img.shape[1]
    for idx, pt_2d in enumerate(p2ds):
        p2ds[idx, 0] = np.clip(pt_2d[0], 0, w)
        p2ds[idx, 1] = np.clip(pt_2d[1], 0, h)
    
    img = cv2.arrowedLine(img, (p2ds[0,0], p2ds[0,1]), (p2ds[1,0], p2ds[1,1]), (0,0,255), thickness)
    img = cv2.arrowedLine(img, (p2ds[0,0], p2ds[0,1]), (p2ds[2,0], p2ds[2,1]), (0,255,0), thickness)
    img = cv2.arrowedLine(img, (p2ds[0,0], p2ds[0,1]), (p2ds[3,0], p2ds[3,1]), (255,0,0), thickness)
    return img


def draw_grasp_bbox(img, bbox, thickness=1):
    return cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), thickness)

def draw_graspI(img, p2ds, color, thickness=1):
    h, w = img.shape[0], img.shape[1]
    for idx, pt_2d in enumerate(p2ds):
        p2ds[idx, 0] = np.clip(pt_2d[0], 0, w)
        p2ds[idx, 1] = np.clip(pt_2d[1], 0, h)
    
    img = cv2.line(img, (p2ds[0,0], p2ds[0,1]), (p2ds[1,0], p2ds[1,1]), color, thickness)
    img = cv2.line(img, (p2ds[2,0], p2ds[2,1]), (p2ds[3,0], p2ds[3,1]), color, thickness)
    img = cv2.line(img, (p2ds[4,0], p2ds[4,1]), (p2ds[5,0], p2ds[5,1]), color, thickness)
    return img

def get_angle_pca(pts, pointclouds=None):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
    return None, angle, eigenvectors, eigenvalues

def get_angle_rbb(pts, pointclouds):
    rect = cv2.minAreaRect(pts)
    center = np.int0(rect[0])
    width, height = rect[1]
    if width > height:
        angle = np.radians(rect[2])
    else:
        angle = np.radians(rect[2]+90)

    trans = pointclouds[center[1], center[0], :]
    trans[2] = np.nanmedian(pointclouds[center[1]-5:center[1]+6, center[0]-5:center[0]+6, 2])
    return trans, angle, None, None

def euler_from_quaternion(x, y, z, w, degree=False):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = atan2(t3, t4)
     
        if degree:
            return np.degrees(roll_x), np.degrees(pitch_y), np.degrees(yaw_z)
        else:
            return roll_x, pitch_y, yaw_z

def euler_from_matrix( rot, degree=False ):
    sy = sqrt( rot[2, 1]*rot[2, 1]+rot[2, 2]*rot[2, 2] )
    singular = sy < 1e-6

    roll = atan2( rot[2, 1], rot[2, 2] )
    pitch = atan2(-rot[2, 0], sqrt( rot[2, 1]*rot[2, 1]+rot[2, 2]*rot[2, 2] ) )
    yaw = atan2( rot[1, 0], rot[0, 0] )
    
    if degree:
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
    else:
        return roll, pitch, yaw

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

def euler_to_matrix(roll, pitch, yaw):
    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0, 1]])
    Ry_pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx_roll = np.array([
        [1,            0,             0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]])
    # R = RzRyRx
    rotMat = np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))
    return rotMat


def calc_calibMatrix(obj_pose, ee2cam, base2ee):
    ee_rorate = True

    # calc end effector -> camera
    # roration이 없는 행렬 선언
    mat1 = np.array([
    [1, 0,  0,  ee2cam['x']],
    [0, 1,  0,  ee2cam['y']],
    [0, 0,  1,  ee2cam['z']],
    [0, 0,  0,           1]])
    
    # end effector에서 회전이 있는 경우,
    if ee_rorate:
        # use custom function

        # quat_ee = [ee2cam['ori_x'], ee2cam['ori_y'], ee2cam['ori_z'], ee2cam['ori_w']]
        # ee_roll, ee_pitch, ee_yaw = quaternion_to_euler(quat_ee)
        # print('calib quaternion_to_euler', ee_roll, ee_pitch, ee_yaw)
        # ee_rotation_mat = euler_to_matrix(ee_roll, ee_pitch, ee_yaw)

        # use open3d function

        quat_array = np.array([ee2cam['ori_w'], ee2cam['ori_x'], ee2cam['ori_y'], ee2cam['ori_z']])
        ee_rotation_mat = o3.geometry.get_rotation_matrix_from_quaternion(quat_array)

        # print('calib rotation matrix', ee_rotation_mat)

        mat1[:3,:3] = ee_rotation_mat

    # use custom function
    # quat_base = [base2ee['ori_x'], base2ee['ori_y'], base2ee['ori_z'], base2ee['ori_w']]
    # init_roll, init_pitch, init_yaw = quaternion_to_euler(quat_base)
    # angle_mat = euler_to_matrix(init_roll, init_pitch, init_yaw)


    # use open3d function
    quat_base = np.array([base2ee['ori_w'], base2ee['ori_x'], base2ee['ori_y'], base2ee['ori_z']])
    angle_mat = o3.geometry.get_rotation_matrix_from_quaternion(quat_base)

    # print('calib rotation matrix', angle_mat)

    base_calib_mat = np.identity(4)
    base_calib_mat[:3,:3] = angle_mat
    base_calib_mat[0,3] = base2ee['x']
    base_calib_mat[1,3] = base2ee['y']
    base_calib_mat[2,3] = base2ee['z']

    # calibration
    calib_ee = np.dot(mat1, obj_pose)
    # print('calib_ee',calib_ee)
    calib_base = np.dot(base_calib_mat, calib_ee)

    return calib_base
    
