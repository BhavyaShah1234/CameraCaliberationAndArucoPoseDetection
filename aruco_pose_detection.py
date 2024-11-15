import math
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z # in radians

cap = cv.VideoCapture(0)
detector = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_50), cv.aruco.DetectorParameters())
aruco_marker_length = 0.01
camera_calibration_parameters_filename = 'calibration_chessboard.yaml'
cv_file = cv.FileStorage(camera_calibration_parameters_filename, cv.FILE_STORAGE_READ) 
mtx = cv_file.getNode('K').mat()
dst = cv_file.getNode('D').mat()
cv_file.release()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    marker_dict = {}
    corners, marker_ids, rejected = cv.aruco.detectMarkers(frame, cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_50), parameters=cv.aruco.DetectorParameters())
    if marker_ids is not None:
        rvecs, tvecs, obj_points = cv.aruco.estimatePoseSingleMarkers(corners, aruco_marker_length, mtx, dst)
        for i, marker_id in enumerate(marker_ids):
            # Store the translation (i.e. position) information
            transform_translation_x = tvecs[i][0][0]
            transform_translation_y = tvecs[i][0][1]
            transform_translation_z = tvecs[i][0][2]
            rotation_matrix = np.eye(4)
            rotation_matrix[0:3, 0:3] = cv.Rodrigues(np.array(rvecs[i][0]))[0]
            r = R.from_matrix(rotation_matrix[0:3, 0:3])
            quat = r.as_quat()
            # Quaternion format
            transform_rotation_x = quat[0]
            transform_rotation_y = quat[1]
            transform_rotation_z = quat[2]
            transform_rotation_w = quat[3]
            
            # Euler angle format in radians
            roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x, transform_rotation_y, transform_rotation_z, transform_rotation_w)
            roll_x = math.degrees(roll_x)
            pitch_y = math.degrees(pitch_y)
            yaw_z = math.degrees(yaw_z)
            frame = cv.drawFrameAxes(frame, cameraMatrix=mtx, distCoeffs=dst, rvec=rvecs, tvec=tvecs, length=aruco_marker_length, thickness=1)
    cv.imshow('Frame', frame)
    if cv.waitKey(1) == 27:
        break
cap.release()
