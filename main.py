import cv2
import numpy as np
import skvideo.io

DEFAULT_POINTS = np.array([[0, 1920], [0, 0], [1080, 1920], [1080, 0]])


def _generate_aruco_dict():
    aruco_dict = cv2.aruco.custom_dictionary(0, 4, 1)
    # create bytesList array to fill with 3 markers later
    aruco_dict.bytesList = np.empty(shape=(4, 2, 4), dtype=np.uint8)
    # add new markers
    mybits = np.array([[1, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0]], dtype=np.uint8)
    aruco_dict.bytesList[0] = cv2.aruco.Dictionary_getByteListFromBits(mybits)
    mybits = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 1, 0]], dtype=np.uint8)
    aruco_dict.bytesList[1] = cv2.aruco.Dictionary_getByteListFromBits(mybits)
    mybits = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
    aruco_dict.bytesList[2] = cv2.aruco.Dictionary_getByteListFromBits(mybits)
    mybits = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 0], [1, 1, 0, 1]], dtype=np.uint8)
    aruco_dict.bytesList[3] = cv2.aruco.Dictionary_getByteListFromBits(mybits)

    # adjust dictionary parameters for better marker detection
    parameters = cv2.aruco.DetectorParameters_create()
    # parameters.cornerRefinementMethod = 5
    # parameters.errorCorrectionRate = 0.3

    return aruco_dict, parameters, DEFAULT_POINTS


def _remove_shadow(frame):
    dilated_img = cv2.dilate(frame, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    return 255 - cv2.absdiff(frame, bg_img)


def _denoise(frame):
    return cv2.fastNlMeansDenoising(frame, None)


def _warp_frame(frame, src_points, dst_points, shape):
    homography_mat, status = cv2.findHomography(src_points, dst_points)
    return cv2.warpPerspective(frame, homography_mat, shape)


def _aruco_detection(frame, points):
    locations, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)
    locations = locations[:4]
    if len(locations) > 0:
        raw_new_points = np.empty((0, 2))
        for location in locations:
            raw_new_points = np.append(raw_new_points, np.array([np.mean(location[0], axis=0)]), axis=0)
        # sort those points to get the following order:
        # [topright, topleft, bottomright, bottomleft]
        # the order still be maintained even if some points are missing
        new_points = raw_new_points[np.argsort(raw_new_points[:, 0])]
        # if we got all 4 points (best case) then just replace it
        if new_points.shape[0] == 4:
            points = new_points
        # if we got only 3 points, then we should assume that
        # the missing point is bottomright since almost everyone
        # are right handed
        elif new_points.shape[0] == 3:
            if points.shape[0] == 4:
                np.append(new_points, points[3])
            else:
                np.append(new_points,
                          np.array([points.max(axis=0, initial=0), points.max(axis=1, initial=0)]))
            points = new_points
        # there are no more then 2 points, then we will try to use the old point
        # and replace some of them with the new points
        elif points.shape[0] == 4:
            for idx, point in enumerate(new_points):
                min_idx = 0
                for ref_idx, ref_point in enumerate(points):
                    dist = np.linalg.norm(point - ref_point)
                    if dist < ref_point[min_idx]:
                        min_idx = ref_idx
                points[min_idx] = point
    return _warp_frame(frame, points, DEFAULT_POINTS, (1080, 1920))


aruco_dict, params, points = _generate_aruco_dict()
writer = skvideo.io.FFmpegWriter("outputvideo.mp4")
cap = cv2.VideoCapture('test1.avi')
frame_order = 0
while cap.isOpened() and frame_order < 400:
    print("frame " + str(frame_order))
    if frame_order % 2:
        ret, frame = cap.read()
        try:
            gray = _denoise(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        except cv2.error:
            continue
        aruco_frame = _aruco_detection(gray, points)
        writer.writeFrame(_remove_shadow(aruco_frame))
    frame_order += 1
writer.close()
