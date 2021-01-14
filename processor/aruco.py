import cv2
import time
import numpy as np

import processor

DEFAULT_POINTS = np.array([[0, 1920], [0, 0], [1080, 1920],  [1080, 0]])


class ArucoProcessor(processor.BaseProcessor):

    def __init__(self, src_queue):
        super().__init__(src_queue)
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

        self.aruco_dict = aruco_dict
        self.parameters = parameters
        self.points = DEFAULT_POINTS

    @staticmethod
    def _warp_frame(frame, src_points, dst_points, shape):
        homography_mat, status = cv2.findHomography(src_points, dst_points)
        return cv2.warpPerspective(frame, homography_mat, shape)

    def run(self) -> None:
        print('Aruco processor started')
        frame_order = 0
        while True:
            t = time.time()
            frame_tuple = self._src_receiver.recv()
            print(time.time() - t)
            if frame_tuple[1].shape == (1, 1):
                print('Received dead frame')
                self.sender.send(frame_tuple)
                break
            print("Aruco frame:", frame_tuple[0])
            frame_order = frame_tuple[0]
            frame = frame_tuple[1]

            locations, ids, rejected = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.parameters)

            # Just get the top four point. Not the best idea
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
                    self.points = new_points
                # if we got only 3 points, then we should assume that
                # the missing point is bottomright since almost everyone
                # are right handed
                elif new_points.shape[0] == 3:
                    if self.points.shape[0] == 4:
                        np.append(new_points, self.points[3])
                    else:
                        np.append(new_points,
                                  np.array([self.points.max(axis=0, initial=0), self.points.max(axis=1, initial=0)]))
                    self.points = new_points
                # there are no more then 2 points, then we will try to use the old point
                # and replace some of them with the new points
                elif self.points.shape[0] == 4:
                    for idx, point in enumerate(new_points):
                        min_idx = 0
                        for ref_idx, ref_point in enumerate(self.points):
                            dist = np.linalg.norm(point - ref_point)
                            if dist < ref_point[min_idx]:
                                min_idx = ref_idx
                        self.points[min_idx] = point
            # Edge case, no marker found.
            # There maybe no paper or the aruco is to hard to detect. Try to use the old points
            else:
                continue
            self.sender.send((frame_order, self._warp_frame(frame, self.points, DEFAULT_POINTS, (1080, 1920))))
            frame_order += 1
        self.sender((frame_order, np.zeros((1, 1))))
        print('Aruco processor finished')
