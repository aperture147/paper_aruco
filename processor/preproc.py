from processor import BaseProcessor
import cv2
import numpy as np


class PreProcessingProcessor(BaseProcessor):

    def __init__(self, video_path="test.avi"):
        self.video_path = video_path
        super().__init__(None)

    @staticmethod
    def _remove_shadow(frame):
        dilated_img = cv2.dilate(frame, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        return 255 - cv2.absdiff(frame, bg_img)

    @staticmethod
    def _denoise(frame):
        return cv2.fastNlMeansDenoising(frame, None)

    def run(self) -> None:
        print('Preprocessing processor started')
        cap = cv2.VideoCapture(self.video_path)
        frame_order = 0
        while cap.isOpened() and frame_order < 100:
            if frame_order % 2:
                ret, frame = cap.read()
                try:
                    gray = self._denoise(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    self.queue.put((frame_order, gray))
                except cv2.error:
                    continue
            frame_order += 1
        self.queue.put((frame_order, np.zeros((1, 1))))
        print('Preprocessing processor finished')
        cap.release()
