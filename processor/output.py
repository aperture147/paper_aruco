from processor import BaseProcessor
import cv2


class OutputVideoProcessor(BaseProcessor):

    def __init__(self, src_queue):
        super().__init__(src_queue)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi', fourcc, 15, (1920, 1080))

    def run(self) -> None:
        print('Output video processor started')
        while True:
            frame_tuple = self.src_queue.get()
            if frame_tuple[1].shape == (1, 1):
                break
            self.out.write(cv2.cvtColor(frame_tuple[1], cv2.COLOR_GRAY2BGR))
        print('Output video processor finished')
        self.out.release()
