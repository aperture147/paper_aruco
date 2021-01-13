from processor import BaseProcessor
import cv2


class OutputVideoProcessor(BaseProcessor):

    def __init__(self, src_queue):
        super().__init__(src_queue)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi', fourcc, 20, (1080, 1920))

    def run(self) -> None:
        print('Output video processor started')
        while True:
            frame_tuple = self.src_queue.get()
            if frame_tuple[1].shape == (1, 1):
                break
            self.out.write(frame_tuple[1])
        print('Output video processor finished')
        self.out.release()
