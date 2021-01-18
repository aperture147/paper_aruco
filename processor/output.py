from processor import BaseProcessor
import skvideo.io


class OutputVideoProcessor(BaseProcessor):

    def __init__(self, src_queue):
        super().__init__(src_queue)
        self.writer = skvideo.io.FFmpegWriter("outputvideo.mp4")

    def run(self) -> None:
        print('Output video processor started')
        while True:
            frame_tuple = self._src_receiver.recv()
            if frame_tuple[1].shape == (1, 1):
                print('Received dead frame')
                break
            self.writer.writeFrame(frame_tuple[1])
        self.writer.close()
        print('Output video processor finished')
