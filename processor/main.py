import processor

if __name__ == '__main__':
    pre = processor.PreProcessingProcessor()
    aruco = processor.ArucoProcessor(pre.queue)
    output = processor.OutputVideoProcessor(aruco.queue)

    pre.start()
    aruco.start()
    output.start()

    input("Press Enter to stop...")