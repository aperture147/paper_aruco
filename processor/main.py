import processor

if __name__ == '__main__':
    pre = processor.PreProcessingProcessor()
    aruco = processor.ArucoProcessor(pre.receiver)
    output = processor.OutputVideoProcessor(aruco.receiver)

    pre.start()
    aruco.start()
    output.start()

    # input("Press Enter to stop...\n")
