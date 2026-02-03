import cv2
import numpy as np

def createSideBySide(userFile):
    # Open the two videos
    instructorVideo = cv2.VideoCapture("webapp-backend\HH1Annotated.mp4")
    studentVideo = cv2.VideoCapture(userFile)

    # Get left video's frame size and rate
    frameWidth = int(instructorVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(instructorVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(instructorVideo.get(cv2.CAP_PROP_FPS))

    # sframeWidth = int(studentVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
    # sframeHeight = int(studentVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # sfps = int(studentVideo.get(cv2.CAP_PROP_FPS))

    fontX = frameWidth - 300
    fontY = frameHeight - 100

    outputVideoFilePath = 'webapp-backend/FINALoutput.mp4'
    # Create output video
    outputVideoFile = cv2.VideoWriter(outputVideoFilePath, cv2.VideoWriter_fourcc(*"avc1"), fps, (frameWidth * 2, frameHeight))

    font = cv2.FONT_HERSHEY_SIMPLEX
    frameNum = 0

    while True:
        frameNum += 1
        success1, frame1 = instructorVideo.read()
        success2, frame2 = studentVideo.read()

        if not success1 or not success2:
            break

        frame2 = cv2.resize(frame2, (frameWidth, frameHeight))

        canvas = np.zeros((frameHeight, frameWidth*2, 3), dtype=np.uint8)
        
        canvas[:, :frameWidth] = frame1
        canvas[:, frameWidth:] = frame2 
        
        if frameNum < 100:
            cv2.putText(canvas, 'ACCURACY: 100%', (fontX, fontY), font, 2, (38, 182, 103), 4, cv2.LINE_4)
            cv2.putText(canvas, 'Good job!', (fontX, fontY+50), font, 1.5, (255, 255, 255), 2, cv2.LINE_4)

        else:
            cv2.putText(canvas, 'ACCURACY: 0%', (fontX, fontY), font, 2, (177, 55, 55), 4, cv2.LINE_4)
            cv2.putText(canvas, 'Needs more practice.', (fontX, fontY+50), font, 1.5, (255, 255, 255), 2, cv2.LINE_4)

        cv2.putText(canvas, 'Demo', (100, fontY+50), font, 1.5, (255, 255, 255), 2, cv2.LINE_4)

        outputVideoFile.write(canvas)

    instructorVideo.release()
    studentVideo.release()
    outputVideoFile.release()
    print("Side by side video created.")
    print(outputVideoFilePath)
    return 'FINALoutput.mp4'
