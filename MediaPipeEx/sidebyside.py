import cv2
import numpy as np

# Open the two videos
leftVideo = cv2.VideoCapture('MediaPipeEx/video.mp4')
rightVideo = cv2.VideoCapture('MediaPipeEx/output_video.mp4')

# Get left video's frame size and rate
frameWidth = int(leftVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(leftVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(leftVideo.get(cv2.CAP_PROP_FPS))

fontX = frameWidth - 300
fontY = frameHeight - 100

# Create output video
out = cv2.VideoWriter('MediaPipeEx/sidebyside.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (frameWidth * 2, frameHeight))

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    success1, frame1 = leftVideo.read()
    success2, frame2 = rightVideo.read()

    if not success1 or not success2:
        break

    # frame2 = cv2.resize(frame2, (frameWidth, frameHeight))

    canvas = np.zeros((frameHeight, frameWidth*2, 3), dtype=np.uint8)
    
    canvas[:, :frameWidth] = frame1
    canvas[:, frameWidth:] = frame2 

    cv2.putText(canvas, 'ACCURACY: 100%', (fontX, fontY), font, 2, (38, 182, 103), 4, cv2.LINE_4)
    cv2.putText(canvas, 'Good job!', (fontX, fontY+50), font, 1.5, (255, 255, 255), 2, cv2.LINE_4)

    out.write(canvas)

leftVideo.release()
rightVideo.release()
out.release()