from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

FPS = 24

base_options = python.BaseOptions(model_asset_path='MediaPipeEx/pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO)

detector = vision.PoseLandmarker.create_from_options(options)

# Use OpenCV’s VideoCapture to start capturing from the webcam.
cam = cv2.VideoCapture(0)
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("MediaPipeEx/live_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))

# Create a loop to read the latest frame from the camera using VideoCapture#read()
frame_idx = 0
while True:
    ret, frame = cam.read()

    # Convert BGR (OpenCV) to RGB
    numpy_frame_from_opencv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
        
    # Run pose detection with timestamp in milliseconds
    frame_timestamp_ms = int((frame_idx / FPS) * 1000)

    pose_landmarker_result = detector.detect_for_video(mp_image, frame_timestamp_ms)    
    annotated_frame = draw_landmarks_on_image(numpy_frame_from_opencv, pose_landmarker_result)

    # Convert back to BGR and write to output
    bgr_annotated = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    out.write(bgr_annotated)

    # Display the captured frame
    cv2.imshow('Camera', bgr_annotated)

    frame_idx += 1
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

    
# Cleanup
cam.release()
out.release()
cv2.destroyAllWindows()