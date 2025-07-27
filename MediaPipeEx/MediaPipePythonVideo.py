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

base_options = python.BaseOptions(model_asset_path='MediaPipeEx/pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    running_mode=vision.RunningMode.VIDEO)
detector = vision.PoseLandmarker.create_from_options(options)

# Use OpenCV’s VideoCapture to load the input video.
cap = cv2.VideoCapture("MediaPipeEx/video.mp4")

# Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
# You’ll need it to calculate the timestamp for each frame.
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
out = cv2.VideoWriter("MediaPipeEx/output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Loop through each frame in the video using VideoCapture#read()
frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR (OpenCV) to RGB
    numpy_frame_from_opencv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)

    # Run pose detection with timestamp in milliseconds
    frame_timestamp_ms = int((frame_idx / fps) * 1000)
    
    # Perform pose landmarking on the provided single image.
    # The pose landmarker must be created with the video mode.
    pose_landmarker_result  = detector.detect_for_video(mp_image, frame_timestamp_ms)

    # Draw landmarks
    annotated_frame = draw_landmarks_on_image(numpy_frame_from_opencv, pose_landmarker_result)

    # Convert back to BGR and write to output
    bgr_annotated = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    out.write(bgr_annotated)

    frame_idx += 1

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()