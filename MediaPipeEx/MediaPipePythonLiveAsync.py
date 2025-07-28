from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading

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

FPS = 20

# Global shared frame storage
latest_annotated_frame = None
frame_lock = threading.Lock()

PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult

# # Create a pose landmarker instance with the live stream mode:
# def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     print('pose landmarker result: {}'.format(result))

# Callback from MediaPipe
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_annotated_frame
    annotated = draw_landmarks_on_image(output_image.numpy_view(), result)

    with frame_lock:
        latest_annotated_frame = annotated


base_options = python.BaseOptions(model_asset_path='MediaPipeEx/pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=print_result)

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
    cv2.imshow("Camera", frame)


    # Convert BGR (OpenCV) to RGB
    numpy_frame_from_opencv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
        
    # Run pose detection with timestamp in milliseconds
    frame_timestamp_ms = int((frame_idx / FPS) * 1000)
    detector.detect_async(mp_image, frame_timestamp_ms)

    # Send live image data to perform pose landmarking.
    # The results are accessible via the `result_callback` provided in
    # the `PoseLandmarkerOptions` object.
    # The pose landmarker must be created with the live stream mode.
    with frame_lock:
        if latest_annotated_frame is not None:
            bgr_annotated = cv2.cvtColor(latest_annotated_frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_annotated)
            # cv2.imshow("Pose Landmarks", bgr_annotated)
  
    frame_idx += 1
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cam.release()
out.release()
cv2.destroyAllWindows()