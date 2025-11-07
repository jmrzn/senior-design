from KeyFrameDetector.key_frame_detector import keyframeDetection

source_video = "MediaPipeEx\SAMPLE_Instructor.mp4"
output_dir = "MediaPipeEx\Instructor"
threshold = 0.4

keyframeDetection(
    source=source_video,
    dest=output_dir,
    Thres=threshold,
    plotMetrics=True,     
    verbose=True         
)