import cv2
from datetime import datetime


def read_frames_from_video(video_path, num_seconds=30):
    frames = list()
    capture = cv2.VideoCapture(video_path)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_seconds * fps > total_frames:
        print(f'Cannot read {num_seconds} seconds from video...')

    num_frames = min(num_seconds * fps, total_frames)
    num_seconds = num_frames / fps

    print(f"Start reading {num_frames} frames ({num_seconds:.2f} seconds) from {video_path} ({fps} fps)")

    for i in range(num_frames):
        image = capture.read()[1]
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        frames.append(image)
    capture.release()

    print(f"Successfully read {len(frames)} frames from {video_path}")

    return frames, fps
        


def preprocess_frames(raw_frames):
    res_frames = raw_frames
    # Ideas:
        # Crop image
        # Filter only certain color pixels
        # Dilate over lanes to connect broken lines into one lane (- - - -) -> (-------)
        # How to deal with shades (going under bridge)?
            # Normalize colors? 
            # Intensify dark color?
            # Contrast?

    return res_frames


def detect_lanes(preprocessed_frames):
    res_frames = preprocessed_frames
    # How to find best lines?
        # Canny
        # Hough Parabula ?? self implement (Omri) or use HoughLines (Mark)
    # How to detect lane transition?
        # transform lines from (x,y) to (rho, theta) and detect changes in rho (distance to origin)
            # Distance is decreasing -> LEFT
            # Distance is increasing -> RIGHT 
            # Use threshold to deal with noise
    # How to follow correct lanes during transition?
        # Compare to last frame
        # Lanes change will happen naturally when transition is complete
    return res_frames


def save_frames_to_video(frames, fps):
    video_path = fr'videos\output\lanes.mp4'
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames[0].shape[:2]
    color = True
    
    writer = cv2.VideoWriter(video_path, codec, fps, (w, h), color)

    print(f"Writing {len(frames)} frames to {video_path} ({fps} fps)")

    for frame in frames:
        writer.write(frame)
    writer.release()


if __name__ == '__main__':

    input_video_path = r'videos\input\video1.mp4'    
    raw_frames, fps = read_frames_from_video(input_video_path)

    preprocessed_frames = preprocess_frames(raw_frames)
    final_frames = detect_lanes(preprocessed_frames)

    save_frames_to_video(final_frames, fps)