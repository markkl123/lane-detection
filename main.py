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

    save_frames_to_video(final_frames, fps)