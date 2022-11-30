def load_video_frames(filename):
    pass


def preprocess_frames(raw_frames):
    res_frames = list()
    # Ideas:
        # Crop image
        # Filter only certain color pixels
        # Dilate over lanes to connect broken lines into one lane (- - - -) -> (-------)
        # How to deal with shades (going under bridge)?
            # Normalize colors? 
            # Intensify dark color?
            # Contrast?

    return res_frames


def detect_lanes(raw_frames):
    res_frames = list()
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


def save_frames_to_video(frames):
    pass


if __name__ == '__main__':
    video_filename = "video.mp3"
    raw_frames = load_video_frames(video_filename)

    res_frames = preprocess_frames(raw_frames)
    final_frames = detect_lanes(res_frames)

    save_frames_to_video(final_frames)