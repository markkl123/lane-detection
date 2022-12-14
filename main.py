import cv2
import numpy as np

# Set the slice of the frame to use for lane detection
# (in percentage of the frame width and height)
Slice_width = 0,100
Slice_high = 62

# Keep track of the detected lanes in the last frame
last_lanes = []
last_lanes_arr = []

# Initialize flags for left and right lanes changes
left_flag = [False,0]
right_flag = [False,0]

def preprocess_frame(frame):
    # Get the dimensions of the frame
    im_size = frame.shape[:2]

    # Crop the frame to the region of interest (i.e. the road)
    # The region is defined by the Slice_width and Slice_high variables
    # which specify the percentage of the frame width and height to use
    cropped_image = frame[(Slice_high * im_size[0]) // 100:, (Slice_width[0]*im_size[1] // 100): (Slice_width[1]*im_size[1] //100), :]

    # apply a Gaussian blur to the frame
    blur = cv2.GaussianBlur(cropped_image, (5, 5), 0)

    # apply Canny edge detection to the blurred frame
    canny = cv2.Canny(blur, 120, 200)

    # Use dilation to strengthen the detected edges
    kernel = np.ones((3, 1))
    canny = cv2.dilate(canny,kernel)

    # create a mask to remove everything except the region of interest (i.e. the road)
    mask = np.zeros_like(canny)
    ignore_mask_color = 255
    imshape = canny.shape
    vertices = np.array([[(0,imshape[0]),(280, 5), (570, 5), (imshape[1],imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    res_frames = cv2.bitwise_and(canny, mask)

    return res_frames,cropped_image

def line_filter(lines_origin):
    global last_lanes

    # Initialize an empty list to store the filtered lines
    res = []

    # Extract the line parameters (rho, theta) from the lines
    lines = lines_origin[:, 0, :]

    for line in lines:

        # If two lane lines have already been found, break out of the loop
        if len(res)>1:
            break

        # Apply filters to the line to determine if it is a lane line
        # Filter 1: The slope of the line must be within a certain range
        # Filter 2: The distance of the line from the previous frame's detected lines must be within a certain range
        # Filter 3: The distance between the line and the other lines that have passed the filters must be within a certain range
        if (0<line[1]< 1.15 or 2.05 <line[1]<np.pi) and (np.array(last_lanes).shape[0] < 1 or (np.any(np.abs(last_lanes[:,1]-line[1])<0.01) and np.any(np.abs(last_lanes[:,0]-line[0])<6))) and \
                (len(res) == 0 or
                np.all(np.abs(line[0] - np.array(res)[:, 0])> 200) and np.all(np.abs(line[0] - np.array(res)[:, 0])< 650)):
                res.append(line)

    # If there are more than 1 line in the result or there were no previous
    # frame's detected lines, set the last_lanes variable to the detected lines
    if len(res)>1 or np.array(last_lanes).shape[0] == 0:
        last_lanes = np.array(res[:2])

    # Otherwise, set last_lanes to an empty array and try detecting lines again
    else:
        last_lanes = []
        return(line_filter(lines_origin))

    return np.array(res[:2])

def Draw_area(img,lines):
    # Get the dimensions of the image
    h, w = img.shape[:2]

    # Get the parameters of the two detected lines
    r1,t1 = lines[0]
    r2, t2 = lines[1]

    # Generate x values for the lines
    x = np.arange(w)[::-1]

    # Define the equations of the two lines using the detected line parameters
    line1 = lambda x: (r1-x*np.cos(t1))/np.sin(t1)
    line2 = lambda x: (r2-x*np.cos(t2))/np.sin(t2)

    # Compute the y values of the lines at each x value
    y1 = line1(x)
    y2 = line2(x)

    # Filter out points that are outside the bounds of the image
    points1 = np.array([[[xi, yi]] for xi, yi in zip(x, y1) if (0 <= xi < w and 0 <= yi < h)]).astype(np.int32)
    points2 = np.array([[[xi, yi]] for xi, yi in zip(x, y2) if (0 <= xi < w and 0 <= yi < h)]).astype(np.int32)

    # Create an array of the points of the polygon representing the area between the lines
    points = np.concatenate((points1, points2))

    # Draw the polygon on the original image
    drawn_image = img.copy()
    cv2.fillPoly(drawn_image, [points], color=[0, 0, 220])
    points = np.concatenate((points1, points2[::-1]))
    cv2.fillPoly(drawn_image, [points], color=[0, 0, 220])

    # Transparency factor
    alpha = 0.5
    # Make transparent
    drawn_image = cv2.addWeighted(drawn_image, alpha, img, 1 - alpha, 0)

    return drawn_image

# Detects lane lines in a frame of a video.
#Returns: An image with the detected lane lines drawn on it
def detect_lane(frame,cropped_im,origin_im):
    global last_lanes_arr
    global left_flag
    global right_flag

    # Set step sizes for Hough Transform and the thresh hold
    r_step, t_step, TH = 1, np.pi/180, 17

    # Use Hough transform to detect lines in the frame
    lines = cv2.HoughLines(frame, r_step, t_step, TH)

    if lines is not None and lines.shape[0] > 1:
        # Filter the detected lines using the line_filter function
        lines = line_filter(lines)

    if lines is None or lines.shape[0] < 2:
        # If no lines were detected or only one line was detected,
        # use the last detected lines instead
        lines = last_lanes_arr[-1]

    if lines is not None and lines.size > 2:

        # For each line in the set of detected lines
        for r_t in lines:

            # Get the line's parameters in polar coordinates (rho, theta)
            rho = r_t[0]
            theta = r_t[1]

            # Convert the line's polar coordinates to Cartesian coordinates
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))

            # Draw the line on the cropped image
            cv2.line(cropped_im, (x1, y1), (x2, y2), (0, 0, 220), thickness=5)

        if len(last_lanes_arr) > 12:
            # Calculate the change in the detected lines over the last few frames

            change = []
            change.append((np.abs(lines[0][0])+ np.abs(lines[1][0])) - (np.abs(last_lanes_arr[-1][0][0]) + np.abs(last_lanes_arr[-1][1][0])))

            for i in range(1,12):
                change.append((np.abs(last_lanes_arr[-i][0][0]) + np.abs(last_lanes_arr[-i][1][0])) - (np.abs(last_lanes_arr[-i-1][0][0]) + np.abs(last_lanes_arr[-i-1][1][0])))
            change = np.array(change)

            # If a left lane change is detected
            if not right_flag[0] and (left_flag[0] or sum(change > 2)>9):
                if not left_flag[0]:
                    left_flag[0] = True

                else:
                    left_flag[1] += 1
                    if left_flag[1] > 100:
                        left_flag = [False,0]

                # Write a message indicating a left lane change on the image
                cv2.putText(origin_im, "LEFT LANE CHANGE", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

            # If a right lane change is detected
            if not left_flag[0] and (right_flag[0] or sum(change < -2)>9):
                if not right_flag[0]:
                    right_flag[0] = True
                else:
                    right_flag[1] += 1
                    if right_flag[1] > 100:
                        right_flag = [False,0]

                # Write a message indicating a right lane change on the image
                cv2.putText(origin_im, "RIGHT LANE CHANGE", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        if lines is not None and lines.shape[0] > 1:
            # Add the detected lines to the list of last detected lines
            last_lanes_arr.append(lines)

        # Draw on the frame the area between the detected lanes
        cropped_im = Draw_area(cropped_im,lines)

    # Get the size of the original frame
    im_size = origin_im.shape[:2]

    # Overlay the detected lanes onto the original frame
    origin_im[(Slice_high * im_size[0]) // 100:, (Slice_width[0] * im_size[1] // 100): (Slice_width[1] * im_size[1] // 100), :] = cropped_im
    res_frames = origin_im

    return res_frames

def find_lane(frame):
    # Preprocess the frame to isolate the lane lines
    preprocessed_frame, cropped_image = preprocess_frame(frame)

    # Detect the lane lines in the preprocessed frame
    return detect_lane(preprocessed_frame, cropped_image, frame)


def read_frames_from_video(video_path, num_seconds=120):

    # Initialize an empty list to store the frames
    frames = list()

    # Use OpenCV to open the video and get the video capture
    capture = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    # Get the total number of frames in the video
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the total number of frames to read from the video
    # Use the minimum of the requested number of seconds and the total number of frames
    num_frames = min(num_seconds * fps, total_frames)

    # Recalculate the number of seconds based on the number of frames that will be read
    num_seconds = num_frames / fps

    # Print a message indicating that the frames are being read from the video
    print(f"Start reading {num_frames} frames ({num_seconds:.2f} seconds) from {video_path} ({fps} fps)")

    # Read the specified number of frames from the video, one frame at a time
    for i in range(num_frames):
        # Read a frame from the video
        image = capture.read()[1]
        # Append the frame to the frames list
        frames.append(image)

    # Release the video capture
    capture.release()

    # Print a message indicating that all frames have been read successfully
    print(f"Successfully read {len(frames)} frames from {video_path}")

    # Return the frames and fps as output
    return frames, fps


def save_frames_to_video(frames, fps):

    # Define the path where the video file will be saved
    video_path = fr'lanes.mp4'

    # Define the codec to be used for the video file
    codec = cv2.VideoWriter_fourcc(*"mp4v")

    # Get the dimensions of the frames
    h, w = frames[0].shape[:2]

    # Set the video to be written in color
    color = True

    # Create a VideoWriter object
    writer = cv2.VideoWriter(video_path, codec, fps, (w, h), color)

    # Print a message to the console indicating that the frames are being written to the video file
    print(f"Writing {len(frames)} frames to {video_path} ({fps} fps)")

    # Write each frame to the video file
    for frame in frames:
        writer.write(frame)

    # Release the writer object
    writer.release()


if __name__ == '__main__':

    # Path to the input video file
    input_video_path = r'video1.mp4'

    # Read the frames from the input video
    raw_frames, fps = read_frames_from_video(input_video_path)

    # Apply the lane finding algorithm to each frame
    final_frames = [find_lane(frame) for frame in raw_frames]

    # Save the processed frames to a new video file
    save_frames_to_video(final_frames, fps)