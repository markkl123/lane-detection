import cv2
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import itertools

figsize = (10, 10)

Slice_width = 0,100
Slice_high = 62
last_lanes = []
print_flag = False
counter2 = 0
last_lanes_arr = []
left_flag = [False,0]
right_flag = [False,0]

def print_img(image,name="image"):
    if (print_flag):
        plt.figure(figsize=figsize)
        plt.imshow(image)
        plt.title(name)
        plt.show()


def preprocess_frame(raw_frames):
    im_size = raw_frames.shape[:2]
    cropped_image = raw_frames[(Slice_high * im_size[0]) // 100:, (Slice_width[0]*im_size[1] // 100): (Slice_width[1]*im_size[1] //100), :]
    print_img(cropped_image, "cropped_image")
    #filtered_image = cv2.inRange(cropped_image, np.array([180,180,180]),np.array([255,255,255]))
    #print_img(filtered_image, "filtered_image")
    # kernel = np.ones((5,2))
    # eroded_im = cv2.erode(filtered_image, kernel)
    # eroded_im = cv2.dilate(eroded_im,kernel)
    # print_img(eroded_im, "eroded_im")
    # res_frames = filtered_image
    #gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    # apply a Gaussian blur to the frame
    blur = cv2.GaussianBlur(cropped_image, (5, 5), 0)

    # apply Canny edge detection to the blurred frame
    canny = cv2.Canny(blur, 120, 200)
    print_img(canny, "canny")
    kernel = np.ones((3, 1))
    canny = cv2.dilate(canny,kernel)
    print_img(canny, "dilated_im")
    # create a mask to remove everything except the region of interest (i.e. the road)
    mask = np.zeros_like(canny)
    ignore_mask_color = 255
    imshape = canny.shape
    vertices = np.array([[(0,imshape[0]),(260, 5), (590, 5), (imshape[1],imshape[0])]], dtype=np.int32) #320 , 490
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    res_frames = cv2.bitwise_and(canny, mask)
    print_img(res_frames, "pre")
    # Ideas:
        # Crop image
        # Filter only certain color pixels
        # Dilate over lanes to connect broken lines into one lane (- - - -) -> (-------)
        # How to deal with shades (going under bridge)?
            # Normalize colors?
            # Intensify dark color?
            # Contrast?

    return res_frames,cropped_image

def line_filter(lines_origin):
    global last_lanes
    global counter2
    res = []
    lines = lines_origin[:, 0, :]

    for line in lines:
        if len(res)>1:
            break
        if (0<line[1]< 1.15 or 2.05 <line[1]<np.pi) and (np.array(last_lanes).shape[0] < 1 or (np.any(np.abs(last_lanes[:,1]-line[1])<0.01) and np.any(np.abs(last_lanes[:,0]-line[0])<6))) and \
                (len(res) == 0 or
                np.all(np.abs(line[0] - np.array(res)[:, 0])> 200) and np.all(np.abs(line[0] - np.array(res)[:, 0])< 650)):
                res.append(line)
    if len(res)>1 or np.array(last_lanes).shape[0] == 0:
        last_lanes = np.array(res[:2])
    else:
        last_lanes = []

        counter2 += 1
        print("counter2 = " + str(counter2))
        return(line_filter(lines_origin))

    return np.array(res[:2])

def Draw_area(img,lines):
    h, w = img.shape[:2]
    r1,t1 = lines[0]
    r2, t2 = lines[1]

    x = np.arange(w)[::-1]
    line1 = lambda x: (r1-x*np.cos(t1))/np.sin(t1)
    line2 = lambda x: (r2-x*np.cos(t2))/np.sin(t2)
    y1 = line1(x)
    y2 = line2(x)

    points1 = np.array([[[xi, yi]] for xi, yi in zip(x, y1) if (0 <= xi < w and 0 <= yi < h)]).astype(np.int32)
    points2 = np.array([[[xi, yi]] for xi, yi in zip(x, y2) if (0 <= xi < w and 0 <= yi < h)]).astype(np.int32)

    points = np.concatenate((points1, points2))
    drawn_image = img.copy()
    cv2.fillPoly(drawn_image, [points], color=[180, 180, 180])
    points = np.concatenate((points1, points2[::-1]))
    cv2.fillPoly(drawn_image, [points], color=[180, 180, 180])
    print_img(drawn_image,"drawn_image")
    return drawn_image

def detect_lane(raw_frames,cropped_im,origin_im):
    global last_lanes_arr
    global left_flag
    global right_flag
    r_step, t_step, TH = 1, np.pi/180, 17
    lines = cv2.HoughLines(raw_frames, r_step, t_step, TH)
    prev = np.array(last_lanes).copy()
    if lines is not None and lines.shape[0] > 1:
        lines = line_filter(lines)
        # lines = lines[:, 0, :]
    if lines is None or lines.shape[0] < 2:
        lines = last_lanes_arr[-1]
    if lines is not None and lines.size > 2:

        # lines = lines[:, 0, :]
        for r_t in lines:
            rho = r_t[0]
            theta = r_t[1]

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))

            cv2.line(cropped_im, (x1, y1), (x2, y2), (0, 0, 255), thickness=5)

        if len(last_lanes_arr) > 12:
            change = []
            change.append((np.abs(lines[0][0])+ np.abs(lines[1][0])) - (np.abs(last_lanes_arr[-1][0][0]) + np.abs(last_lanes_arr[-1][1][0])))

            for i in range(1,12):
                change.append((np.abs(last_lanes_arr[-i][0][0]) + np.abs(last_lanes_arr[-i][1][0])) - (np.abs(last_lanes_arr[-i-1][0][0]) + np.abs(last_lanes_arr[-i-1][1][0])))
            change = np.array(change)
            if not right_flag[0] and (left_flag[0] or sum(change > 1.7)>9):
                if not left_flag[0]:
                    left_flag[0] = True
                else:
                    left_flag[1] += 1
                    if left_flag[1] > 100:
                        left_flag = [False,0]
                cv2.putText(cropped_im, "LEFT LANE CHANGE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if not left_flag[0] and (right_flag[0] or sum(change < -1.7)>9):
                if not right_flag[0]:
                    right_flag[0] = True
                else:
                    right_flag[1] += 1
                    if right_flag[1] > 100:
                        right_flag = [False,0]
                cv2.putText(cropped_im, "RIGHT LANE CHANGE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if lines is not None and lines.shape[0] > 1:
            last_lanes_arr.append(lines)
        # print_img(res, "res")
        cropped_im = Draw_area(cropped_im,lines)

    im_size = origin_im.shape[:2]
    origin_im[(Slice_high * im_size[0]) // 100:, (Slice_width[0] * im_size[1] // 100): (Slice_width[1] * im_size[1] // 100), :] = cropped_im
    print_img(origin_im, "final result")
    res_frames = origin_im
    # How to find best lines?
        # Hough Parabula ?? self implement (Omri) or use HoughLines (Mark)
    # How to detect lane transition?
        # transform lines from (x,y) to (rho, theta) and detect changes in rho (distance to origin)
            # Distance is decreasing -> LEFT
            # Distance is increasing -> RIGHT
            # Use threshold to deal with noise USED!!!
    # How to follow correct lanes during transition?
        # Compare to last frame
        # Lanes change will happen naturally when transition is complete
    return res_frames

def find_lane(frame):
    preprocessed_frame, cropped_image = preprocess_frame(frame)
    return detect_lane(preprocessed_frame, cropped_image, frame)


def read_frames_from_video(video_path, num_seconds=60):
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
        frames.append(image)
    capture.release()

    print(f"Successfully read {len(frames)} frames from {video_path}")

    return frames, fps


def save_frames_to_video(frames, fps):
    video_path = fr'lanes.mp4'
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames[0].shape[:2]
    color = True
    
    writer = cv2.VideoWriter(video_path, codec, fps, (w, h), color)

    print(f"Writing {len(frames)} frames to {video_path} ({fps} fps)")

    for frame in frames:
        writer.write(frame)
    writer.release()


if __name__ == '__main__':

    input_video_path = r'video1.mp4'
    raw_frames, fps = read_frames_from_video(input_video_path)
    final_frames = [find_lane(frame) for frame in raw_frames]
    save_frames_to_video(final_frames, fps)

print(counter2)