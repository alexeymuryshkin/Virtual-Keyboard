#   ROBT 310 - Final Project - Virtual Keyboard
#   Team: Daryn Kalym, Alibek Manabayev, Alexey Muryshkin
#   Date: April 27, 2018

#   modules
import cv2 as cv
import numpy as np

from KeyboardLayout import identify_keyboard, transform_image

#   global variables
sep_dist = 10  # cm
focal_len = 0.367  # cm


# frame1, frame2 = None, None
# background_up, background_down = None, None
# col_index_matrixY1, col_index_matrixY2 = None, None
# col_index_matrixX1, col_index_matrixX2 = None, None

#   functions

def make_disparity(img_l, img_r):
    stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(img_l, img_r)
    return disparity


# Noise Filtering
def noise_filtering(frame):
    res = np.uint8(frame)
    res = gray_image = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    res = cv.equalizeHist(res)
    res = med_filt_image = cv.medianBlur(res, 3)

    return res


def process_frame(frame, background):
    threshold = 50

    res = np.abs(np.int32(frame) - np.int32(background))

    mask = np.abs(np.int32(frame) - np.int32(background)) > threshold
    back_sub_image = np.zeros(frame.shape)
    back_sub_image[mask] = 255
    res = back_sub_image

    kernel = np.ones((7, 7), np.uint8)
    erosion = cv.erode(back_sub_image, kernel, iterations=1)
    kernel = np.ones((10, 10), np.uint8)
    res = dilation = cv.dilate(erosion, kernel, iterations=1)

    return np.uint8(res)


def get_center_y(img, col_index_matrix):
    mask = img == 255
    # n = np.sum(np.uint32(mask))
    # ySum = np.sum( col_index_matrix[mask] )

    # return int( round(ySum / n) ) if n else None
    return np.min(col_index_matrix[mask]) if mask.any() else None


def get_center_x(img, col_index_matrix):
    mask = img == 255
    # n = np.sum(np.uint32(mask))
    # xSum = np.sum( col_index_matrix[mask] )

    # return int( round(xSum / n) ) if n else None
    return np.min(col_index_matrix[mask]) if mask.any() else None


def display_video_real_time(camera_up, camera_down):
    global sep_dist, focal_len

    new_shape = []

    # background reading
    ret, background_up = camera_up.read()
    # background_up = noiseFiltering( np.flip( np.array(background_up, dtype=np.uint8), axis = 1 ) )
    layout_up, points_up, background_up = identify_keyboard(background_up)

    ret, background_down = camera_down.read()
    # background_down = noiseFiltering( np.flip( np.array(background_down, dtype=np.uint8), axis = 1 ) )
    layout_down, points_down, background_down = identify_keyboard(background_down)

    if points_up is not None and points_down is not None and len(points_up) == 4 and len(points_down) == 4:
        background_up = transform_image(background_up, points_up)
        background_down = transform_image(background_down, points_down)

        background_up = noise_filtering(np.flip(np.array(background_up, dtype=np.uint8), axis=1))
        background_down = noise_filtering(np.flip(np.array(background_down, dtype=np.uint8), axis=1))

        new_shape = tuple(np.minimum(background_up.shape, background_down.shape))
        background_up = cv.resize(background_up, new_shape[::-1])
        background_down = cv.resize(background_down, new_shape[::-1])

    col_index_matrix_y1 = np.array([[j for j in range(background_up.shape[1])] for i in range(background_up.shape[0])])
    col_index_matrix_x1 = np.array([[i for j in range(background_up.shape[1])] for i in range(background_up.shape[0])])
    col_index_matrix_y2 = np.array(
        [[j for j in range(background_down.shape[1])] for i in range(background_down.shape[0])])
    col_index_matrix_x2 = np.array(
        [[i for j in range(background_down.shape[1])] for i in range(background_down.shape[0])])

    # creating video streaming windows
    # o_window_name1 = "WebCam Video Streaming 1"
    # cv.namedWindow(o_window_name1, flags=cv.WINDOW_AUTOSIZE)
    # cv.moveWindow(o_window_name1, 0, 0)

    p_window_name_up = "Processed Video Streaming UP"
    cv.namedWindow(p_window_name_up, flags=cv.WINDOW_NORMAL)
    # cv.moveWindow(p_window_name_up, 0, 0)

    p_window_name_down = "Processed Video Streaming DOWN"
    cv.namedWindow(p_window_name_down, flags=cv.WINDOW_NORMAL)
    # cv.moveWindow(p_window_name_down, background_down.shape[1], 0)

    b_window_name_up = "Background Image UP"
    cv.namedWindow(b_window_name_up, flags=cv.WINDOW_NORMAL)
    # cv.moveWindow(b_window_name_up, 0, 0)
    cv.imshow(b_window_name_up, np.uint8(background_up))
    cv.resizeWindow(b_window_name_up, background_up.shape[1] // 3, background_up.shape[0] // 3)

    b_window_name_down = "Background Image DOWN"
    cv.namedWindow(b_window_name_down, flags=cv.WINDOW_NORMAL)
    # cv.moveWindow(b_window_name_down, background_up.shape[1] // 3, 0)
    cv.imshow(b_window_name_down, np.uint8(background_down))
    cv.resizeWindow(b_window_name_down, background_down.shape[1] // 3, background_down.shape[0] // 3)

    o_window_name_stero = "Disparity"
    cv.namedWindow(o_window_name_stero, flags=cv.WINDOW_NORMAL)
    # cv.moveWindow(o_window_name_stero, background_down.shape[1], 0)

    while True:
        ret, frame_up = camera_up.read()
        if not ret:
            break

        ret, frame_down = camera_down.read()
        if not ret:
            break

        key = cv.waitKey(30)
        c = chr(key & 255)
        if c in ['q', 'Q', chr(27)]:
            break
        elif c in ['b', 'B']:
            background_up = np.copy(frame_up)
            layout_up, points_up, background_up = identify_keyboard(background_up)

            background_down = np.copy(frame_down)
            layout_down, points_down, background_down = identify_keyboard(background_down)

            if points_up is not None and points_down is not None and len(points_up) == 4 and len(points_down) == 4:
                background_up = transform_image(background_up, points_up)
                background_down = transform_image(background_down, points_down)

                background_down = noise_filtering(np.flip(np.array(background_down, dtype=np.uint8), axis=1))
                background_up = noise_filtering(np.flip(np.array(background_up, dtype=np.uint8), axis=1))

                new_shape = tuple(np.minimum(background_up.shape, background_down.shape))
                background_up = cv.resize(background_up, new_shape[::-1])
                background_down = cv.resize(background_down, new_shape[::-1])

            cv.resizeWindow(b_window_name_up, background_up.shape[1] // 3, background_up.shape[0] // 3)
            # cv.moveWindow(b_window_name_up, 0, 0)

            cv.resizeWindow(b_window_name_down, background_down.shape[1] // 3, background_down.shape[0] // 3)
            # cv.moveWindow(b_window_name_down, background_up.shape[1], 0)

            cv.imshow(b_window_name_up, np.uint8(background_up))
            cv.imshow(b_window_name_down, np.uint8(background_down))

            col_index_matrix_y1 = np.array(
                [[j for j in range(background_up.shape[1])] for i in range(background_up.shape[0])])
            col_index_matrix_x1 = np.array(
                [[i for j in range(background_up.shape[1])] for i in range(background_up.shape[0])])

            col_index_matrix_y2 = np.array(
                [[j for j in range(background_down.shape[1])] for i in range(background_down.shape[0])])
            col_index_matrix_x2 = np.array(
                [[i for j in range(background_down.shape[1])] for i in range(background_down.shape[0])])

        # proc_img_up = np.copy(frame_up)
        # proc_img_down = np.copy(frame_down)

        if points_up is not None and points_down is not None and len(points_up) == 4 and len(points_down) == 4:
            frame_up = transform_image(frame_up, points_up)
            frame_down = transform_image(frame_down, points_down)

            frame_up = cv.resize(frame_up, new_shape[::-1])
            frame_down = cv.resize(frame_down, new_shape[::-1])

            frame_up = np.flip(np.array(frame_up, dtype=np.uint8), axis=1)
            frame_down = np.flip(np.array(frame_down, dtype=np.uint8), axis=1)

            proc_img_up = np.copy(frame_up)
            proc_img_down = np.copy(frame_down)

            frame_up = cv.cvtColor(frame_up, cv.COLOR_BGR2GRAY)
            frame_down = cv.cvtColor(frame_down, cv.COLOR_BGR2GRAY)

            cv.imshow(p_window_name_up, np.uint8(frame_up))
            cv.imshow(p_window_name_down, np.uint8(frame_down))

            proc_img_up = noise_filtering(proc_img_up)
            proc_img_down = noise_filtering(proc_img_down)

            proc_img_up = process_frame(proc_img_up, background_up)
            proc_img_down = process_frame(proc_img_down, background_down)
            # cv.imshow(p_window_name, np.uint8( proc_img ))

            center_y1 = get_center_y(proc_img_up, col_index_matrix_y1)
            center_y2 = get_center_y(proc_img_down, col_index_matrix_y2)
            center_x1 = get_center_x(proc_img_up, col_index_matrix_x1)
            center_x2 = get_center_x(proc_img_down, col_index_matrix_x2)

            if center_x1 is not None and center_x2 is not None:
                mask = proc_img_up != 255
                stereo = make_disparity(cv.rotate(frame_up, cv.ROTATE_90_CLOCKWISE),
                                        cv.rotate(frame_down, cv.ROTATE_90_CLOCKWISE))
                # stereo[mask] = 0
                # stereo_dif = np.abs(np.int32(make_disparity(frame_up, frame_down)) - np.int32(
                #     make_disparity(background_up, background_down)))
                # print(stereo_dif)
                cv.imshow(o_window_name_stero, stereo)

            if center_y2 is not None:
                proc_img_down[:, center_y2] = 255

            if center_y1 is not None:
                proc_img_up[:, center_y1] = 255

            if center_x2 is not None:
                proc_img_down[center_x2, :] = 255

            if center_x1 is not None:
                proc_img_up[center_x1, :] = 255
            # if center_y1 is not None and center_y2 is not None:
            # pass
            # print(sep_dist * focal_len / abs(center_y1 - center_y2) if abs(center_y1 - center_y2) > 0 else 0)

        # cv.imshow(p_window_name_up, np.uint8( proc_img_up ))
        # cv.imshow(p_window_name_down, np.uint8( proc_img_down ))

    camera_up.release()
    camera_down.release()
    cv.destroyAllWindows()


def get_video_capture(idd):
    if isinstance(idd, int):
        cap = cv.VideoCapture(idd)
    else:
        cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print('Cannot initialize video capture with id {}'.format(idd))
        # sys.exit(-1)

    return cap


# main program
def main():
    dev_id_up = int(input('Please enter the id of the opened video capturing device UP:\n'))
    camera_up = get_video_capture(dev_id_up)
    dev_id_down = int(input('Please enter the id of the opened video capturing device DOWN:\n'))
    camera_down = get_video_capture(dev_id_down)

    display_video_real_time(camera_up, camera_down)


if __name__ == '__main__':
    main()
