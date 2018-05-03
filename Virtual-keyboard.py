#   ROBT 310 - Final Project - Virtual Keyboard
#   Team: Daryn Kalym, Alibek Manabayev, Alexey Muryshkin
#   Date: April 27, 2018

#   modules
import cv2 as cv
import numpy as np

from KeyboardLayout import identify_keyboard, transform_image, make_vertical_disparity, make_horizontal_disparity

#   global variables
sep_dist = 10  # cm
focal_len = 0.367  # cm
p_window_up = "Processed Video Streaming UP"
p_window_down = "Processed Video Streaming DOWN"
b_window_name_up = "Background Image UP"
b_window_name_down = "Background Image DOWN"
o_window_name_stereo = "Disparity"


def get_video_capture(idd):
    if isinstance(idd, int):
        cap = cv.VideoCapture(idd)
    else:
        cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print('Cannot initialize video capture with id {}'.format(idd))
        # sys.exit(-1)

    return cap



def noise_filtering(frame):
    res = np.uint8(frame)
    # res = gray_image = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
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


def setup_windows():
    cv.namedWindow(p_window_up, flags=cv.WINDOW_NORMAL)
    # cv.moveWindow(p_window_name_up, 0, 0)

    cv.namedWindow(p_window_down, flags=cv.WINDOW_NORMAL)
    # cv.moveWindow(p_window_name_down, background_down.shape[1], 0)

    cv.namedWindow(b_window_name_up, flags=cv.WINDOW_NORMAL)
    # cv.moveWindow(b_window_name_up, 0, 0)

    cv.namedWindow(b_window_name_down, flags=cv.WINDOW_NORMAL)
    # cv.moveWindow(b_window_name_down, background_up.shape[1] // 3, 0)

    cv.namedWindow(o_window_name_stereo, flags=cv.WINDOW_NORMAL)
    # cv.moveWindow(o_window_name_stereo, background_down.shape[1], 0)


class VK:

    def __init__(self, id_up, id_down) -> None:
        super().__init__()
        self.camera_up = get_video_capture(id_up)
        self.camera_down = get_video_capture(id_down)
        self.new_shape = (720, 1280)
        self.points_up = None
        self.points_down = None
        self.background_up = None
        self.background_down = None
        self.col_index_matrix_y1 = None
        self.col_index_matrix_y2 = None
        self.col_index_matrix_x1 = None
        self.col_index_matrix_x2 = None
        setup_windows()

    def setup_background(self):
        # background reading
        ret, background_up = self.camera_up.read()
        if not ret:
            return
        print("Upper image")
        layout_up, points_up, background_up = identify_keyboard(background_up)
        # if points_up is None or len(background_up) != 4:
        #     return

        print("Lower image")
        ret, background_down = self.camera_down.read()
        if not ret:
            return
        layout_down, points_down, background_down = identify_keyboard(background_down)
        # if points_down is None or len(background_down) != 4:
        #     return

        self.points_up = points_up
        self.points_down = points_down

        if self.points_up is not None and self.points_down is not None and len(self.points_up) == 4 and len(
                self.points_down) == 4:
            background_up = transform_image(background_up, points_up)
            background_down = transform_image(background_down, points_down)

        background_up = noise_filtering(np.flip(np.array(background_up, dtype=np.uint8), axis=1))
        background_down = noise_filtering(np.flip(np.array(background_down, dtype=np.uint8), axis=1))

        new_shape = tuple(np.minimum(background_up.shape, background_down.shape))
        background_up = cv.resize(background_up, new_shape[::-1])
        background_down = cv.resize(background_down, new_shape[::-1])
        cv.imshow(b_window_name_up, background_up)
        cv.imshow(b_window_name_down, background_down)
        cv.resizeWindow(b_window_name_up, background_up.shape[1] // 3, background_up.shape[0] // 3)
        cv.resizeWindow(b_window_name_down, background_down.shape[1] // 3, background_down.shape[0] // 3)

        self.background_up = background_up
        self.background_down = background_down
        self.col_index_matrix_y1 = np.array(
            [[j for j in range(background_up.shape[1])] for i in range(background_up.shape[0])])
        self.col_index_matrix_x1 = np.array(
            [[i for j in range(background_up.shape[1])] for i in range(background_up.shape[0])])

        self.col_index_matrix_y2 = np.array(
            [[j for j in range(background_down.shape[1])] for i in range(background_down.shape[0])])
        self.col_index_matrix_x2 = np.array(
            [[i for j in range(background_down.shape[1])] for i in range(background_down.shape[0])])

    def display_video_real_time(self):
        cv.waitKey(1)
        self.setup_background()

        while True:
            ret, frame_up = self.camera_up.read()
            if not ret:
                break

            ret, frame_down = self.camera_down.read()
            if not ret:
                break

            key = cv.waitKey(30)
            c = chr(key & 255)
            if c in ['q', 'Q', chr(27)]:
                break
            elif c in ['b', 'B']:
                self.setup_background()

            if self.points_up is None or self.points_down is None or len(self.points_up) != 4 or len(
                    self.points_down) != 4:
                # print("Some error occurred")
                continue

            frame_up = transform_image(frame_up, self.points_up)
            frame_down = transform_image(frame_down, self.points_down)

            frame_up = cv.cvtColor(frame_up, cv.COLOR_RGB2GRAY)
            frame_down = cv.cvtColor(frame_down, cv.COLOR_RGB2GRAY)

            frame_up = cv.resize(frame_up, self.new_shape[::-1])
            frame_down = cv.resize(frame_down, self.new_shape[::-1])

            frame_up = np.flip(np.array(frame_up, dtype=np.uint8), axis=1)
            frame_down = np.flip(np.array(frame_down, dtype=np.uint8), axis=1)

            cv.imshow(p_window_up, np.uint8(frame_up))
            cv.imshow(p_window_down, np.uint8(frame_down))

            stereo = make_vertical_disparity(frame_up, frame_down)
            cv.imshow(o_window_name_stereo, stereo)

            cv.imwrite("images/frame_up.png", frame_up)
            cv.imwrite("images/frame_down.png", frame_down)
        self.end_session()

    def end_session(self):
        self.camera_up.release()
        self.camera_down.release()
        cv.destroyAllWindows()


def main():
    dev_id_up = 0  # int(input('Please enter the id of the opened video capturing device UP:\n'))
    dev_id_down = 1  # int(input('Please enter the id of the opened video capturing device DOWN:\n'))

    vk = VK(dev_id_up, dev_id_down)
    vk.display_video_real_time()


if __name__ == '__main__':
    main()
