import pyzbar.pyzbar as pyzbar
import cv2 as cv
import numpy as np

from transform import four_point_transform


def transform_image(image, points):
    image = four_point_transform(image, points)
    # image = cv.rotate(image, cv.ROTATE_180)
    return image


def identify_keyboard(image) -> (str, np.array):
    image = downsample(image)
    objects = find_objects(image)
    points = []
    new_image = np.copy(image)
    data = None
    for cur in objects:
        if cur.type != 'QRCODE':
            continue
        if data is None:
            data = cur.data
        pt = midpoint(cur.rect)
        points.append(pt)
        new_image = cv.rectangle(new_image, (cur.rect.left, cur.rect.top),
                                 (cur.rect.left + cur.rect.width, cur.rect.top + cur.rect.height), (0, 0, 255))
    # cv.imshow('test', new_image)
    # cv.imwrite('images/qr_test.png', new_image)
    # cv.waitKey(0)
    return data, np.array(points), new_image


def downsample(image: np.ndarray):
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    k = 3
    image = np.right_shift(image, k)
    image = np.left_shift(image, k)
    return image


def midpoint(rect):
    x = rect.left + rect.width // 2
    y = rect.top + rect.height // 2
    return [x, y]


def find_objects(image):
    # Find barcodes and QR codes
    decoded_objects = pyzbar.decode(image)

    # Print results
    for obj in decoded_objects:
        print('Type : ', obj.type)
        print('Data : ', obj.data, '\n')

    return decoded_objects


def make_horizontal_disparity(img_l, img_r):
    stereo = cv.StereoBM_create(numDisparities=16, blockSize=25)
    disparity = stereo.compute(img_l, img_r)    # type: np.ndarray
    disparity = disparity.astype(float)
    mx = disparity.max()
    mn = disparity.min()
    disparity -= mn
    disparity /= mx - mn
    return disparity


def make_vertical_disparity(img_up, img_down):
    img_up = cv.flip(img_up, cv.ROTATE_90_CLOCKWISE)
    img_down = cv.flip(img_down, cv.ROTATE_90_CLOCKWISE)
    disparity = make_horizontal_disparity(img_up, img_down)
    disparity = cv.flip(disparity, cv.ROTATE_90_COUNTERCLOCKWISE)
    return disparity
