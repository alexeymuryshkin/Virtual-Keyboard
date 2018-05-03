import cv2 as cv

from KeyboardLayout import identify_keyboard, transform_image, make_vertical_disparity


def capture_test_image():
    cap = cv.VideoCapture(0)
    filename = "images/test.png"
    while True:
        ret, image = cap.read()
        if not ret:
            print("error occurred capturing image")
            return

        cv.imshow('windows', image)
        q = cv.waitKey(0)
        if q in ('w', 'W', ord('w'), ord('W')):
            if not cv.imwrite(filename, image):
                print("error occurred writing file")
            else:
                print("Image is written")
        else:
            print("No file write")

        if q in ('q', 'Q', ord('q'), ord('Q')):
            break
    cap.release()


def check_layout():
    filename = "images/test.png"
    image = cv.imread(filename)
    layout, points = identify_keyboard(image)
    print(layout)
    print(points)
    transformed = transform_image(image, points)
    cv.imshow("transformed image", transformed)
    # cv.imwrite("images/transformed.png", transformed)
    cv.waitKey(0)


def check_disparity():
    frame_up = cv.imread("images/frame_up.png", cv.IMREAD_GRAYSCALE)
    frame_down = cv.imread("images/frame_down.png", cv.IMREAD_GRAYSCALE)
    disparity = make_vertical_disparity(frame_up, frame_down)
    cv.imshow("disparity", disparity)
    cv.waitKey(0)


def main():
    # capture_test_image()
    # check_layout()
    check_disparity()


if __name__ == "__main__":
    main()
