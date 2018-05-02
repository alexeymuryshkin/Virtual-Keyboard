import cv2 as cv

from KeyboardLayout import identify_keyboard, transform_image


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


def main():
    capture_test_image()
    # check_layout()


if __name__ == "__main__":
    main()
