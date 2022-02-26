import numpy as np
import cv2

def draw_the_lines(image, lines):
    # create a distinct image [0, 255]
    lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=3)

    # finally we have to merge
    image_with_lines = cv2.addWeighted(image, 0.8, lines_image, 1, 0.0)
    return image_with_lines
def region_of_interest(image, region_points):

    # we are going to replace pixels with 0 (black) - the regions
    mask = np.zeros_like(image)

    # the region that we are interested in is the lower triangle - 255
    cv2.fillPoly(mask, region_points, 255)

    # we have to use the mask: we want to keep the regions of original image where
    # the mask has white colored pixels
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def get_lanes(image):
    h, w = image.shape[0], image.shape[1]

    gray_images = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    canny_image = cv2.Canny(gray_images, 100, 120)

    region_of_interest_vertices = [
        (0, h), (w / 2, h * 0.65), (w, h)
    ]

    # we can get rid of the un-relevant part of the image
    # we just keep the lower triangle region
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

    # use the line detection
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=50,
                            lines=np.array([]), minLineLength=40, maxLineGap=150)

    # draw the lines on the image
    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines

video = cv2.VideoCapture('lane_detection_video.mp4')

while video.isOpened():
    success, frame = video.read()

    frame = get_lanes(frame)

    if not success:
        break
    cv2.imshow('Lane Detection Video', frame)
    cv2.waitKey(50)

video.release()
cv2.destroyAllWindows()