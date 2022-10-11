import cv2 as cv

def nothing(x):
    pass

# settings
track_win_name = "trackbar window"
pic_path = "E:/png/1.png"

# read picture
img = cv.imread(pic_path, 1)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.namedWindow(track_win_name,0)

# create trackbars for threshold change
cv.createTrackbar('thres_min', track_win_name, 0, 255, nothing)
cv.createTrackbar('thres_max', track_win_name, 255, 255, nothing)

# dynamic update
while(1):
    # get current positions of four trackbars
    thres_min = cv.getTrackbarPos('thres_min', track_win_name)
    thres_max = cv.getTrackbarPos('thres_max', track_win_name)
    ret, img_after_interaction = cv.threshold(img, thres_min, thres_max, cv.THRESH_BINARY)
    cv.imshow(track_win_name, img_after_interaction)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()