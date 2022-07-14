import cv2
import os
cam = cv2.VideoCapture(cv2.CAP_V4L2)

cv2.namedWindow("test")

img_counter = 0
# folderpath = 'D:/Code/Work/Technology Mission GIFT/anthroprometry-giftbbsr/CapturedImages'
folderpath = "CapturedImages/"
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        # cv2.imwrite(img_name, frame)
        # cv2.imwrite(os.path.join(folderpath, img_name), frame)
        cv2.imwrite(str(folderpath)+img_name, frame)
        # print(frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
