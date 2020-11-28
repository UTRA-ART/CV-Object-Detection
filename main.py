import glob
import math
import time
import random

import cv2
import numpy

import image_tools
import cascade

# image_names = glob.glob("Stop Sign Dataset/data/*.jpg")

cap = cv2.VideoCapture("3.mp4")
# out = cv2.VideoWriter('output.mp4', 0x7634706d, 15, (800,450))


times = []
timeElasped = 0


fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
spf = 1 / fps
print(spf)


num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = num_frame / fps
print(duration)



frameCount = 0;
while(frameCount * fps < num_frame):

    start = time.time()

    cap.set(1, int( frameCount * fps ) );

    ret, img = cap.read()
    if not ret:
        break

    img = image_tools.scale(img, 800)
    cascade.identify_signs(img)

    times += [1]
    frameCount += 1

    # out.write(img)
    cv2.imshow("Temp", img)

    key = cv2.waitKey(1000)#pauses for 3 seconds before fetching next image

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('n'):
        break



print("Frame rate", round(1/(duration/len(times))),"Hz")

cap.release()
cv2.destroyAllWindows()
