# import moviepy.editor as mp
# clip = mp.VideoFileClip("fish_roi.avi")
# clip_resized = clip.resize(height=96, width=96) # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
# clip_resized.write_videofile("fish_roi_resized.avi")

# import cv2
# import numpy as np
#
# cap = cv2.VideoCapture(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/fish_roi.avi')
#
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('fish_roi_resized.avi', fourcc, 5, (96, 96))
#
# while True:
#     ret, frame = cap.read()
#     if ret == True:
#         b = cv2.resize(frame, (96, 96), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
#         out.write(b)
#     else:
#         break
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np

cap = cv2.VideoCapture(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/fish_roi_annotated.avi')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('fish_roi_resized.avi', fourcc, 5, (96, 96))

while True:
    ret, frame = cap.read()
    if ret == True:
        b = cv2.resize(frame, (96, 96), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        out.write(b)
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()