import cv2
import numpy as np

cap = cv2.VideoCapture(0)

frames = []
frame_bounds = ()
animation_mode = False
count = 0
frame_count = 0

def captureFrame(frame, bounds):
    x, y, w, h = bounds

    frame = frame[y:y+h, x:x+w]
    frames.append(frame)
    print("captured", len(frames))


while True:

    _, frame = cap.read()
    count += 1
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 150, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    key = cv2.waitKey(1)

    if key == ord('c'):
        mask = np.zeros(imgray.shape,  np.uint8)
        mask = cv2.drawContours(
            mask, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)
        #mask = cv2.bitwise_not(mask)
        #cv2.imshow("mask", mask)
        captureFrame(mask, frame_bounds)

    elif key == ord('a'):
        animation_mode = True

    elif key == 27:
        if animation_mode is False:
            for i in range(len(frames)):
                cv2.imwrite(f'frame-{i}.jpg', frames[i])
        break

    frame_bounds = (320-180, 240-120, 360, 240)

    if animation_mode is False:
        cv2.putText(frame, f'frame {len(frames)+1}', (320-180, 240-130),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=0, thickness=2)
        cv2.rectangle(frame, (320-180, 240-120),
                      (320+180, 240+120), 0, 3)
    
    else:
        frame = cv2.bitwise_not(np.zeros(frame.shape, np.uint8))

        if count%6==0:
            frame_count += 1
            if frame_count > len(frames)-1:
                frame_count = 0
            print(frame_count)

        h,w = frames[frame_count].shape
        y,x = h-(h//2),w-(w//2)
        to_add = cv2.cvtColor(frames[frame_count],cv2.COLOR_GRAY2BGR)
        frame[y:y+h,x:x+w] = to_add
        

    
    cv2.imshow('Frame', frame)


cap.release()
cv2.destroyAllWindows()
