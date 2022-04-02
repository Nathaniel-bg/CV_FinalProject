from cv2 import getTrackbarPos
from sklearn.cluster import MeanShift, estimate_bandwidth
import cv2
import numpy as np

def getColorMask(frame):

    #setting the HSV values for the color detection
    lowerBound = np.array([0,170,80])
    upperBound = np.array([10,255,255])

    #Creating a mask for the specific color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lowerBound, upperBound)

def nothing(x):
    pass

cv2.namedWindow('image')
cap = cv2.VideoCapture(0)


while cap.isOpened():

    ret, frame = cap.read()
    mask_frame = getColorMask(frame)

    mask_frame = cv2.medianBlur(mask_frame,5)

    cnts = cv2.findContours(mask_frame.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    
    if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) 
            print(center) 
    color = (0, 0, 255)
    if radius > 0.5:
                # draw the circle and centroid on the frame,q
                # then update the list of tracked points
                #cv2.circle(mask_frame, (int(x), int(y)), int(radius), color, 2)
                cv2.circle(frame, (int(x), int(y)), int(radius), color, 2)

    cv2.imshow('image', frame)
    cv2.imshow('mask', mask_frame)

    #Check for inputs during video
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()














