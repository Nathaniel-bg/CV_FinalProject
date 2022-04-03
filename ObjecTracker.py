import cv2
import numpy as np
import EasyPySpin

def getColorMask(frame):

    #setting the HSV values for the color detection
    lowerBound = np.array([0,170,80])
    upperBound = np.array([15,255,255])

    #Creating a mask for the specific color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lowerBound, upperBound)

def draw_grid(img, grid_shape, center, color=(0, 255, 0), thickness=1):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    x_pos = []
    y_pos = []
    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)
        x_pos.append(x)


    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)
        y_pos.append(y)

    #Obtain the min and max x coordinates of the object position
    arr_sizex = len(x_pos)
    prev_val = 0
    for vals in x_pos:
    
        if center[0] >= 0 and center[0] <= x_pos[0]:
            min_x = 0
            max_x = vals
            break

        if center[0] > prev_val and center[0] <= vals:
            min_x = prev_val
            max_x = vals
            break

        if center[0] > x_pos[arr_sizex -1] and center[0] <= w:
            min_x = x_pos[arr_sizex -1]
            max_x = w
            break

        prev_val = vals

    #Obtain the min and max y coordinates of the object position
    arr_sizey = len(y_pos)
    tmp_val = 0
    for vals in y_pos:
    
        if center[1] >= 0 and center[1] <= y_pos[0]:
            min_y = 0
            max_y = vals
            break

        if center[1] > tmp_val and center[1] <= vals:
            min_y = tmp_val
            max_y = vals
            break

        if center[1] > y_pos[arr_sizey -1] and center[1] <= h:
            min_y = y_pos[arr_sizey -1]
            max_y = h
            break

        tmp_val = vals

    #Draw a circle in the center of the box that contains the objec
    center_box = (int((max_x+min_x)/2), int((max_y+min_y)/2)) 
    cv2.circle(img, center_box, radius=5, color=(0,255,0), thickness=-1)

    return img


def main():
    cv2.namedWindow('image')
    #cap = cv2.VideoCapture(0)
    cap = EasyPySpin.VideoCapture(0)

    if not cap.isOpened():
        print('Could not open video device')
        return -1

    while cap.isOpened():

        #Capture the single frame
        ret, frame = cap.read()
        #Process the frame from bayerBG to BGR (Adds color to image)
        frame_bgr = cv2.demosaicing(frame, cv2.COLOR_BayerBG2BGR)
        #Create a mask for the designated color
        mask_frame = getColorMask(frame_bgr)

        #Filter the Mask using a median filter (gets rid of salt and pepper)
        mask_frame = cv2.medianBlur(mask_frame,5)

        #Find the contours of the mask
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
                #print(center) 
        color = (0, 0, 255)
        if radius > 0.5:
                    # draw the circle and centroid on the frame,q
                    # then update the list of tracked points
                    #cv2.circle(mask_frame, (int(x), int(y)), int(radius), color, 2)
                    cv2.circle(frame_bgr, (int(x), int(y)), int(radius), color, 2)

        #Get the image dimensions
        grid_dim = (10,10)
        image_pos= draw_grid(img=frame_bgr, grid_shape=grid_dim, center=center)

       
        #get_pos(center, x_grid, y_grid)

        #cv2.resize(image_grid, None, fx=0.25, fy=0.25)
        cv2.imshow('image', image_pos)
        #cv2.imshow('mask', mask_frame)
        #cv2.imshow('contour', cnts)

        #Check for inputs during video
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()











