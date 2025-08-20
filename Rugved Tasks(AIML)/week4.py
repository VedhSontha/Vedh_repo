import cv2 
import numpy as np
cap = cv2.VideoCapture('1') 
def balltracking():
    while True:
        istrue,frame=cap.read()
        if not istrue:
            print("Failed")
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([35, 100, 100])
        upper = np.array([85, 255, 255]) 

        mask = cv2.inRange(hsv,lower,upper)
        contours,heirarchies= cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>1:
            
            c = max(contours, key=cv2.contourArea)
            ((x,y),r)=cv2.minEnclosingCircle(c)
            M=cv2.moments(c)
            
            cv2.circle(frame,(int(x),int(y)),int(r),(0,0,255),6)

        cv2.imshow("Ball Tracking",frame)

        if cv2.waitKey(10)&0xFF==ord('d'):
            break
    cap.release()
    cv.DestroyAllWindows()
balltracking()
