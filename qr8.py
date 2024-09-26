import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()

    if not ret:
        break 

    blurred=cv2.GaussianBlur(frame,(5,5),0)

    bilateral=cv2.bilateralFilter(blurred,9,75,75)

    gray=cv2.cvtColor(bilateral,cv2.COLOR_BGR2GRAY)

    _,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    

    kernel=np.ones((5,5),np.uint8)
    morph=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)

    edges=cv2.Canny(morph,100,200)

    contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon=0.02*cv2.arcLength(contour,True)
        approx=cv2.approxPolyDP(contour,epsilon,True)

        if len(approx)==4 and cv2.contourArea(contour) >50:
            (x,y,w,h)=cv2.boundingRect(approx)
            aspect_ratio=float(w)/h

            if 0.95 <= aspect_ratio <= 1.05:
                cosines=[]
                for i in range(4):
                    p1 = approx[i][0]
                    p2 = approx[(i + 1) % 4][0]
                    p3 = approx[(i + 2) % 4][0]
                    v1 = p1 - p2
                    v2 = p3 - p2
                    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cosines.append(abs(cosine))
                if all(cos < 0.3 for cos in cosines):
                     cv2.drawContours(frame, [approx], 0, (133, 21, 199), 5)

    cv2.imshow('Detected Black Squares',frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()

