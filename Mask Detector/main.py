import cv2 as cv

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_smile.xml")

capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()

    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray,1.1,50)

    for (x,y,w,h) in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)


    eye = eye_cascade.detectMultiScale(gray,1.1,100)

    for (x,y,w,h) in eye:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


    mouth = mouth_cascade.detectMultiScale(gray,1.1,100)

    for (x,y,w,h) in mouth:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
        

    if len(face) > 0:
        if len(eye) > 0 and len(mouth) > 0:
            cv.putText(frame,("No Mask"),(0,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv.LINE_AA)
        elif len(eye) > 0 and len(mouth) == 0:
            cv.putText(frame,("Mask"),(0,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv.LINE_AA)
        else:
            pass

    
    cv.imshow("Mask Detector",frame)

    if cv.waitKey(1) == ord("q"):
        break

capture.release()
cv.destroyAllWindows()