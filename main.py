import cv2
xml_haar_cascade = 'haarcascade_frontalface_alt2.xml'
faceclass = cv2.CascadeClassifier(xml_haar_cascade)


cv2.VideoCapture(0)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv2.CAP_PROP_EXPOSURE, 0.1)

while not cv2.waitKey(20) & 0xFF == ord("q"):
    ret, frame_color = capture.read()
    gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

    faces = faceclass.detectMultiScale(gray)

    for x, y, w, h in faces:
        cv2.rectangle(frame_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow('color', frame_color)
        cv2.imshow('gray', gray)