import cv2

if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            trc = (x, y)  # face rectangle's top-right corner
            blc = (x + w, y + h)  # face rectangle's bottom-left corner
            rectcolor = (255, 0, 0)  # blue color
            thick = 2  # line thickness

            # draw a rectangle whenever a face is detected
            cv2.rectangle(img, trc, blc, rectcolor, thick)

            # slice image area containing the face
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # read the face area and detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                trc = (ex, ey)  # eye rectangle's top-right corner
                blc = (ex + ew, ey + eh)  # eye rectangle's bottom-left corner
                rectcolor = (0, 255, 0)  # green color
                thick = 2  # line thickness

                # draw a rectangle whenever an eye is detected
                cv2.rectangle(roi_color, trc, blc, rectcolor, thick)

        cv2.imshow('Face Detector', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()