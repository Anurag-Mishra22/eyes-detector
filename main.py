import os

import cv2

import matplotlib.pyplot as plt

data_dir = './data'
models_dir = './models'

face_detactor = cv2.CascadeClassifier(os.path.join(
models_dir, 'haarcascade_frontalface_default.xml'))
eyes_detector = cv2.CascadeClassifier(os.path.join(models_dir, 'haarcascade_eye_tree_eyeglasses.xml'))

for img_path in os.listdir(data_dir):
    img = cv2.imread(os.path.join(data_dir, img_path))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detactor.detectMultiScale(
        img_gray, minNeighbors=20)

    plt.figure()

    for face in faces:
        x1, y1, w, h = face
         
        # img = cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
        
        factor = 2
        face_ = img_gray[y1:y1 + h, x1:x1 + w]
        eyes = eyes_detector.detectMultiScale(cv2.resize(face_, (int(w * factor), int(h * factor))))
        for eye in eyes:
            eye = [int(e / factor) for e in eye]
            x2, y2, w2, h2 = eye
            
            img = cv2.rectangle(img, (x1+x2, y1+y2), (x1+x2+w2, y1+y2+h2), (0, 255, 0), 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)

plt.show()
