import dlib
import cv2
import numpy as np
from imutils import face_utils
import matplotlib.pyplot as plt

# Modellerin dosya yolları
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# Dlib'in yüz dedektörünü ve yüz işaretleyicisini yükle
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
face_rec = dlib.face_recognition_model_v1(face_rec_model_path)

# Görüntüyü yükle
image_path = '/Users/melisabagcivan/Desktop/case/IMG_5083 2.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Yüz tespiti yap
faces = detector(gray, 1)

# Her bir yüzü işle
for i, rect in enumerate(faces):
    # Yüzü belirle ve işaretleyicileri kullanarak yüzü kırp
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cropped_face = image[y:y + h, x:x + w]

    # Kırpılan yüzü kaydet
    cropped_face_bgr = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'/Users/melisabagcivan/Desktop/case/OutputImage/cropped_face_{i + 1}.jpg', cropped_face_bgr)

    # Orijinal görüntüde yüzü işaretle
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y) in shape:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

# Görüntüyü göster
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
