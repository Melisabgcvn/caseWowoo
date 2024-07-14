from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
#import dnnlib
#import dnnlib.tflib as tflib
import PIL.Image
from io import BytesIO
import requests
from PIL import ImageDraw
import pickle
import cv2
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import base64

#StyleGan2 ve stylegan2-ada kullanılabilir.
app = FastAPI()

url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl' #önceden eğitilmiş model 
#url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl' #önceden eğitilmiş farklı model

"""
tflib.init_tf()
with dnnlib.util.open_url(url) as f:
    generator_network, discriminator_network, Gs = pickle.load(f)
"""

image_path = '/Users/melisabagcivan/Desktop/case/IMG_5083 2.jpg'
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

# MTCNN detektörünü oluştur
detector = MTCNN()

# Yüz tespiti yap ve yüzleri kırp
faces = detector.detect_faces(image)
for i, face in enumerate(faces):
    x, y, width, height = face['box']
    
    # Yüzü kırp
    cropped_face = image[y:y+height, x:x+width]
    
    # Kırpılan yüzü kaydet
    cropped_face_bgr = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'/Users/melisabagcivan/Desktop/case/OutputImage/cropped_face_{i+1}.jpg', cropped_face_bgr)

    # Orijinal görüntüde yüzü işaretle
    cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)
    for key, point in face['keypoints'].items():
        cv2.circle(image, point, 2, (0, 255, 0), 2)

# Görüntüyü göster
plt.imshow(image)
plt.axis('off')
plt.show()

# Görüntü yükleme ve işleme fonksiyonu
def load_image_from_url(url):
    response = requests.get(url)
    img = PIL.Image.open(BytesIO(response.content))
    return img

# Yüz tespiti ve izolasyonu için MTCNN kullanımı
def detect_and_crop_face(image):
    # MTCNN kullanarak yüz tespiti
    detector = MTCNN()
    faces = detector.detect_faces(image)
    
    if len(faces) == 0:
        print(" YÜZ BULUNAMADI")
        return None
    
    face = faces[0]  # İlk yüzü al
    x, y, width, height = face['box']
    cropped_face = image[y:y+height, x:x+width]
    
    return cropped_face
"""
@app.post("/age-face/")
async def age_face(file: UploadFile = File(...)):
    # Yüklenen görüntüyü oku
    contents = await file.read()
    image = cv2.cvtColor(np.array(PIL.Image.open(BytesIO(contents))), cv2.COLOR_RGB2BGR)

    # Yüz tespiti yap ve kırp
    cropped_face_image = detect_and_crop_face(image)

    if cropped_face_image is None:
        return JSONResponse(content={"error": "Yüz Bulunamadı"}, status_code=400)

    # Yaşlandırılmış yüzler oluştur
    age_shifts = [10, 30, 50, 70]
    aged_faces = []
    for age_shift in age_shifts:
        aged_face_image = age_face_with_stylegan(cropped_face_image, Gs, age_shift=age_shift)
        buffered = BytesIO()
        aged_face_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        aged_faces.append({"age_shift": age_shift, "image": img_str})
    
    return {"aged_faces": aged_faces}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    """