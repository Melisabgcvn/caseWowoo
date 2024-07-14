1 Yüz Tanıma Modelinin kurulumu ve kullanılması 

Öncelikle MTCNN modelinin kurulumu yapılır.

pip install mtcnn pillow

-MTCNN kütüphanesinin import edilmesi
from mtcnn.mtcnn import MTCNN

MTCNN modeli kullanılarak resimdeki yüz tespit edilir. Tespit edilen yüz kırpılır. Yüz bulunamama durumunda uyarı verilecek şekilde ayarlanır.

<img width="684" alt="Screenshot 2024-07-14 at 15 28 18" src="https://github.com/user-attachments/assets/1beec5a4-1f6d-4a07-82fe-9de8ffd0bf04">

2.   Dlib ile Yüz Tanıma

   
<img width="792" alt="Screenshot 2024-07-14 at 15 27 06" src="https://github.com/user-attachments/assets/a66c0df6-a1ce-49a5-ab2c-a9dfb4b91546">

-Kütüphane kurulumunun terminal üzerinden yapılması
pip install dlib opencv-python

https://github.com/ageitgey/face_recognition_models/blob/master/face_recognition_models/models/dlib_face_recognition_resnet_model_v1.dat

https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat

Adresi üzerinden dat dosyalarının indirilmesi gerekemektedir.

İndirilen dosyalar proje dosyasının içerisine aktarılır.

pip install imutils
Denilerek kütüphane kurulumu yapılır.


2. FastApi kurulumu 

-pip install fastapi uvicorn
ile Fast api kurulumlarını yapıyoruz. 

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

İle fastApi kütüphanesi ve dönüş cevapları için Response kütüphanesi import edilir. 
app = FastAPI() 
FastApi nesnesi oluşturulur.

@app.post("/age-face/")
bir HTTP istemcisi kullanarak /age-face/ yoluna bir POST isteği gönderilebilir. İstek gövdesine bir görüntü dosyası eklenir ve bu dosya üzerinde yüz tespiti ve yaşlandırma işlemi yapılır.

uvicorn.run(app, host="0.0.0.0", port=8000) ile fastApi uygulaması başlatılır. Burada belirtilen host ve port bilgilerinin kullanımdan önce değiştirilmesi gerekmektedir. 

uvicorn master:app --reload
Terminal üzerinden komut ile uygulama başlatılır.
Master : dosyanın ismi(master.py)


3. DOCKER KURULUMU
https://www.docker.com/products/docker-desktop/
Sitesi üzerinden işletim sisteminize uygun docker desktop uygulamasının indirilmesi gerekmektedir.

1.Dockerfile Oluşturma
Dockerfile adında bir dosya oluşturulur. İçerisine proje gerkesinimlerini indirme için gerekli komutlar ve port bilgileri eklenir. 

2. Requirements Dosyası Oluşturma
Proje içerisinde kullanılan kütüphanelerin kurulumları için gerekli kütüphane isimleri ve versiyonları eklenir.

3. Terminal sistemi üzerinde aşağıdaki komutları kullanarak Docker image oluşturulur ve çalıştırılır.

# Docker image oluşturma
docker build -t fastapi-face-detection .

fastapi-face-detection adında bir Docker image oluşturur.

# Docker container çalıştırma
docker run -d -p 8000:8000 fastapi-face-detection

<img width="539" alt="Screenshot 2024-07-14 at 15 24 20" src="https://github.com/user-attachments/assets/8d89494b-da22-452a-b5a1-dacd636f5f56">




Kaynaklar:

https://arxiv.org/pdf/2102.02754

Model alınması için 
https://github.com/NVlabs/stylegan2/blob/master/pretrained_networks.py
