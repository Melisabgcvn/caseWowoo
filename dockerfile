FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "master:app", "--host", "0.0.0.0", "--port", "8000"]


COPY shape_predictor_68_face_landmarks.dat /app/
COPY dlib_face_recognition_resnet_model_v1.dat /app/
