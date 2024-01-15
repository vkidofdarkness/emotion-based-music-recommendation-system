import os
import cv2
import time
import numpy as np
import pandas as pd
from collections import Counter
from tensorflow.keras.models import load_model
import webbrowser

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
WIDTH, HEIGHT = 48, 48
MODEL_PATH = os.getenv('MODEL_PATH', 'emotion_detector.h5')
MUSIC_DATA_PATH = os.getenv('MUSIC_DATA_PATH', 'data_moods.csv')

# Функция для загрузки предварительно обученной модели эмоций.
def load_emotion_model():
    return load_model(MODEL_PATH)

# Функция обнаружения лиц на изображении с использованием каскада Хаара.
def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

# Функция для предсказания эмоции на лице с использованием модели эмоций.
def predict_emotion(face, model):
    face_resized = cv2.resize(face, (WIDTH, HEIGHT))
    face_normalized = face_resized / 255.0
    face_expanded = np.expand_dims(face_normalized, axis=0)
    face_expanded = np.expand_dims(face_expanded, axis=-1)
    return EMOTIONS[np.argmax(model.predict(face_expanded))]

# Функция для поиска музыки на основе настроения.
def find_music(tracks, mood):
    subset = tracks[tracks['mood'] == mood]
    random_song = subset.sample(n=1)
    music_url = "https://open.spotify.com/track/" + random_song['id'].values[0]
    play_music_on_spotify(music_url)       

# Функция для воспроизведения музыки в Spotify (браузер).
def play_music_on_spotify(url):
    webbrowser.open(url)

def main():
    tracks = pd.read_csv(MUSIC_DATA_PATH)
    tracks = tracks[['name','artist', 'id', 'mood']]

    # Загрузка предварительно обученной модели эмоций и каскада Хаара для обнаружения лиц.
    model = load_emotion_model()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
     # Инициализация переменных для хранения обнаруженных эмоций.
    emotions_detected = []
    start_time = time.time()

    while time.time() - start_time < 5:
        ret, frame = cap.read()
        faces = detect_faces(frame, face_cascade)

        # Обработка каждого обнаруженного лица.
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            emotion = predict_emotion(face, model)
            emotions_detected.append(emotion)
            
            # Отображение эмоции и области лица на кадре.
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    most_common_emotion = Counter(emotions_detected).most_common(1)[0][0]
    
    if most_common_emotion == 'happy' or most_common_emotion == 'sad':
        find_music(tracks, "Happy")

    if most_common_emotion == "disgust":
        find_music(tracks, "Sad")

    if most_common_emotion == "surprise" or most_common_emotion == "neutral":
        find_music(tracks, "Energetic")

    if most_common_emotion == "fear" or most_common_emotion == "angry":
        find_music(tracks, "Calm")

if __name__ == "__main__":
    main()