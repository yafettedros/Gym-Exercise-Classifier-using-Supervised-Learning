import cv2
import os

def extract_frames(video_path, save_dir, label):
    os.makedirs(os.path.join(save_dir, label), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_num % 30 == 0:
            frame = cv2.resize(frame, (224, 224))
            save_path = os.path.join(save_dir, label, f"frame_{frame_num}.jpg")
            cv2.imwrite(save_path, frame)
        frame_num += 1
    cap.release()

# Example:
# extract_frames('gym_videos/squats/video1.mp4', 'frames', 'squats')


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_dir = 'frames'
data = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = data.flow_from_directory(train_dir, target_size=(224,224), batch_size=32, class_mode='categorical', subset='training')
val_data = data.flow_from_directory(train_dir, target_size=(224,224), batch_size=32, class_mode='categorical', subset='validation')

model.fit(train_data, validation_data=val_data, epochs=5)
model.save('exercise_model.h5')


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

model = load_model('exercise_model.h5')
labels = ['Push-Up', 'Squat', 'Deadlift', 'Pull-Up', 'Bench Press']

wrong_predictions = []

for label in labels:
    folder = os.path.join('frames', label)
    for img_name in os.listdir(folder):
        path = os.path.join(folder, img_name)
        img = load_img(path, target_size=(224, 224))
        arr = img_to_array(img) / 255.0
        pred = model.predict(np.expand_dims(arr, axis=0), verbose=0)
        guess = labels[np.argmax(pred)]
        if guess != label:
            wrong_predictions.append((img_name, label, guess))

for item in wrong_predictions[:5]:
    print("Wrong Prediction:", item)
