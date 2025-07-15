import cv2
import os
import numpy as np

dataset_path = "dataset"
images, labels = [], []
label_map = {}
label_id = 0

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue

    label_map[label_id] = person_name
    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 200))
        images.append(img)
        labels.append(label_id)
    label_id += 1

images = np.array(images)
labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, labels)
recognizer.save("trained_model.yml")

# Save label map
with open("labels.txt", "w") as f:
    for k, v in label_map.items():
        f.write(f"{k}:{v}\n")

print("Model trained and saved as trained_model.yml")
