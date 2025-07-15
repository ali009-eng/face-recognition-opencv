import cv2
import os

name = input("Enter person's name: ")
save_path = os.path.join("dataset", name)
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0
while count < 50:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        count += 1
        cv2.imwrite(f"{save_path}/{count}.jpg", face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Capturing Faces", frame)
    if cv2.waitKey(1) == ord('q'):
        break

print(f"\nSaved {count} images to {save_path}")
cap.release()
cv2.destroyAllWindows()
