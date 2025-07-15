import cv2

# Load model and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_model.yml")

label_map = {}
with open("labels.txt", "r") as f:
    for line in f:
        k, v = line.strip().split(":")
        label_map[int(k)] = v

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
        label, confidence = recognizer.predict(roi)

        name = label_map.get(label, "Unknown")
        color = (0, 255, 0) if confidence < 100 else (0, 0, 255)
        text = f"{name} ({round(confidence, 1)})" if confidence < 100 else "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
