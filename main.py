
import os
import cv2
import numpy as np
import datetime
import winsound
from pathlib import Path
from tkinter import Tk, Label, Button, messagebox, filedialog, Toplevel
from PIL import ImageTk, Image
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"

TWILIO_SID = "YOUR_TWILIO_SID"
TWILIO_AUTH = "YOUR_TWILIO_AUTH_TOKEN"
TWILIO_FROM = "YOUR_TWILIO_NUMBER"
TWILIO_TO = "DESTINATION_NUMBER"

CLASSES = ["No Accident", "Accident"]
ACCIDENT_CONSEC_FRAMES = 30

def send_sms(message):
    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_AUTH)
        client.messages.create(body=message, from_=TWILIO_FROM, to=TWILIO_TO)
    except Exception as e:
        print("SMS Failed:", e)

def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype("float32") / 255.0
    mean_vec = gray.mean(axis=1)
    return mean_vec.reshape(1, 224, 1)

def predict(model, frame):
    x = preprocess(frame)
    pred = model.predict(x, verbose=0)
    idx = np.argmax(pred[0])
    return CLASSES[idx], float(pred[0][idx])

def run_detection(model, cap):
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, conf = predict(model, frame)

        if label == "Accident":
            count += 1
            if count >= ACCIDENT_CONSEC_FRAMES:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                alert = f"Accident Detected at {timestamp}"
                print(alert)
                winsound.Beep(2500, 1000)
                send_sms(alert)
                count = 0
        else:
            count = 0

        color = (0,0,255) if label=="Accident" else (0,255,0)
        cv2.putText(frame, f"{label} ({conf:.2f})", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Accident Detection System", frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    if not MODEL_PATH.exists():
        messagebox.showerror("Error", "model.h5 not found!")
        return

    model = load_model(str(MODEL_PATH))

    root = Tk()
    root.title("Smart Accident Detection System")
    root.geometry("800x500")
    root.configure(bg="#1e1e1e")
    root.resizable(False, False)

    Label(root, text="Smart AI Accident Detection System",
          font=("Arial",20,"bold"),
          fg="white", bg="#1e1e1e").pack(pady=40)

    def realtime():
        cap = cv2.VideoCapture(0)
        run_detection(model, cap)

    def upload_video():
        file = filedialog.askopenfilename(
            filetypes=[("Video Files","*.mp4 *.avi *.mkv")]
        )
        if file:
            cap = cv2.VideoCapture(file)
            run_detection(model, cap)

    def about():
        win = Toplevel(root)
        win.geometry("400x300")
        win.title("About Project")
        win.configure(bg="#2b2b2b")

        Label(win, text="AI Powered Accident Detection\n\n"
                        "Technologies:\n"
                        "- OpenCV\n"
                        "- TensorFlow\n"
                        "- Tkinter GUI\n"
                        "- Twilio SMS",
              fg="white", bg="#2b2b2b",
              font=("Arial",12)).pack(pady=40)

    Button(root, text="Upload Video", width=25, height=2,
           command=upload_video, bg="#4CAF50", fg="white").pack(pady=10)

    Button(root, text="Real-Time Camera", width=25, height=2,
           command=realtime, bg="#2196F3", fg="white").pack(pady=10)

    Button(root, text="About", width=25, height=2,
           command=about, bg="#9C27B0", fg="white").pack(pady=10)

    Button(root, text="Exit", width=25, height=2,
           command=root.destroy, bg="#f44336", fg="white").pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
