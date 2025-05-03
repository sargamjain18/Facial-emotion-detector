from flask import Flask, render_template, Response, request, redirect, url_for, session
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

app = Flask(__name__)
app.secret_key = 'secret'  # Needed for session usage

# Load models
face_classifier = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
classifier = load_model('model/model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

camera = cv2.VideoCapture(0)

detected_emotion = None

def generate_frames():
    global detected_emotion
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = classifier.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]
                    detected_emotion = label
                    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/start')
def start_detection():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/questions')
def questions():
    session['emotion'] = detected_emotion
    emotion = detected_emotion if detected_emotion else 'Neutral'
    return render_template('questions.html', emotion=emotion, questions=get_mcqs(emotion))

@app.route('/submit', methods=['POST'])
def submit():
    answers = request.form

    scoring_map = {
        "Never": 1,
        "Rarely": 2,
        "Sometimes": 3,
        "Frequently": 4,
        "Deep Breathing": 1,
        "Talking to someone": 2,
        "Avoidance": 3,
        "Aggression": 4
    }

    score = sum([scoring_map.get(value, 0) for value in answers.values()])
    emotion = session.get('emotion', 'Neutral')
    return render_template('report.html', score=score, emotion=emotion)



def get_mcqs(emotion):
    mcq_bank = {
        'Happy': [
            ("How often do you feel genuinely joyful during your daily routine?", ["Never", "Rarely","Sometimes","Frequently"]),
            ("Do you feel a strong sense of purpose in your life?", ["Never", "Rarely","Sometimes","Frequently"]),
            ("Are you satisfied with your current quality of life?", ["Never", "Rarely","Sometimes","Frequently"]),
            ("How often do you feel genuinely joyful in a typical day?", ["Never", "Rarely","Sometimes","Frequently"]),
            ("Do you look forward to daily activities or plans?", ["Never", "Rarely","Sometimes","Frequently"]),
        ],
        'Sad': [
            ("How often do you feel down, depressed, or hopeless?", ["Never", "Rarely","Sometimes","Frequently"]),
            ("Do you feel worthless or excessively guilty?", ["Never", "Rarely","Sometimes","Frequently"]),
            ("How often do you cry or feel like crying?", ["Never", "Rarely","Sometimes","Frequently"]),
            ("Do you feel isolated even when surrounded by people?", ["Never", "Rarely","Sometimes","Frequently"]),
            ("How often do you feel misunderstood or alone?", ["Never", "Rarely","Sometimes","Frequently"]),
        ],
        'Angry': [
            ("How often do you feel irritated by small inconveniences?", ["Never", "Rarely","Sometimes","Frequently"]),
            ("How often do you lose your temper?", ["Never", "Rarely","Sometimes","Frequently"]),
            ("Do you have difficulty forgiving others?", ["Never", "Rarely","Sometimes","Frequently"]),
            ("How do you cope with frustration?", ["Never", "Rarely","Sometimes","Frequently"]),
            ("Do you feel your anger impacts your relationships?", ["Never", "Rarely","Sometimes","Frequently"]),
        ],
        'Neutral': [
            ("How often do you feel emotionally numb or detached?", ["Never", "Rarely","Sometimes","Often"]),
            ("Do you ever feel like you're going through life on autopilot?", ["Never", "Rarely","Sometimes","Often"]),
            ("Do you find it difficult to feel strong emotions (positive or negative)?", ["Never", "Rarely","Sometimes","Often"]),
            ("How often do you feel emotionally flat?", ["Never", "Rarely","Sometimes","Often"]),
            ("Are you able to enjoy activities even if you're not excited about them?", ["Never", "Rarely","Sometimes","Often"]),
        ],
        'Fear': [
            ("How often do you feel anxious without a clear reason?", ["Never", "Rarely", "Sometimes", "Frequently"]),
            ("Do you avoid certain situations due to fear or discomfort?", ["Never", "Rarely", "Sometimes", "Frequently"]),
            ("How often do you feel overwhelmed by your thoughts?", ["Never", "Rarely", "Sometimes", "Frequently"]),
            ("Do you experience physical symptoms like sweating or rapid heartbeat when afraid?", ["Never", "Rarely", "Sometimes", "Frequently"]),
            ("How often do you feel that fear affects your ability to make decisions?", ["Never", "Rarely", "Sometimes", "Frequently"])
        ],
        'Disgust': [
        ("How often do you feel physically repelled by certain smells or tastes?", ["Never", "Rarely", "Sometimes", "Frequently"]),
        ("Do you find yourself avoiding people or places that make you feel uncomfortable?", ["Never", "Rarely", "Sometimes", "Frequently"]),
        ("How sensitive are you to things that you consider unsanitary or unclean?", ["Never", "Rarely", "Sometimes", "Frequently"]),
        ("Do you get easily disgusted by things others might find normal?", ["Never", "Rarely", "Sometimes", "Frequently"]),
        ("Does seeing certain images (like insects or dirt) disturb you?", ["Never", "Rarely", "Sometimes", "Frequently"])
        ],
        'Surprise': [
        ("How often do you find yourself surprised by unexpected events?", ["Never", "Rarely", "Sometimes", "Frequently"]),
        ("When something unexpected happens, do you feel excited or overwhelmed?", ["Never", "Rarely", "Sometimes", "Frequently"]),
        ("How often do you find it hard to predict the outcome of a situation?", ["Never", "Rarely", "Sometimes", "Frequently"]),
        ("Do surprises tend to change your mood significantly?", ["Never", "Rarely", "Sometimes", "Frequently"]),
        ("How comfortable are you with sudden, unexpected changes in your routine?", ["Never", "Rarely", "Sometimes", "Frequently"])
        ],    
    }
    return mcq_bank.get(emotion, mcq_bank['Neutral'])

if __name__ == "__main__":
    app.run(debug=True)
