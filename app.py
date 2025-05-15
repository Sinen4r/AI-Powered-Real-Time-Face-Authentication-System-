from flask import Flask, render_template, Response,request,session
from ultralytics import YOLO
import cv2
from utulities import capture_and_process,add_new_user,capture_and_recognize
import os
import pickle
from insightface.app import FaceAnalysis
import faiss
import numpy as np
import mediapipe as mp

app = Flask(__name__)
app.secret_key = os.urandom(24)
camera = cv2.VideoCapture(0)

face_app = FaceAnalysis(name='buffalo_l')  
face_app.prepare(ctx_id=0, det_size=(160, 160))

model = YOLO("model.pt")


EYE_AR_THRESH = 0.3

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils


if os.path.exists("faces.index"):
    faiss_index = faiss.read_index("faces.index")
    with open("id_to_info.pkl", "rb") as f:
        id_to_info = pickle.load(f)
else:
    faiss_index = faiss.IndexFlatL2(512)
    id_to_info = {}


# def process_frame(frame):
#     # 🧠 PLACEHOLDER: Replace this with your model (YOLO, FaceNet, etc)
#     # For now, just return the same frame
#     return frame

# def gen_frames(enrolment=False):
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             conf,processed = process_frame(frame)
#             ret, buffer = cv2.imencode('.jpg', processed)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

@app.route('/start-course', methods=['POST'])
def userVerification():
    id_card = request.form['id_card']
    return render_template('stream.html', id_card=id_card,enrolment=False)

@app.route('/face_enrolment', methods=['POST'])
def userRegistration():
    id_card = request.form['id_card']
    name=request.form['name']
    classe=request.form["class"]

    session['name'] = name
    session['id_card'] = id_card
    session['class'] = classe

    return render_template('stream.html',enrolment=True)


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/login')
def login():
    idd = session.get('id_card')
    if not idd:
        return "Error: ID card not provided", 400
    print(f"ID card: {idd}")
    return Response(capture_and_recognize(model, scale=0.5, faiss_index=faiss_index, id_to_info=id_to_info,face_app=face_app,idd=idd),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/take_course',methods=['POST'])
def take_course():
    idd=request.form['id_card']
    session['id_card'] = idd
    return render_template('login.html')


@app.route('/video_feed')
def video_feed():
    enrolment_flag = request.args.get('enrolment', 'false').lower() == 'true'
    name = session.get('name')
    id_card = session.get('id_card')
    classe = session.get('class')
    return Response(capture_and_process(model,enrolment=enrolment_flag,faiss_index=faiss_index, id_to_info=id_to_info,name=name, ci=id_card, group=classe,face_app=face_app),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/registration')
def registration():
    return render_template('registration.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

if __name__ == '__main__':
    app.run(debug=True)

# Load the custom YOLOv8 face detection model



