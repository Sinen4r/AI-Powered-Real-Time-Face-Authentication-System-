import cv2 as cv
from ultralytics import YOLO

import threading
from insightface.app import FaceAnalysis
import faiss
import numpy as np
import pprint
import os
import pickle
from flask import session,render_template
from faceliveness import detect_liveness


# def capture_and_process(model, enrolment=False,scale=0.5,faiss_index=None, id_to_info=None,name="", ci="", group=""):
#     cap = cv.VideoCapture(0)

#     if not cap.isOpened():
#         print("Cannot access camera")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         frame = cv.flip(frame, 1)
#         current_frame = frame.copy()

#         small_frame = cv.resize(current_frame, (0, 0), fx=scale, fy=scale)

#         results = model(small_frame, verbose=False)[0]

#         for box in results.boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = [int(i / scale) for i in (x1, y1, x2, y2)]
#             conf = float(box.conf)

#             if conf > 0.5:
#                 cv.rectangle(current_frame, (x1, y1), (x2, y2), (129, 200, 500), 2)
#                 cv.putText(current_frame, f"Face {conf:.2f}", (x1, y1 - 10),
#                            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
#                 if enrolment and conf>0.8:
#                     face_image = current_frame[y1:y2, x1:x2]
#                     add_new_user(name,ci,group, face_image,faiss_index,id_to_info)
#                     yield "data: ENROLLMENT_DONE\n\n"                    
#                     break

#         ret, buffer = cv.imencode('.jpg', current_frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def capture_and_process(model, enrolment=False, scale=0.5, faiss_index=None, id_to_info=None, name="", ci="", group="",face_app=None):
    cap = cv.VideoCapture(0,cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FPS, 25)  
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)  
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Cannot access camera")
        yield "data: ERROR: Cannot access camera\n\n"
        return
    count=0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            yield "data: ERROR: Failed to grab frame\n\n"
            
        else:
            frame = cv.flip(frame, 1)
            current_frame = frame.copy()

            small_frame = cv.resize(current_frame, (0, 0), fx=scale, fy=scale)

            results = model(small_frame, verbose=False)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = [int(i / scale) for i in (x1, y1, x2, y2)]
                conf = float(box.conf)

                if conf > 0.5:
                    cv.rectangle(current_frame, (x1, y1), (x2, y2), (129, 200, 500), 2)
                    cv.putText(current_frame, f"Face {conf:.2f}", (x1-20, y1 - 10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv.putText(current_frame,"Processing the face Stay Still!!",((x1-20),y2+15),cv.FONT_HERSHEY_SIMPLEX,0.6,(191,95,0),2)
                    
                    if enrolment and conf > 00.99 and count>50 :
                            face_image = current_frame[y1:y2, x1:x2]
                            add_new_user(name, ci, group, face_image, faiss_index, id_to_info,face_app)
                            yield "data: ENROLLMENT_DONE\n\n"
                            return  
                
            count+=1
            ret, buffer = cv.imencode('.jpg', current_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
def get_embedding(face_image,face_app):
    cv.imwrite("face.png",face_image)
    faces = face_app.get(face_image)
    
    if faces:
        embedding= faces[0].embedding  
    return embedding / np.linalg.norm(embedding)



def add_new_user(name,ci,group, image,faiss_index,id_to_info,face_app):
    embedding = get_embedding(image,face_app)
    if embedding is not None:
        embedding = np.array([embedding]).astype('float32')
        faiss_index.add(embedding)
        person_id = len(id_to_info)
        id_to_info[person_id] = {
            "name": name,
            "group": group,
            "id_card": ci
        }
        faiss.write_index(faiss_index, "faces.index")
        with open("id_to_info.pkl", "wb") as f:
            pickle.dump(id_to_info, f)

        print(f"{name} added to database.")
    else:
        print("Face not detected.")




def capture_and_recognize(model, scale=0.5, faiss_index=None, id_to_info=None,face_app=None,idd=None):
    cap = cv.VideoCapture(0,cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FPS, 25)  
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)  
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]    # [left corner, top-mid1, top-mid2, right corner, bottom-mid1, bottom-mid2]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    NOSE_TIP = 1
    if not cap.isOpened():
        print("Cannot access camera")
        yield "data: ERROR: Cannot access camera\n\n"
        return
    count=0
    countt=0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            yield "data: ERROR: Failed to grab frame\n\n"
            
        else:
            frame = cv.flip(frame, 1)
            current_frame = frame.copy()

            small_frame = cv.resize(current_frame, (0, 0), fx=scale, fy=scale)

            results = model(small_frame, verbose=False)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = [int(i / scale) for i in (x1, y1, x2, y2)]
                conf = float(box.conf)

                if conf > 0.5:
                    cv.rectangle(current_frame, (x1, y1), (x2, y2), (129, 200, 500), 2)
                    cv.putText(current_frame, f"Face {conf:.2f}", (x1, y1 - 10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv.putText(current_frame,"Recognising the face Stay Still..",((x1-20),y2+15),cv.FONT_HERSHEY_SIMPLEX,0.6,(191,95,0),2)
                    liveness=detect_liveness(current_frame[y1:y2, x1:x2])
                    if not liveness:
                        countt+=1
                    else:
                        countt-=1
                        if conf>0.8 and count>50 :
                            face_image = current_frame[y1:y2, x1:x2]
                            print("recognize_face")
                            yield recognize_face(face_image,faiss_index,id_to_info,idd,face_app)
                        
                    if countt>14:
                        print("Spoofing")
                        yield "data: Spoofing_detected\n\n"
            count+=1
            ret, buffer = cv.imencode('.jpg', current_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def recognize_face(image,faiss_index,id_to_info,id,face_app):
    
    embedding = get_embedding(image,face_app)
    if embedding is None:
        print("no face")
        return "No face"
    embedding = np.array([embedding]).astype('float32')
    D, I = faiss_index.search(embedding, 1)
    print(D[0])
    if D[0][0] < 1.0:  
        print("login succ")
        if id_to_info[I[0][0]]["id_card"]==id:
            return  "data: login_succesuful\n\n"
    else:
        print("fail")
        return "data: login_failed"

