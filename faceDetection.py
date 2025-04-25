import cv2 as cv
from ultralytics import YOLO
import threading
# from insightface.app import FaceAnalysis
# import faiss
import numpy as np
frame = None
running = True

# Load YOLO model
model = YOLO("model.pt")
# faiss_index = faiss.read_index("faces.index")

def capture_frames():
    global frame, running
    cap = cv.VideoCapture(0)
    while running:
        ret, frm = cap.read()
        
        if ret:

            frame = cv.flip(frm,1)
    cap.release()

thread = threading.Thread(target=capture_frames)
thread.start()

scale=0.5
while True:
    if frame is not None:
        current_frame = frame.copy()
        small_frame = cv.resize(current_frame, (0, 0), fx=scale, fy=scale)

        results = model(small_frame, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1,y1,x2,y2=[int(i/scale) for i in (x1, y1, x2, y2 )]
            conf = float(box.conf)
            if conf > 0.5:
                cv.rectangle(current_frame, (x1, y1), (x2, y2), (129, 200, 500), 2) 
                cv.putText(current_frame, f"Face {conf:.2f}", (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv.imshow("YOLOv8 Face Detection (Threaded)", current_frame)

    if cv.waitKey(1) == ord("q"):
        running = False
        break

cv.destroyAllWindows()
thread.join()  


# # Initialize InsightFace
# face_app = FaceAnalysis(name='buffalo_l')  
# face_app.prepare(ctx_id=0, det_size=(640, 640))


# def get_embedding(face_image):
#     faces = face_app.get(face_image)
#     if faces:
#         return faces[0].embedding  # Returns 512-dim vector
#     return None




# dimension = 512
# faiss_index = faiss.IndexFlatL2(dimension)
# id_to_info = {}

# def add_new_user(name,ci,group, image):
#     embedding = get_embedding(image)
#     if embedding is not None:
#         embedding = np.array([embedding]).astype('float32')
#         faiss_index.add(embedding)
#         person_id = len(id_to_info)
#         id_to_info[person_id] = {
#             "name": name,
#             "group": group,
#             "id_card": ci
#         }
#         faiss.write_index(faiss_index, "faces.index")

#         print(f"{name} added to database.")
#     else:
#         print("Face not detected.")

# def recognize_face(image):
#     embedding = get_embedding(image)
#     if embedding is None:
#         return "No face"
#     embedding = np.array([embedding]).astype('float32')
#     D, I = faiss_index.search(embedding, 1)
#     if D[0][0] < 1.0:  # Distance threshold, tweak as needed
#         return id_to_info[I[0][0]]["id_card"]
#     else:
#         return "Unknown"

# # Load later:
# import pickle
# with open("id_to_name.pkl", "wb") as f:
#     pickle.dump(id_to_info, f)
