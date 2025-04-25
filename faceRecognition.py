
import cv2 as cv
from ultralytics import YOLO
import threading
from insightface.app import FaceAnalysis
import faiss
import numpy as np
import pprint
import os
import pickle

face_app = FaceAnalysis(name='buffalo_l')  
face_app.prepare(ctx_id=0, det_size=(160, 160))


def get_embedding(face_image):
    faces = face_app.get(face_image)
    if faces:
        return faces[0].embedding  # Returns 512-dim vector
    return None


img=cv.imread("face.png")
if os.path.exists("faces.index"):
    faiss_index = faiss.read_index("faces.index")
    with open("id_to_info.pkl", "rb") as f:
        id_to_info = pickle.load(f)
else:
    faiss_index = faiss.IndexFlatL2(512)
    id_to_info = {}



def add_new_user(name,ci,group, image):
    embedding = get_embedding(image)
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

def recognize_face(image):
    embedding = get_embedding(image)
    if embedding is None:
        return "No face"
    embedding = np.array([embedding]).astype('float32')
    D, I = faiss_index.search(embedding, 1)
    if D[0][0] < 200.0:  
        return id_to_info[I[0][0]]["id_card"]
    else:
        return "Unknown"
    
def get_faiss_database_contents(faiss_index, id_to_info):
    """
    Retrieve contents of FAISS index and corresponding metadata.
    
    Args:
        faiss_index: FAISS index containing face embeddings
        id_to_info: Dictionary mapping index IDs to user metadata
    
    Returns:
        List of dictionaries containing user data
    """
    if faiss_index is None or id_to_info is None:
        return []
    
    num_embeddings = faiss_index.ntotal  # Total number of embeddings
    users = []
    
    for i in range(num_embeddings):
        if i in id_to_info:
            user_info = id_to_info[i].copy()  # Copy to avoid modifying original
            user_info['index'] = i  # Add FAISS index for reference
            users.append(user_info)
        else:
            # Handle cases where metadata is missing for an index
            users.append({
                'index': i,
                'id_card': 'Unknown',
                'name': 'Unknown',
                'group': 'Unknown'
            })
    
    return users
users = get_faiss_database_contents(faiss_index, id_to_info)
print(users)
print(recognize_face(img))