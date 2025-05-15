import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

# Eye aspect ratio (EAR) threshold for blink detection

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# EAR helper function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Mapping MediaPipe landmark indices to eye landmarks (approximation of dlib 68 landmarks)
LEFT_EYE = [33, 160, 158, 133, 153, 144]    # [left corner, top-mid1, top-mid2, right corner, bottom-mid1, bottom-mid2]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1

def detect_liveness(image):
    EYE_AR_THRESH = 0.26
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return "Face not detected"

    landmarks = results.multi_face_landmarks[0].landmark

    # Extract eye landmarks
    left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
    right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]

    # Compute EAR
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    blink_detected = left_ear < EYE_AR_THRESH or right_ear < EYE_AR_THRESH
    print(left_ear,left_ear)

    # Head movement detection (based on symmetry)
    nose = np.array([landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h])
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)

    left_nose_dist = np.linalg.norm(nose - left_eye_center)
    right_nose_dist = np.linalg.norm(nose - right_eye_center)
    movement_detected = abs(left_nose_dist - right_nose_dist) > 10

    # Debug prints
    if blink_detected:
        print("Blink detected!")
    if movement_detected:
        print("Head movement detected!")
    # if results.multi_face_landmarks:
    #     for face_landmarks in results.multi_face_landmarks:
    #         mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

    return (True) if blink_detected or movement_detected else (False)

# Testing with an image
# image = cv2.imread('spoofImg.jpg')
# result = detect_liveness(image)
# print(result)

# # Draw face mesh for visualization
# rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# results = face_mesh.process(rgb_image)
# if results.multi_face_landmarks:
#     for face_landmarks in results.multi_face_landmarks:
#         mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

# # Show result
# cv2.imshow("Face", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
