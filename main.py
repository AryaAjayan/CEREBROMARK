import cv2
import numpy as np
import insightface
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Settings
MODEL_NAME = "buffalo_l"
THRESHOLD = 0.6
DB_DIR = "live_face_db"  # Save cropped faces (optional, for inspection)
os.makedirs(DB_DIR, exist_ok=True)

# Face DB: a list of (embedding, label)
face_db = []

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

def find_match(embedding, db, threshold=THRESHOLD):
    if not db:
        return None
    sims = [cosine_sim(embedding, e) for e, _ in db]
    max_idx = np.argmax(sims)
    if sims[max_idx] > threshold:
        return db[max_idx][1]
    else:
        return None

def main():
    app = insightface.app.FaceAnalysis(name=MODEL_NAME, providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    cap = cv2.VideoCapture(0)
    next_face_id = 1

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = app.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            emb = face.embedding
            # Check DB for match
            label = find_match(emb, face_db)
            if label is None:
                label = f"face{next_face_id}"
                next_face_id += 1
                face_db.append((emb.copy(), label))
                # Optionally save face crop for inspection
                h, w = frame.shape[:2]
                x1, y1 = max(0, bbox[0]), max(0, bbox[1])
                x2, y2 = min(w, bbox[2]), min(h, bbox[3])
                cropped = frame[y1:y2, x1:x2]
                if cropped.size > 0 and (x2 > x1) and (y2 > y1):
                    cv2.imwrite(os.path.join(DB_DIR, f"{label}.jpg"), cropped)
                else:
                    print(f"Warning: empty crop for {label}, not saving.")
            # Draw bounding box and label
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
            cv2.putText(frame, label, (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.imshow("Live Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()