import cv2
import numpy as np
import insightface
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_NAME = "buffalo_l"
THRESHOLD = 0.6
DB_DIR = "live_face_db"
os.makedirs(DB_DIR, exist_ok=True)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

def unique_embeddings(embeddings, threshold=THRESHOLD):
    """Deduplicate embeddings based on cosine similarity."""
    unique = []
    for emb in embeddings:
        if not any(cosine_sim(emb, u) > threshold for u in unique):
            unique.append(emb)
    return unique

def save_embeddings(name, embeddings):
    person_dir = os.path.join(DB_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    for i, emb in enumerate(embeddings):
        np.save(os.path.join(person_dir, f"{name}_{i}.npy"), emb)

def load_db():
    db = []
    for name in os.listdir(DB_DIR):
        person_dir = os.path.join(DB_DIR, name)
        if not os.path.isdir(person_dir):
            continue
        for file in os.listdir(person_dir):
            if file.endswith('.npy'):
                emb = np.load(os.path.join(person_dir, file))
                db.append((emb, name))
    return db

def find_match(embedding, db, threshold=THRESHOLD):
    if not db:
        return None
    sims = [cosine_sim(embedding, e) for e, _ in db]
    max_idx = np.argmax(sims)
    if sims[max_idx] > threshold:
        return db[max_idx][1]
    else:
        return None

def register_faces(app):
    cap = cv2.VideoCapture(0)
    print("Face Registration Mode")
    while True:
        name = input("\nEnter name for this session (or blank to finish registration): ").strip()
        if not name:
            break
        print(f"Recording for '{name}'. Press 's' to save, 'q' to discard and re-enter name.")
        session_embeddings = []
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera Error")
                break
            faces = app.get(frame)
            for face in faces:
                bbox = face.bbox.astype(int)
                emb = face.embedding
                # Add only unique embeddings for this session
                if not any(cosine_sim(emb, e) > THRESHOLD for e in session_embeddings):
                    session_embeddings.append(emb)
                # Draw
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
                cv2.putText(frame, name, (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.imshow("Face Registration", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('s'):
                unique_embs = unique_embeddings(session_embeddings)
                save_embeddings(name, unique_embs)
                print(f"Saved {len(unique_embs)} unique faces for '{name}'")
                break
            elif k == ord('q'):
                print(f"Discarded session for '{name}'")
                break
        cv2.destroyAllWindows()
    cap.release()

def recognize_faces(app, db):
    cap = cv2.VideoCapture(0)
    print("\nRecognition Mode: Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = app.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            emb = face.embedding
            label = find_match(emb, db)
            if label is None:
                label = "Unknown"
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
            cv2.putText(frame, label, (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    app = insightface.app.FaceAnalysis(name=MODEL_NAME, providers=['CUDAExecutionProvider']) #remove the CUDA THINGY to make it run on any dev
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Registration
    register_faces(app)

    # Load DB for recognition
    db = load_db()
    print(f"\nDatabase loaded with {len(db)} faces.")

    # Offer recognition mode
    while True:
        use_recog = input("\nRun recognition with current DB? (y/n): ").strip().lower()
        if use_recog == 'y':
            recognize_faces(app, db)
            break
        elif use_recog == 'n':
            print("Exiting.")
            break

if __name__ == "__main__":
    main()