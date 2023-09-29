import asyncio
import websockets
import cv2
import numpy as np
import base64
from keras_facenet import FaceNet
import joblib
import io
from PIL import Image

# Load pre-trained FaceNet model
facenet_model = FaceNet()

# data = 'data_master_terbaru.pkl'
# data_master = joblib.load(data)

# Nama file model yang telah disimpan sebelumnya
knn_model = 'new_knn_model.pkl'

#model_anti_spoofing = load_model('model_antispoof.h5')

# Memuat model dari file
best_model_knn = joblib.load(knn_model)

# load cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

threshold_knn = 0.72

async def get_embedding(img):

    binary_data = base64.b64decode(img.split(',')[1])

    # Convert binary data to numpy array
    image_np = np.array(Image.open(io.BytesIO(binary_data)))
    
    # Detect faces
    wajah = face_cascade.detectMultiScale(image_np, 1.1, 4)

    # If no face is detected, skip to next image
    if len(wajah) == 0:
        return None

    # Extract face region
    x1, y1, width, height = wajah[0]
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image_np[y1:y2, x1:x2]

    # Resize image to (160, 160)
    img = cv2.resize(face, (160, 160))

    # Convert image to tensor
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    # Prediksi kelas gambar
    # pred = model_anti_spoofing.predict(img)

    # Embed face using model
    embedding = facenet_model.embeddings(img)[0, :]

    return embedding

async def get_label(frame):
    vector = await get_embedding(frame)

    # Check if embedding is None
    if vector is None:
        label_knn = "Tidak Terdeteksi"
        score_knn = 0
    else:
        # cosine_similarities = np.dot(data_master['embeddings'], vector) / (np.linalg.norm(data_master['embeddings'], axis=1) * np.linalg.norm(vector))
        # nearest_index = np.argmax(cosine_similarities)
        # label = data_master['labels'][nearest_index]
        # score = cosine_similarities[nearest_index]
        # if score < 0.75:
        #     label = "Tidak Terdaftar"

        vector = vector.reshape(1, -1)

        y_pred_knn = best_model_knn.predict(vector)

        # Calculate distances to nearest neighbors used for prediction
        distances, _ = best_model_knn.kneighbors(vector)

        # Apply threshold_knn to predictions and assign new labels
        for i, pred_label in enumerate(y_pred_knn):
            score_knn = distances[i][0]
            if distances[i][0] > threshold_knn:
                label_knn = "Tidak Terdaftar"  # Assign a new label if above threshold
            else:
                label_knn = pred_label

    return label_knn, score_knn

async def echo(websocket, path):
    try:
        async for message in websocket:
            print(f"Menerima data gambar")

            # Lakukan pra-pemrosesan gambar
            label_knn, a = await get_label(message)  # Pemanggilan fungsi get_label
            print(f"Label: {label_knn}, Score: {a}")

            # Periksa apakah koneksi masih terbuka sebelum mengirim pesan
            if not websocket.closed:
                await websocket.send(label_knn)

    except websockets.exceptions.ConnectionClosedOK:
        # Tangani kasus di mana koneksi ditutup dengan benar
        print("Koneksi ditutup dengan benar.")
    except Exception as e:
        # Tangani pengecualian lainnya
        print(f"Error: {e}")
    finally:
        # Tambahkan kode pembersihan jika diperlukan
        pass

start_server = websockets.serve(echo, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()