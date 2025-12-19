import os
import boto3
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Configuration
# -----------------------------
PROCESSED_BUCKET = os.getenv("PROCESSED_BUCKET")
MODEL_BUCKET = os.getenv("MODEL_BUCKET")          # optional
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")

DATASET_KEY = "dataset.npz"
LOCAL_TMP_DIR = "/tmp"
LOCAL_DATASET_PATH = f"{LOCAL_TMP_DIR}/dataset.npz"
LOCAL_MODEL_PATH = f"{LOCAL_TMP_DIR}/model.h5"

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

if not all([PROCESSED_BUCKET, AWS_REGION]):
    raise EnvironmentError("Missing required environment variables")

# -----------------------------
# AWS Client
# -----------------------------
s3 = boto3.client("s3", region_name=AWS_REGION)

# -----------------------------
# Download Dataset
# -----------------------------
def download_dataset():
    print("⬇️ Downloading dataset from S3...")
    s3.download_file(
        PROCESSED_BUCKET,
        DATASET_KEY,
        LOCAL_DATASET_PATH
    )
    print("✅ Dataset downloaded")

# -----------------------------
# Load Matrices
# -----------------------------
def load_data():
    with np.load(LOCAL_DATASET_PATH) as data:
        X = data["X"]
        y = data["y"]

    print(f"Loaded X shape: {X.shape}")
    print(f"Loaded y shape: {y.shape}")

    return X, y

# -----------------------------
# Build CNN Model
# -----------------------------
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train():
    download_dataset()

    X, y = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = build_model(
        input_shape=X_train.shape[1:],
        num_classes=len(np.unique(y))
    )

    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    model.save(LOCAL_MODEL_PATH)
    print(f"✅ Model saved locally at {LOCAL_MODEL_PATH}")

    # Optional: upload trained model to S3
    if MODEL_BUCKET:
        s3.upload_file(
            LOCAL_MODEL_PATH,
            MODEL_BUCKET,
            "model.h5"
        )
        print("☁️ Model uploaded to S3")


if __name__ == "__main__":
    train()
