import os
import io
import boto3
import numpy as np
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
RAW_BUCKET = os.getenv("RAW_BUCKET")
PROCESSED_BUCKET = os.getenv("PROCESSED_BUCKET")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")

if not all([RAW_BUCKET, PROCESSED_BUCKET, AWS_REGION]):
    raise EnvironmentError("Missing required environment variables")

CATEGORIES = {
    "cats": 0,
    "dogs": 1
}

IMAGE_SIZE = (224, 224)
DATASET_KEY = "dataset/dataset.npz"
LOCAL_TMP_DIR = "/tmp"
LOCAL_DATASET_PATH = f"{LOCAL_TMP_DIR}/dataset.npz"

# -----------------------------
# AWS Client
# -----------------------------
s3 = boto3.client("s3", region_name=AWS_REGION)

# -----------------------------
# Image → Matrix
# -----------------------------
def image_to_matrix(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    # Convert to NumPy matrix
    matrix = np.asarray(image, dtype=np.float32)

    # Normalize (0–255 → 0–1)
    matrix /= 255.0
    return matrix

# -----------------------------
# Build Matrix Dataset
# -----------------------------
def build_matrix_dataset():
    X = []
    y = []

    for category, label in CATEGORIES.items():
        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(
            Bucket=RAW_BUCKET,
            Prefix=f"{category}/"
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue

                response = s3.get_object(Bucket=RAW_BUCKET, Key=key)
                image_bytes = response["Body"].read()

                matrix = image_to_matrix(image_bytes)

                X.append(matrix)
                y.append(label)

                print(f"Processed → {key}")

    # Convert lists to matrices
    X = np.stack(X)        # (N, 224, 224, 3)
    y = np.array(y)        # (N,)

    # Save as one dataset
    np.savez_compressed(
        LOCAL_DATASET_PATH,
        X=X,
        y=y
    )

    # Upload to S3
    s3.upload_file(
        LOCAL_DATASET_PATH,
        PROCESSED_BUCKET,
        DATASET_KEY
    )

    print("✅ Matrix dataset created successfully")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    build_matrix_dataset()
