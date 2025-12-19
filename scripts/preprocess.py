import os
import io
import boto3
import csv
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
LOCAL_BASE_DIR = "/tmp/dataset"

os.makedirs(LOCAL_BASE_DIR, exist_ok=True)

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

    matrix = np.asarray(image, dtype=np.float32) / 255.0
    return matrix  # (224, 224, 3)

# -----------------------------
# Store Matrix Dataset
# -----------------------------
def build_matrix_dataset():
    counter = 0

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

                counter += 1
                sample_name = f"{category[:-1]}_{counter:04d}"

                local_sample_dir = os.path.join(LOCAL_BASE_DIR, sample_name)
                os.makedirs(local_sample_dir, exist_ok=True)

                # Download image
                response = s3.get_object(Bucket=RAW_BUCKET, Key=key)
                image_bytes = response["Body"].read()

                matrix = image_to_matrix(image_bytes)

                # -----------------------------
                # Save matrix as CSV (2D rows)
                # -----------------------------
                matrix_path = os.path.join(local_sample_dir, "image_matrix.csv")
                with open(matrix_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    for row in matrix:
                        writer.writerow(row.flatten().tolist())

                # Save label
                label_path = os.path.join(local_sample_dir, "label.txt")
                with open(label_path, "w") as f:
                    f.write(str(label))

                # Upload to S3
                s3.upload_file(
                    matrix_path,
                    PROCESSED_BUCKET,
                    f"image_matrix.csv"
                )
                s3.upload_file(
                    label_path,
                    PROCESSED_BUCKET,
                    f"label.txt"
                )

                print(f"✔ Stored matrix for {key}")

    print("✅ Matrix-based dataset created successfully")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    build_matrix_dataset()
