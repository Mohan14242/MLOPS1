import os
import io
import csv
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
CSV_KEY = "dataset/image_dataset.csv"
LOCAL_CSV_PATH = "/tmp/image_dataset.csv"

# -----------------------------
# AWS Client
# -----------------------------
s3 = boto3.client("s3", region_name=AWS_REGION)

# -----------------------------
# Image → Flattened Matrix
# -----------------------------
def image_to_flat_matrix(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    matrix = np.asarray(image, dtype=np.float32) / 255.0
    return matrix.flatten()  # 1D vector

# -----------------------------
# Build CSV Dataset
# -----------------------------
def build_csv_matrix_dataset():
    header_written = False

    with open(LOCAL_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)

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

                    flat_pixels = image_to_flat_matrix(image_bytes)

                    # Write header once
                    if not header_written:
                        header = ["label"] + [
                            f"pixel_{i}" for i in range(len(flat_pixels))
                        ]
                        writer.writerow(header)
                        header_written = True

                    writer.writerow([label] + flat_pixels.tolist())
                    print(f"Processed → {key}")

    # Upload CSV to S3
    s3.upload_file(
        LOCAL_CSV_PATH,
        PROCESSED_BUCKET,
        CSV_KEY
    )

    print("✅ CSV matrix dataset created successfully")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    build_csv_matrix_dataset()
