import os
import io
import csv
import boto3
from PIL import Image

# -----------------------------
# Configuration (ENV VARS)
# -----------------------------
RAW_BUCKET = os.getenv("RAW_BUCKET")
PROCESSED_BUCKET = os.getenv("PROCESSED_BUCKET")

if not RAW_BUCKET or not PROCESSED_BUCKET:
    raise EnvironmentError(
        "❌ RAW_BUCKET and PROCESSED_BUCKET environment variables must be set"
    )

CATEGORIES = {
    "cats": 0,
    "dogs": 1
}

IMAGE_SIZE = (224, 224)

LOCAL_TMP_DIR = "/tmp/images"
os.makedirs(LOCAL_TMP_DIR, exist_ok=True)

# -----------------------------
# AWS Client
# -----------------------------
s3 = boto3.client("s3")

# -----------------------------
# Image Transformation
# -----------------------------
def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    return image

# -----------------------------
# Main Processing Logic
# -----------------------------
def process_images():
    labels = []
    image_counter = 0

    for category, label in CATEGORIES.items():
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=RAW_BUCKET,
            Prefix=f"{category}/"
        )

        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith("/"):
                    continue

                image_counter += 1
                filename = f"{category[:-1]}_{image_counter:04d}.jpg"
                local_path = os.path.join(LOCAL_TMP_DIR, filename)

                # Download image
                response = s3.get_object(Bucket=RAW_BUCKET, Key=key)
                image_bytes = response["Body"].read()

                # Transform image
                image = transform_image(image_bytes)
                image.save(local_path, format="JPEG")

                # Upload to processed bucket
                s3.upload_file(
                    local_path,
                    PROCESSED_BUCKET,
                    f"images/{filename}"
                )

                labels.append([filename, label])
                print(f"Processed {key} → images/{filename}")

    # -----------------------------
    # Save labels.csv
    # -----------------------------
    labels_path = os.path.join(LOCAL_TMP_DIR, "labels.csv")
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "label"])
        writer.writerows(labels)

    s3.upload_file(
        labels_path,
        PROCESSED_BUCKET,
        "labels.csv"
    )

    print("✅ Image processing completed successfully")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    process_images()
