import kagglehub
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tqdm import tqdm

# ✅ Download Kaggle dataset
dataset_path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
print("Dataset saved to:", dataset_path)

# CSV path
csv_path = os.path.join(dataset_path, "styles.csv")

# ✅ Load dataset
df = pd.read_csv(csv_path, on_bad_lines="skip")

# ✅ Add image filename
df["image"] = df["id"].astype(str) + ".jpg"

# ✅ Save smaller sample (use full later if needed)
sample = df.head(1000).copy()
sample[["id", "gender", "masterCategory", "productDisplayName", "image"]].to_csv("images.csv", index=False)
print("✅ Saved images.csv with metadata + image paths")

# ✅ Precompute features
img_dir = os.path.join(dataset_path, "images")

base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=base_model.input, outputs=base_model.output)

features = []

for _, row in tqdm(sample.iterrows(), total=len(sample)):
    img_path = os.path.join(img_dir, row["image"])
    if not os.path.exists(img_path):
        features.append(np.zeros(2048))  # placeholder
        continue

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = model.predict(x, verbose=0)[0]
    features.append(feat)

features = np.array(features)
np.save("features.npy", features)
print("✅ Saved features.npy")
