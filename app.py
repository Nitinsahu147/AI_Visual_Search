import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Load pre-trained MobileNetV2
# -----------------------------
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# -----------------------------
# Load dataset features & metadata
# -----------------------------
features = np.load("features.npy")  # Precomputed dataset features
df = pd.read_csv("metadata.csv")    # Your dataset metadata (image paths, names, etc.)

# Fit Nearest Neighbors model on precomputed features
nn_model = NearestNeighbors(n_neighbors=5, metric="cosine")
nn_model.fit(features)

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

def extract_features(img_path):
    """Extract feature vector from uploaded image using MobileNetV2"""
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = model.predict(x)
    return feat.flatten()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    try:
        # Extract features
        query_feat = extract_features(filepath)

        # Find similar images
        distances, indices = nn_model.kneighbors([query_feat])
        results = df.iloc[indices[0]].to_dict(orient="records")

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
