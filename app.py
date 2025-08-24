from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

app = Flask(__name__)

# ✅ Load dataset + features
df = pd.read_csv("images.csv")
features = np.load("features.npy")

# ✅ Prepare model for query images
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=base_model.input, outputs=base_model.output)

# ✅ Kaggle images directory
DATASET_IMAGES = os.path.expanduser("~/.cache/kagglehub/datasets/paramaggarwal/fashion-product-images-small/versions/1/images")

def cosine_similarity(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    file = request.files["query_image"]
    if not file:
        return jsonify({"error": "No file uploaded"})

    file_path = "static/temp_query.jpg"
    file.save(file_path)

    # Extract query features
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    query_feat = model.predict(x, verbose=0)[0]

    # Compare with precomputed features
    sims = [cosine_similarity(query_feat, feat) for feat in features]
    top_idx = np.argsort(sims)[::-1][:5]

    results = df.iloc[top_idx].copy()
    results["similarity"] = [sims[i] for i in top_idx]

    # ✅ Send clean data
    results = results[["id", "gender", "masterCategory", "productDisplayName", "image", "similarity"]]

    return jsonify(results.to_dict(orient="records"))

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(DATASET_IMAGES, filename)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Railway sets this automatically
    app.run(host="0.0.0.0", port=port)
