from flask import Flask, request, render_template
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_url = None

    if request.method == "POST":
        image = request.files["image"]
        if image:
            image_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(image_path)
            raw_image = Image.open(image_path).convert("RGB")

            inputs = processor(raw_image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # Fix Windows path separators
            image_url = image_path.replace("\\", "/")

    return render_template("index.html", caption=caption, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
