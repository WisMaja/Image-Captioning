from flask import Flask, request, render_template
import requests
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Konfiguracja Azure AI Vision
AZURE_API_KEY = "D2XN3qoofGN9weRIp69lINPjnNfX9ZHlZyHjSzDEupaZ9MtBSPodJQQJ99BFAC5RqLJXJ3w3AAAEACOG69kT"
AZURE_ENDPOINT = "https://imagecaptioningservice.cognitiveservices.azure.com/"
VISION_API_URL = AZURE_ENDPOINT + "vision/v3.2/analyze?visualFeatures=Description"

@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_url = None

    if request.method == "POST":
        image = request.files["image"]
        if image:
            image_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(image_path)

            # Wczytanie obrazu binarnie
            with open(image_path, "rb") as image_data:
                headers = {
                    "Ocp-Apim-Subscription-Key": AZURE_API_KEY,
                    "Content-Type": "application/octet-stream"
                }
                response = requests.post(VISION_API_URL, headers=headers, data=image_data)
                response.raise_for_status()
                analysis = response.json()

                # Pobranie opisu
                captions = analysis.get("description", {}).get("captions", [])
                if captions:
                    caption = captions[0]["text"]
                else:
                    caption = "Nie udało się wygenerować opisu."

            image_url = image_path.replace("\\", "/")

    return render_template("index.html", caption=caption, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
