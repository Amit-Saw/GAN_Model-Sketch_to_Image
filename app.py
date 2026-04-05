import os
import base64

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from utils.inference import generate_image_from_sketch_bytes

load_dotenv()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", 10 * 1024 * 1024))


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
def index():
    return render_template("index.html", generated_image_url=None, error=None)


@app.post("/predict")
def predict():
    file = request.files.get("sketch")
    if not file or file.filename == "":
        return jsonify({"ok": False, "error": "Please upload a sketch image."}), 400

    if not allowed_file(file.filename):
        return jsonify({"ok": False, "error": "Unsupported file type. Use PNG, JPG, JPEG, or WEBP."}), 400

    try:
        output_bytes = generate_image_from_sketch_bytes(file.read())
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Inference failed: {exc}"}), 500

    encoded = base64.b64encode(output_bytes).decode("ascii")
    generated_image_url = f"data:image/png;base64,{encoded}"
    return jsonify({"ok": True, "generated_image_url": generated_image_url})


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
