# Sketch to Image Flask App

This project serves a simple web UI where users upload a sketch and receive a generated image using `assets/generator.pt`.

## Folder structure

- `app.py` Flask app and routes
- `utils/inference.py` model loading and inference logic
- `templates/index.html` upload UI
- `assets/` model and notebook artifacts
- `requirements.txt` Python dependencies
- `Procfile` process entry for PaaS deployment
- `.env.example` deployment/runtime environment variables

## Image handling

- Uploaded sketches and generated outputs are processed in memory.
- No input/output image files are persisted on disk.

## Local run

1. Activate environment

```bat
.venv\Scripts\activate.bat
```

2. Install dependencies

```bat
pip install -r requirements.txt
```

3. Run the app

```bat
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Deployment notes

- Keep `assets/generator.pt` available or set `MODEL_PATH` in env.
- For cloud deployment, use `gunicorn` with `Procfile`.
- Set `FLASK_DEBUG=0` in production.

## Datasets & Resources

### Model Training Dataset
- [Edge 2 Real Image](https://www.kaggle.com/datasets/arsyadmuhammad/edge-2-real-image/data) - Dataset used for training the GAN model

### Additional Resources
- [TU Berlin Hand Sketch Image Dataset](https://www.kaggle.com/datasets/zara2099/tu-berlin-hand-sketch-image-dataset?select=TUBerlin) - Sketch dataset for reference
- [130K Images 512x512 Universal Image Embeddings](https://www.kaggle.com/datasets/rhtsingh/130k-images-512x512-universal-image-embeddings?resource=download) - Image dataset resource
