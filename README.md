# Neural Art

Neural Art is a web application for **neural style transfer**: it takes a content image
and a style image and produces a new image that keeps the content's structure while
adopting the style image's visual texture and color palette. It's built around **AdaIN
(Adaptive Instance Normalization)** style transfer, implemented in PyTorch, and served
through a small Flask app.

## What it does

- You upload a content image and a style image through the browser.
- An `alpha` value controls how strongly the style is applied (0 = original content,
  1 = full style transfer).
- The app runs both images through a pretrained VGG encoder, aligns the content
  features' statistics to the style features' statistics (the AdaIN step), decodes the
  result back into an image with a trained decoder network, and displays it.

## Main functionality

- `GET/POST /` — upload form and result page (`NST_Code/app.py`, `NST_Code/templates/index.html`).
  Handles content/style image upload, runs the style transfer, and renders the stylized
  output alongside the inputs.
- `GET /uploads/<filename>` — serves uploaded and generated images.
- `GET /examples/<filename>` — serves bundled example images (`NST_Code/examples/`).

## Project components

- `NST_Code/app.py` — Flask application: routes, upload handling, and the style-transfer
  inference call.
- `NST_Code/utils/models.py` — `VGGEncoder` and `Decoder` model definitions.
- `NST_Code/utils/utils.py` — the AdaIN operation (`adaptive_instance_normalization`,
  `calc_mean_std`) and the dataset loader used for training.
- `NST_Code/train.py` — training script for the decoder network (the VGG encoder stays
  frozen/pretrained).
- `NST_Code/templates/index.html` — the upload/result web UI.
- `NST_Code/vgg_normalised.pth` — pretrained VGG encoder weights, used at inference and
  training time.
- `NST_Code/experiment/final_exp/` — a trained decoder checkpoint (`decoder_final.pth`,
  loaded by `app.py` at startup) plus sample outputs and the config log
  (`options.txt`) from the training run that produced it.
- `NST_Code/content_data/`, `NST_Code/style_data/` — sample content/style images used for
  training and experimentation.
- `Demo_IO_Images/` — example input/output pairs showing the style transfer in action.
- `code.ipynb` — an exploratory notebook that visualizes VGG feature activations across
  content and style layers.
- `requirements.txt` — Python dependencies.
- `Procfile.txt` — process definition for deploying the Flask app with `gunicorn`.

## Setup and running locally

1. Install dependencies (Python 3, PyTorch):
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app from inside `NST_Code/`, since `app.py` loads its model weights
   (`vgg_normalised.pth`, `experiment/final_exp/decoder_final.pth`) using paths relative
   to that directory:
   ```bash
   cd NST_Code
   python app.py
   ```
   This starts a local development server at `http://localhost:5000`.

## Dependencies

From `requirements.txt`: Flask, Flask-Bootstrap, Flask-WTF, NumPy, Pillow, PyTorch,
torchvision, tqdm, Werkzeug, WTForms, and gunicorn (for production deployment).
