# NeuralArt

Neural style transfer web application implementing Adaptive Instance Normalization
(AdaIN): upload a content image and a style image, adjust the style strength, and get a
stylized result back in the browser.

**Live Demo:** https://neuralart-r9ff.onrender.com

## Overview

NeuralArt implements **AdaIN (Adaptive Instance Normalization)** style transfer in
PyTorch, exposed through a Flask web application. It takes a content image and a style
image and produces a new image that keeps the content's structure while adopting the
style image's visual texture and color palette.

- A VGG19-based encoder, used as a frozen feature extractor and truncated at `relu4_1`,
  converts both images into feature representations.
- AdaIN aligns the content features' statistics to the style features' statistics.
- A trained decoder reconstructs the result back into an RGB image.
- An adjustable `alpha` parameter controls how strongly the style is applied.
- The application is deployed on Render (see the Live Demo link above).

This is a portfolio/research-style implementation of a published technique, not a
commercial product.

## Demo / Results

Content image styled with two different style references (`Demo_IO_Images/`):

| Content | Style | Result |
|---|---|---|
| ![Content image](Demo_IO_Images/i-p/i_p%20image.jpg) | ![Style image 1](Demo_IO_Images/i-p/style%201.png) | ![Stylized result 1](Demo_IO_Images/o-p/o_p%20style%201.jpg) |
| ![Content image](Demo_IO_Images/i-p/i_p%20image.jpg) | ![Style image 2](Demo_IO_Images/i-p/style%202.jpg) | ![Stylized result 2](Demo_IO_Images/o-p/o_p%20style%202.jpg) |

## How It Works

- **Encoder** — a VGG19 backbone, used as a frozen, pretrained feature extractor (not
  trained by this project) and truncated at `relu4_1`. It converts the content image and
  the style image into feature maps.
- **AdaIN** — aligns the channel-wise mean and standard deviation of the content feature
  map to those of the style feature map
  (`adaptive_instance_normalization` in `NST_Code/utils/utils.py`).
- **Alpha** — interpolates between the original content feature representation
  (`alpha = 0`) and the fully AdaIN-transformed representation (`alpha = 1`), controlling
  how strongly the style is applied.
- **Decoder** — a trained network, structured as a mirror of the encoder, that
  reconstructs an RGB image from the (alpha-blended) transformed feature representation.

## Architecture

```
Content Image + Style Image
            │
            ▼
  VGG Encoder (frozen, truncated at relu4_1)
            │
            ▼
          AdaIN
            │
            ▼
      Alpha Blending
            │
            ▼
     Trained Decoder
            │
            ▼
     Stylized Image
```

## Routes

- `GET/POST /` — upload form and result page (`NST_Code/app.py`, `NST_Code/templates/index.html`).
  Handles content/style image upload, runs the style transfer, and renders the stylized
  output alongside the inputs.
- `GET /uploads/<filename>` — serves uploaded and generated images.
- `GET /examples/<filename>` — serves bundled example images (`NST_Code/examples/`).

## Features

- Neural style transfer using AdaIN
- VGG19-based feature extraction
- Trained image decoder
- Adjustable style strength (`alpha`)
- Content/style image upload through the browser
- Bundled example transformations
- Flask web interface
- Deployed on Render
- Memory-conscious, lazy model loading (the VGG encoder and decoder are loaded once, on
  first request, rather than at startup, to fit within Render's memory limits)

## Tech Stack

- Python
- PyTorch / torchvision
- Flask, Flask-Bootstrap, Flask-WTF, WTForms
- Pillow
- Gunicorn (production server)
- Render (deployment)

## Project Structure

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
  lazy-loaded by `app.py` on first request, not at startup -- see the memory-loading
  note above) plus sample outputs and the config log (`options.txt`) from the
  training run that produced it.
- `NST_Code/content_data/`, `NST_Code/style_data/` — sample content/style images used for
  training and experimentation.
- `Demo_IO_Images/` — example input/output pairs showing the style transfer in action.
- `code.ipynb` — an exploratory notebook that visualizes VGG feature activations across
  content and style layers.
- `requirements.txt` — Python dependencies.
- `Procfile` — process definition for deploying the Flask app with `gunicorn`.
- `render.yaml` — documents the Render web service configuration (build/start commands,
  Python version, env vars) so the deployment is reproducible from the repo.
- `LICENSE` — MIT license.

## Installation & Local Setup

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

## Training

The decoder was trained using this project's own training pipeline (`NST_Code/train.py`):
the (frozen) VGG19 encoder provides content and style features, AdaIN produces the target
feature representation, and the decoder is optimized against a content loss (matching the
AdaIN target) plus a style loss (matching channel-wise mean/std across the encoder's
feature levels) — the loss formulation used in the AdaIN paper. The VGG19 encoder itself
is not trained by this project; it is used as a fixed, pretrained feature extractor. The
configuration and log for the training run that produced the bundled decoder checkpoint
are recorded in `NST_Code/experiment/final_exp/options.txt`.

## Limitations

- This is a portfolio/research-style implementation of a published technique, not a
  production-scale SaaS service.
- Deployed on Render's free tier, which may spin down after a period of inactivity — the
  first request after idle time can take noticeably longer while the service wakes up.
- Style-transfer inference (running a VGG encoder and a decoder network per request) is
  more computationally heavy than a typical Flask request.
- The trained model checkpoints (`vgg_normalised.pth`, `decoder_final.pth`) are large
  binary files, which makes the repository relatively large to clone.

## References

- Huang, X., & Belongie, S. (2017). *Arbitrary Style Transfer in Real-Time with Adaptive
  Instance Normalization.* ICCV 2017.
