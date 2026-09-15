import gc
import os
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired
from PIL import Image
from torchvision import transforms
import io

# AdaIN model components: VGG encoder/decoder architecture and the AdaIN transfer utilities
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std


app = Flask(__name__)
# SECRET_KEY signs Flask-WTF's session/CSRF tokens. In any real deployment this
# must come from the SECRET_KEY environment variable (set it in Render's
# Environment Variables) -- it is intentionally not hardcoded here anymore. The
# fallback value below is used only when no SECRET_KEY is set in the environment
# (e.g. quick local development) and is not a production secret.
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-only-insecure-key-do-not-use-in-production')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
# Reject grossly oversized request bodies (e.g. a multi-hundred-MB raw photo
# or a decompression-bomb-style file) before Flask even buffers them into
# memory. safe_open_image() below still downsizes anything up to this limit,
# so 10MB is purely a hard backstop against pathological uploads, not the
# normal expected file size.
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB per request
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Production-safe image size limits -----------------------------------
# Two separate caps, for two separate reasons:
#
# MAX_UPLOAD_DIMENSION bounds how large a decoded upload is ever allowed to
# get in memory, regardless of what the model needs. Without this, a photo
# with a huge pixel count (a modern phone panorama, a scanned image, etc.)
# would be fully decoded to its native resolution by PIL before any resizing
# ever ran -- an 8000x6000 photo alone is ~140MB as a raw RGB buffer, and
# decoding both the content and style image that way can approach Render's
# 512MB ceiling before the model has even run once.
#
# INFERENCE_SIZE bounds what actually gets fed to the model. Render's free
# tier previously OOM-crashed repeatedly at 512px (see Render's own event
# log: "Ran out of memory (used over 512MB) while running your code").
# 256px was profiled on 2026-09-14 as the resolution that reliably fits
# within the 512MB limit alongside PyTorch's own baseline memory footprint,
# and was confirmed again in production on 2026-09-15 after a git-history
# cleanup accidentally reverted this file to the older 512px version and the
# OOM crashes came back immediately. Keep this in sync with any future
# profiling -- do not raise it without re-measuring peak RSS on a
# Render-equivalent memory budget.
MAX_UPLOAD_DIMENSION = 1536
INFERENCE_SIZE = 256


class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Single sync Gunicorn worker (see Procfile) -- pin PyTorch to one thread too,
# so a single request doesn't spin up a pool of CPU threads that briefly
# inflates peak memory/CPU on Render's smallest instance for no throughput
# benefit (there's never more than one request in flight at a time here).
torch.set_num_threads(1)

# The VGG encoder (~77MB checkpoint) and AdaIN decoder (~14MB checkpoint) are
# intentionally NOT loaded here at module import time. Gunicorn imports this
# module (`app:app`) before it can bind to $PORT, so importing it must stay
# cheap -- eagerly loading ~90MB+ of weights on top of PyTorch's own resident
# memory here was pushing the process over Render's 512MiB limit before it
# ever finished starting up. Models are loaded lazily, once, on first actual
# use (see load_models()), and the same objects are reused for every
# subsequent request.
_encoder = None
_decoder = None


def load_models():
    """Load (once) and cache the VGG encoder and AdaIN decoder.

    Safe under a single sync Gunicorn worker (WEB_CONCURRENCY=1): a sync
    worker processes one request at a time within its process, so this
    simple "load if not loaded yet" check needs no additional locking.
    """
    global _encoder, _decoder
    if _encoder is None or _decoder is None:
        _encoder = VGGEncoder('vgg_normalised.pth').to(device)
        _decoder = Decoder().to(device)
        _decoder.load_state_dict(
            torch.load('experiment/final_exp/decoder_final.pth', map_location=device)
        )
        _encoder.eval()
        _decoder.eval()
        gc.collect()
    return _encoder, _decoder

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def safe_open_image(path, max_dim=MAX_UPLOAD_DIMENSION):
    """Open an image file bounded to at most max_dim on its longer side.

    Uses JPEG "draft" mode where the format supports it, which asks libjpeg
    to decode directly at a reduced resolution instead of decoding the full
    native resolution and only *then* resizing -- for a large photo this
    avoids ever materializing the full-size pixel buffer in memory at all.
    draft() is a documented no-op for formats that don't support it (e.g.
    PNG), so it's safe to call unconditionally.
    """
    image = Image.open(path)
    try:
        image.draft('RGB', (max_dim, max_dim))
    except Exception:
        pass
    image = image.convert('RGB')
    if max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim), Image.LANCZOS)
    return image


def resize_max_side(image, max_size):
    """Resize so the LONGER side is at most max_size, preserving aspect ratio.

    torchvision's transforms.Resize(N) only bounds the *shorter* side, so an
    extreme-aspect-ratio image (a panorama, a cropped screenshot, etc.) could
    still produce a very large tensor on its long side even after that
    resize. Bounding the longer side instead keeps worst-case tensor size
    predictable no matter the input's aspect ratio.
    """
    width, height = image.size
    longer_side = max(width, height)
    if longer_side <= max_size:
        return image
    scale = max_size / float(longer_side)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_image = resize_max_side(content_image, INFERENCE_SIZE)
    style_image = resize_max_side(style_image, INFERENCE_SIZE)

    to_tensor = transforms.ToTensor()
    content_image = to_tensor(content_image).unsqueeze(0).to(device)
    style_image = to_tensor(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)



@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(e):
    form = UploadForm()
    return render_template(
        'index.html', form=form, result_image=None, content_image=None,
        style_image=None,
        error='That file is too large (max 10MB per image). Please upload a smaller image.',
    ), 413


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename
        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
        else:
            style_filename = form.style_path.data

        # content/style FileFields intentionally have no InputRequired
        # validator, so a previously-uploaded image can be reused via the
        # content_path/style_path hidden fields without re-selecting a file.
        # That means validate_on_submit() can be True even when neither a
        # new file nor a carried-over hidden path is present -- so the
        # "missing image" check has to happen explicitly here rather than
        # relying on form validation to catch an empty submission.
        if not content_filename:
            error = 'Please upload content image'
        elif not style_filename:
            error = 'Please upload style image'
        else:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

            try:
                content_image = safe_open_image(content_path)
                style_image = safe_open_image(style_path)

                alpha = float(form.alpha.data)
                encoder, decoder = load_models()
                stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)

                result_filename = 'stylized_' + content_filename
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized_image, result_path)

                # Release request-scoped image/tensor objects promptly rather
                # than waiting for the request to finish returning, and force
                # a collection pass -- CPython's refcounting frees most of
                # this immediately, but gc.collect() also clears any
                # reference cycles PyTorch's autograd/tensor machinery can
                # leave behind, so peak memory doesn't creep up across
                # requests on a long-lived single worker process.
                del content_image, style_image, stylized_image
                gc.collect()

                result_image = result_filename
            except Exception as e:
                error = str(e)
                gc.collect()
    elif request.method == 'POST':
        # form.validate_on_submit() was False on an actual POST submission
        # (e.g. a disallowed file extension) -- a plain GET (first page load)
        # never reaches this branch, so `error` stays None on first load.
        error = 'Please upload a valid content and style image (png, jpg, or jpeg).'

    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename,
                           style_image=style_filename, error=error)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    # Only used for local `python app.py` runs -- production (Render) serves
    # via Gunicorn per the Procfile, which never executes this block. The
    # interactive Werkzeug debugger is opt-in via FLASK_DEBUG=1 so it can
    # never be on by accident.
    debug = os.environ.get('FLASK_DEBUG') == '1'
    run_simple('localhost', 5000, app, use_reloader=debug, use_debugger=debug)
