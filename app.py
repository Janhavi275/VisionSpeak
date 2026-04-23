import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch

from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------- LOAD MODELS --------------------

@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    caption_model.to(device)

    return processor, caption_model, tokenizer, translation_model, device

processor, caption_model, tokenizer, translation_model, device = load_models()

# -------------------- COLORIZATION MODEL --------------------
import os
import urllib.request

@st.cache_resource
def load_color_model():
    os.makedirs("models", exist_ok=True)

    prototxt = "models/colorization_deploy_v2.prototxt"
    model = "models/colorization_release_v2.caffemodel"
    points = "models/pts_in_hull.npy"

    def download(url, path, min_size):
        if not os.path.exists(path) or os.path.getsize(path) < min_size:
            if os.path.exists(path):
                os.remove(path)
            urllib.request.urlretrieve(url, path)

    # ✅ Stable OpenCV links
    download(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/colorization_deploy_v2.prototxt",
        prototxt,
        10000
    )

    download(
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_colorization_20170828/colorization_release_v2.caffemodel",
        model,
        50000000
    )
    download(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/pts_in_hull.npy",
        points,
        10000
    )

    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")

    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    return net
    

def colorize_image(image):
    net = load_color_model()   

    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    L = cv2.split(lab)[0]

    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)

    return (colorized * 255).astype("uint8")



# -------------------- IMAGE ENHANCEMENT --------------------

def enhance_image(image):
    img = np.array(image)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(enhanced_img)


# -------------------- CAPTION --------------------

def generate_caption(image):
    image = image.resize((384, 384))

    inputs = processor(images=image, return_tensors="pt").to(device)
    output = caption_model.generate(**inputs)

    return processor.decode(output[0], skip_special_tokens=True)

# -------------------- TRANSLATION --------------------

def translate_caption(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = translation_model.generate(**inputs)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------- UI --------------------

st.set_page_config(page_title="VisionSpeak AI", layout="centered")

st.title("🧠 VisionSpeak AI")
st.subheader("Image Enhancement + Colorization + Captioning")

mode = st.radio("Select Mode", ["Enhancement", "Colorization"])

file = st.file_uploader("Upload Image", type=["jpg", "png"])

if file:
    try:
        image = Image.open(file).convert("RGB")

        st.subheader("📷 Original Image")
        st.image(image, use_column_width=True)

        if mode == "Enhancement":
            output_image = enhance_image(image)
        else:
            output_image = colorize_image(image)

        caption = generate_caption(image)
        hindi = translate_caption(caption)

        st.subheader("🎨 Processed Image")
        st.image(output_image, use_column_width=True)

        st.subheader("📝 Caption (English)")
        st.success(caption)

        st.subheader("🌍 Caption (Hindi)")
        st.info(hindi)

    except Exception as e:
        st.error(f"Error occurred: {e}")
