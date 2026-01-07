import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Injury Detection System",
    layout="centered"
)

st.title("🩹 Injury Detection System")
st.write(
    "Upload an image of a body part to check whether an external injury is present. "
    "The model also provides a visual explanation (Grad-CAM)."
)

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Image preprocessing (same as training)
# -------------------------------
IMG_SIZE = 224

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    weights = MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Binary classifier
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )

    # Load trained weights
    model.load_state_dict(
        torch.load("model/injury_model.pth", map_location=device)
    )

    model.to(device)
    model.eval()
    return model

model = load_model()

# -------------------------------
# Grad-CAM implementation
# -------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

# Initialize Grad-CAM
target_layer = model.features[-1][0]
gradcam = GradCAM(model, target_layer)

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    # Prediction
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)

    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_class].item()

    class_names = ["Normal", "Injury"]
    prediction = class_names[pred_class]

    st.markdown("---")
    st.subheader("Prediction")

    if prediction == "Injury":
        st.error(f"🩹 **Injury Detected**\n\nConfidence: **{confidence*100:.2f}%**")
    else:
        st.success(f"✅ **No Injury Detected**\n\nConfidence: **{confidence*100:.2f}%**")

    # -------------------------------
    # Grad-CAM generation
    # -------------------------------
    cam = gradcam.generate(input_tensor, pred_class)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = 1 - cam  # invert so hot = important

    # Prepare original image
    orig_img = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0

    # Resize CAM
    heatmap = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    heatmap = heatmap / 255.0

    overlay = 0.6 * heatmap + 0.4 * orig_img

    # -------------------------------
    # Display Grad-CAM
    # -------------------------------
    st.markdown("---")
    st.subheader("Model Explanation (Grad-CAM)")

    col1, col2 = st.columns(2)

    with col1:
        st.image(orig_img, caption="Original Image", use_column_width=True)

    with col2:
        st.image(
            overlay,
            caption="Grad-CAM (Hot regions influenced the prediction)",
            use_column_width=True
        )

    st.info(
        "⚠️ This system is intended as a **screening and decision-support tool**, "
        "not as a medical diagnosis. Always consult a medical professional."
    )
