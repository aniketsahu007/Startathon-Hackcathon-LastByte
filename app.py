import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from model import get_model


st.set_page_config(
    page_title="AI Adventure Mapping App",
    page_icon="🏕️",
    layout="wide"
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    model = get_model(num_classes=6)
    model.load_state_dict(
        torch.load("checkpoints/best_model.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model

model = load_model()


def preprocess(image):
    image = np.array(image).astype(np.float32)
    original_size = image.shape[:2]

    image = cv2.resize(image, (512, 512))

    image = image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

    return image, original_size



TERRAIN_LABELS = {
    0: "Background",
    1: "Walkable Path",
    2: "Rocky Zone",
    3: "Vegetation",
    4: "Rough Terrain",
    5: "Obstacle Area"
}

# -------------------------------------------------
# COLOR MAP
# -------------------------------------------------
COLORS = np.array([
    [0, 0, 0],        # Background
    [0, 255, 0],      # Walkable Path (Green)
    [255, 0, 0],      # Rocky Zone (Red)
    [0, 0, 255],      # Vegetation (Blue)
    [255, 255, 0],    # Rough Terrain (Yellow)
    [255, 0, 255],    # Obstacle Area (Purple)
], dtype=np.uint8)


def decode_mask(mask):
    return COLORS[mask]


# -------------------------------------------------
# OVERLAY
# -------------------------------------------------
def overlay_mask(image, mask_rgb):
    return cv2.addWeighted(image, 0.6, mask_rgb, 0.4, 0)


page = st.sidebar.radio(
    "Navigation",
    ["Home", "Analyze Terrain", "About"]
)


if page == "Home":

    st.title(" AI Adventure Mapping System")

    st.markdown("""
    An AI-powered terrain analysis tool for hikers, explorers, and off-road adventurers.

    Upload a terrain image and the system will automatically:
    - Identify walkable paths
    - Detect rocky zones
    - Highlight vegetation
    - Mark rough or unsafe terrain
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Model Type", "UNet ")
        st.metric("Best Validation IoU", "68%")

    with col2:
        st.metric("Input Resolution", "512 × 512")
        st.metric("Framework", "PyTorch + Streamlit")

    st.success("Navigate to 'Analyze Terrain' to try the demo.")



elif page == "Analyze Terrain":

    st.title(" Terrain Analyzer")

    uploaded_file = st.file_uploader(
        "Upload terrain image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Terrain")
            st.image(image_np, width="stretch")

        input_tensor, _ = preprocess(image)

        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        colored_mask = decode_mask(prediction)

        colored_mask = cv2.resize(
            colored_mask,
            (image_np.shape[1], image_np.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        overlay = overlay_mask(image_np, colored_mask)

        with col2:
            st.subheader("AI Terrain Map")
            st.image(colored_mask, width="stretch")

        st.markdown("---")
        st.subheader("Overlay Visualization")
        st.image(overlay, width="stretch")

 
        st.markdown("### Detected Terrain Zones")

        detected_classes = np.unique(prediction)

        for cls in detected_classes:
            if cls in TERRAIN_LABELS:
                st.write(f"✓ {TERRAIN_LABELS[cls]}")



elif page == "About":

    st.title(" About the Project")

    st.markdown("""
    ### Architecture Flow

    User → Web Interface → Segmentation Model → Terrain Map → User

    ### Technical Details

    - Model: Custom UNet
    - Loss: CrossEntropy + Dice Loss
    - Optimizer: AdamW
    - Evaluation Metric: Mean IoU

    ### Real-Life Application

    This system can assist:
    - Hikers
    - Off-road drivers
    - Drone-based terrain mapping
    - Outdoor navigation planning

    It provides quick terrain awareness to improve safety and planning.
    """)

    st.success("Adventure Mapping Prototype – Hackathon Edition")
