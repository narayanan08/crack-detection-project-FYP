import cv2
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

# Load your trained YOLO model
model = YOLO("best.pt")  # Replace with your actual model path

def detect_cracks(image):
    """Detect cracks using Canny edge detection and draw contours in red."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = image.copy()
    cv2.drawContours(output_image, contours, -1, (0, 0, 255), 2)

    return output_image, edges

def estimate_crack_depth(image):
    """Estimate crack depth based on intensity shadows (dark = deep)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized_image = cv2.equalizeHist(blurred_image)

    _, shadow_mask = cv2.threshold(equalized_image, 60, 255, cv2.THRESH_BINARY_INV)
    shadow_region = cv2.bitwise_and(equalized_image, equalized_image, mask=shadow_mask)

    depth_estimation = 255 - shadow_region
    depth_normalized = cv2.normalize(depth_estimation, None, 0, 255, cv2.NORM_MINMAX)

    return depth_normalized

def run_yolo(image):
    """Run YOLO model and return results."""
    results = model(image)
    return results

# Streamlit UI
st.title("Concrete Crack Detection and Depth Estimation")

uploaded_file = st.file_uploader("Upload a concrete image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image = np.array(pil_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = run_yolo(image)
    st.image(results[0].plot(), caption="YOLO Detection Output", use_column_width=True)

    # Check for cracks (class index 4)
    crack_detected = any(
        int(box.cls[0]) == 4 and box.conf[0].item() > 0.25
        for result in results for box in result.boxes
    )

    if crack_detected:
        st.success("Crack detected!")

        # Canny + contour highlight
        highlighted_image, canny_edges = detect_cracks(image)
        highlighted_image_rgb = cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB)
        st.image(highlighted_image_rgb, caption="Canny Edge Detection + Crack Contours", use_column_width=True)

        # Display raw Canny edges
        st.image(canny_edges, caption="Raw Canny Edges", channels="GRAY", use_column_width=True)

        # Depth estimation
        depth_map = estimate_crack_depth(image)
        fig, ax = plt.subplots()
        im = ax.imshow(depth_map, cmap='jet', interpolation='none')
        ax.set_title('Estimated Crack Depth Map')
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)

    else:
        st.warning("No crack detected (class 4 not found). Other defects may be present.")
