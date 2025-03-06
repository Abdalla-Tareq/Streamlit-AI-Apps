import streamlit as st
import cv2
import random
from ultralytics import YOLO

# Function to handle file uploads and object detection
def detect_objects(uploaded_file, model_path):
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_file", "wb") as f:
            f.write(uploaded_file.getvalue())
        input_path = "temp_file"

        try:
            model = YOLO(model_path)  # Load your YOLO model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        try:
            # Perform object detection based on file type
            if uploaded_file.type.startswith("image"):
                img = cv2.imread(input_path)
                results = model(img)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
                        conf = box.conf[0]
                        cls = int(box.cls[0])
                        class_name = result.names[cls]

                        color = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Show the image with bounding boxes in Streamlit
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                st.image(img_rgb, channels="RGB", caption="Processed Image", use_column_width=True)
            elif uploaded_file.type.startswith("video"):
                st.video(input_path)  # Display the video in Streamlit
                # You can implement video processing here for Streamlit if needed
            else:
                st.error("Unsupported file type. Please upload an image or video.")
        except Exception as e:
            st.error(f"Error during object detection: {e}")
    else:
        st.warning("Please upload a file.")

# Streamlit app
st.title("Object Detection App")

# File uploader
uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# Model path (replace with your actual model path if needed)
model_path = "yolov9c.pt"  # Default path
# Allow users to specify model path
custom_model_path = st.text_input("Enter custom model path (optional):", "")

if custom_model_path:
    model_path = custom_model_path

if st.button("Detect Objects"):
    detect_objects(uploaded_file,model_path)