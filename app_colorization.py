import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import tempfile
import os
import numpy as np

# Define the ColorizationNet class
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.middle = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.output_conv = nn.Conv2d(64, 3, kernel_size=1)  # RGB output

    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        # Middle
        middle = self.middle(encoded)
        # Decoder
        decoded = self.decoder(middle)
        # Output
        out = self.output_conv(decoded)
        return out

# Load the model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r"D:\python\Project_Deep Learning\model_full.pth"

    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please ensure the file is in the correct directory.")
        st.stop()

    model = None
    try:
        model = torch.load(model_path, map_location=device)
        model.eval()
        model.to(device)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    if model is None:
        st.error("Failed to load the model. Please check the model file.")
        st.stop()

    return model

model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image processing functions
transform_image = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Streamlit app
st.title("Image and Video Colorization App")

option = st.sidebar.selectbox("Select file type:", ("Image", "Video"))

if option == "Image":
    uploaded_file = st.file_uploader("Upload a grayscale image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the image
        img = Image.open(uploaded_file)

        # Display the original image
        st.image(img, caption="Original Grayscale Image", use_column_width=True)

        # Convert the image to a tensor and process it
        img_tensor = transform_image(img).unsqueeze(0).to(device)

        # Colorize the image using the model
        with torch.no_grad():
            colorized_tensor = model(img_tensor)

        # Convert the colorized tensor to an image and display it
        colorized_img = transforms.ToPILImage()(colorized_tensor.squeeze(0).cpu())
        st.image(colorized_img, caption="Colorized Image", use_column_width=True)

elif option == "Video":
    uploaded_file = st.file_uploader("Upload a grayscale video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name

        st.video(temp_video_path)

        clip = VideoFileClip(temp_video_path)

        try:
            # Process and colorize video frames
            colorized_frames = []
            transform_frame = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

            for frame in clip.iter_frames():
                frame_tensor = transform_frame(frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    colorized_frame = model(frame_tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
                    colorized_frame = (colorized_frame * 255).astype(np.uint8)
                    colorized_frames.append(colorized_frame)

            # Create a new video with the colorized frames
            output_path = "output_colored_video.mp4"
            fps = clip.fps
            colorized_clip = ImageSequenceClip(colorized_frames, fps=fps)
            colorized_clip.write_videofile(output_path, codec="libx264")

            st.success("Video colorization complete! You can download it below.")
            with open(output_path, "rb") as f:
                st.download_button("Download Colorized Video", f, file_name="colored_video.mp4")
        finally:
            clip.close()

# Clean up temporary files
if "temp_video_path" in locals():
    try:
        os.remove(temp_video_path)
    except PermissionError:
        st.warning("Temporary video file could not be deleted. It might still be in use.")
if "output_path" in locals():
    try:
        os.remove(output_path)
    except PermissionError:
        st.warning("Output video file could not be deleted. It might still be in use.")
