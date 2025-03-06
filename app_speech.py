import streamlit as st
import whisper
import os
from pydub import AudioSegment

AudioSegment.ffmpeg = r"C:\PATH\ffmpeg.exe"


st.set_page_config(
    page_title="Whisper APP",
    layout="wide",
    initial_sidebar_state="auto",
)


upload_path = "uploads/"
download_path = "downloads/"
transcript_path = "transcripts/"


os.makedirs(upload_path, exist_ok=True)
os.makedirs(download_path, exist_ok=True)
os.makedirs(transcript_path, exist_ok=True)


audio_tags = {'comments': 'Converted using pydub!'}


@st.cache_data(show_spinner=True)
def to_mp3(audio_file, output_audio_file, upload_path, download_path):
    try:
        ext = audio_file.name.split('.')[-1].lower()  
        input_path = os.path.join(upload_path, audio_file.name)  
        output_path = os.path.join(download_path, output_audio_file)  
        audio_data = None

        if ext == "wav":
            audio_data = AudioSegment.from_wav(input_path)
        elif ext == "mp3":
            audio_data = AudioSegment.from_mp3(input_path)
        elif ext == "ogg":
            audio_data = AudioSegment.from_ogg(input_path)
        elif ext == "wma":
            audio_data = AudioSegment.from_file(input_path, "wma")
        elif ext == "aac":
            audio_data = AudioSegment.from_file(input_path, "aac")
        elif ext == "flac":
            audio_data = AudioSegment.from_file(input_path, "flac")
        elif ext == "flv":
            audio_data = AudioSegment.from_flv(input_path)
        elif ext == "mp4":
            audio_data = AudioSegment.from_file(input_path, "mp4")
        else:
            raise ValueError("Unsupported file format!")  

        audio_data.export(output_path, format="mp3", tags=audio_tags)
        return output_audio_file
    except Exception as e:
        st.error(f"Error while converting to MP3: {e}")
        return None


@st.cache_resource(show_spinner=True)
def process_audio(filename, model_type):
    try:
        model = whisper.load_model(model_type)  
        result = model.transcribe(filename)  
        return result["text"]
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None


@st.cache_data(show_spinner=True)
def save_transcript(transcript_data, txt_file):
    try:
        with open(os.path.join(transcript_path, txt_file), "w") as f:
            f.write(transcript_data)  
    except Exception as e:
        st.error(f"Error while saving transcript: {e}")


def clean_directory(path):
    try:
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))  
    except Exception as e:
        st.warning(f"Could not clean directory {path}: {e}")


st.title("Speech Recognition using Whisper")
st.info('Supports all popular audio formats - WAV, MP3, MP4, OGG, WMA, AAC, FLAC, FLV')
uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg", "wma", "aac", "flac", "mp4", "flv"])

if uploaded_file is not None:
    
    input_path = os.path.join(upload_path, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  

    
    st.write(f"Input file path: {input_path}")

    
    with st.spinner("Processing Audio..."):
        output_audio_file = uploaded_file.name.split('.')[0] + '.mp3'
        output_audio_file = to_mp3(uploaded_file, output_audio_file, upload_path, download_path)

    
    st.write(f"Output file path: {os.path.join(download_path, output_audio_file)}")

    if output_audio_file:
        
        audio_file_path = os.path.join(download_path, output_audio_file)
        
        if os.path.exists(audio_file_path):
            with open(audio_file_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            st.audio(audio_bytes)  

            
            whisper_model_type = st.radio("Choose Whisper Model:", ('Tiny', 'Base', 'Small', 'Medium', 'Large'))

            
            if st.button("Generate Transcript"):
                with st.spinner("Generating Transcript..."):
                    transcript = process_audio(audio_file_path, whisper_model_type.lower()) 
                    if transcript:
                        output_txt_file = output_audio_file.split('.')[0] + ".txt"
                        save_transcript(transcript, output_txt_file)  
                        
                        
                        transcript_path_full = os.path.join(transcript_path, output_txt_file)
                        with open(transcript_path_full, "r") as file:
                            transcript_data = file.read()
                        
                        st.text_area("Transcript", transcript_data, height=300)  
                        st.download_button("Download Transcript", transcript_data, file_name=output_txt_file) 
                    else:
                        st.error("Error: Failed to generate transcript!")
        else:
            st.error("Error: The converted audio file was not found!")