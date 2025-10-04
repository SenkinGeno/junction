import streamlit as st
import speech_recognition as sr
import os

from analyze import *


st.set_page_config(
    page_title="Audio Folder Manager",
    page_icon="🎧",
    layout="centered"
)

st.title("🎙️ Audio Folder Input & Management")
st.write(
    """
Enter the path to a folder containing audio files (.wav, .m4a).  
The app will list the files and allow you to delete them if needed.
"""
)

# ---------------------- FOLDER INPUT ----------------------
folder_path = st.text_input("Enter the path to your audio folder:")

if folder_path:
    folder_path = os.path.abspath(folder_path)

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        audio_files = [f for f in os.listdir(folder_path) if f.endswith((".wav", ".m4a"))]

        if audio_files:
            if st.button("▶️ Analyze All Audio Files"):
                st.info("Starting analysis...")

                r = sr.Recognizer()
                directory = os.fsencode(folder_path)
                for file in os.listdir(directory):
                    filename = os.fsdecode(file)
                    if filename.endswith(".wav"):
                        with sr.AudioFile(directory.decode() + "/" + filename) as source:
                            audio_data = r.record(source)

                        try:
                            text = r.recognize_google(audio_data)
                            result = analyze_data(text)
                            if len(result) > 0:
                                st.write(f"{filename}: has extremism")
                            else:
                                st.write(f"{filename}: no extremism detected")
                        except sr.UnknownValueError:
                            st.write(f"{filename}: Sorry, could not understand the audio.")
                        except sr.RequestError:
                            st.write(f"{filename}: Could not connect to Google API.")
                        continue
                    else:
                        continue

                st.success("✅ Analysis complete!")

        else:
            st.info("No audio files (.wav, .m4a) found in this folder.")
    else:
        st.error("Folder path does not exist or is not a directory.")
