import streamlit as st
import speech_recognition as sr
import os
import io

from analyze import *
from transcript_with_timestamps import *


st.set_page_config(
    page_title="Audio Folder Manager",
    page_icon="🎧",
    layout="centered"
)

st.title("🎙️ Audio Folder Input & Management")
if st.button("Refresh"):
    st.markdown(
        """
        <script>
        window.location.reload();
        </script>
        """,
        unsafe_allow_html=True
    )
st.write("Enter the path to a folder containing audio files (.wav, .mp4).")

# ---------------------- FOLDER INPUT ----------------------
folder_path = st.text_input("Enter the path to your audio folder:")
if st.button("▶️ Analyze All Audio Files"):
    if folder_path:
        folder_path = os.path.abspath(folder_path)

        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            audio_files = [f for f in os.listdir(folder_path) if f.endswith((".wav", ".m4a"))]

            if audio_files:
                st.info("Starting analysis...")

                r = sr.Recognizer()
                directory = os.fsencode(folder_path)
                for file in os.listdir(directory):
                    filename = os.fsdecode(file)
                    if filename.endswith(".wav"):
                        

                        try:
                            data = get_transcript_with_timestamps(os.path.join(folder_path, filename))
                            print(data)
                            result = analyze_data(data)
                            if len(result) > 0:
                                st.write(f"{filename}: has extremism at " + "".join([str(round(ts, 2)) + "s, " for ts in result]))
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
                st.info("No audio files (.wav, .m4a) found in this folder!")
        else:
            st.error("Folder path does not exist or is not a directory!")
    else:
        st.error("No folder path has been chosen!")
st.title("Or")
st.write("Upload one or more audio files (.wav, .mp4).")
uploaded_files = st.file_uploader(
    "Upload your audio files",
    type=["wav", "mp4"],
    accept_multiple_files=True
)
if st.button("▶️ Analyze"):
    if uploaded_files:
        st.info("Starting analysis...")
        r = sr.Recognizer()
        for file in uploaded_files:

            try:
                data = get_transcript_with_timestamps(io.BytesIO(file.read()))
                print(data)
                result = analyze_data(data)



                if len(result) > 0:
                    st.write(f"{file.name}: has extremism at " + "".join([str(round(ts, 2)) + "s" for ts in result]))
                else:
                    st.write(f"{file.name}: no extremism detected")
            except sr.UnknownValueError:
                st.error(f"{file.name}: Sorry, could not understand the audio.")
            except sr.RequestError:
                st.error(f"{file.name}: Could not connect to Google API.")
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

        st.success("✅ Analysis complete!")
    else:
        st.error("No files have been chosen!")