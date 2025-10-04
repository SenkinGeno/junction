import speech_recognition as sr
import csv
import os

r = sr.Recognizer()

directory = os.fsencode("AudioData")
f = open("AudioText.txt", "w")


for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".wav"):
        with sr.AudioFile("AudioData/" + filename) as source:
            audio_data = r.record(source)

        try:
            text = r.recognize_google(audio_data)
            with open("AudioText.txt", "a") as f:
                f.write(text + '\n')

        except sr.UnknownValueError:
            print("Sorry, could not understand the audio.")
        except sr.RequestError:
            print("Could not connect to Google API.")
        # print(os.path.join(directory, filename))
        continue
    else:
        continue

with open("AudioText.txt", "r") as txt_file, \
        open("AudioDataset.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)

    for line in txt_file:
        writer.writerow([line.strip()])
