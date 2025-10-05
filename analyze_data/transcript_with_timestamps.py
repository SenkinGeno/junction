from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu")  

def get_transcript_with_timestamps(audio_path):
    segments, info = model.transcribe(audio_path, word_timestamps=True, language="en")

    words = []  # each: {"i": int, "word": str, "start": float, "end": float}
    for seg in segments:
        for w in seg.words or []:
            if w.start is None or w.end is None: 
                continue
            words.append({"i": len(words), "word": w.word, "start": float(w.start), "end": float(w.end)})

    return words
