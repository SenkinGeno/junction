from datasets import Value, concatenate_datasets, load_dataset

extremists = load_dataset("csv", data_files="datasets/generated_extremist.csv")['train']
non_extremists = load_dataset("csv", data_files="datasets/generated_non_extremist.csv")['train']
ds1 = load_dataset("tdavidson/hate_speech_offensive")['train']
ds2 = load_dataset("SetFit/hate_speech_offensive")['train']
ds3 = load_dataset("Paul/hatecheck")['test']
ds4 = load_dataset("csv", data_files="datasets/GoldStanderDataSet.csv", encoding="cp1252")['train']
ds5 = load_dataset("csv", data_files="datasets/bullydetect.csv")['train']
ds6 = load_dataset("Vinod-IE/IDMBreview")['train']
ds7 = load_dataset("csv", data_files="datasets/structure_based.csv")['train']
ds8 = load_dataset("csv", data_files="datasets/non-toxic-comments-sample.csv")['train']

ds6 = ds6.filter(lambda x: x["sentiment"] == "positive")

ds1 = ds1.rename_column("class", "label")
ds1 = ds1.rename_column("tweet", "text")

ds3 = ds3.rename_column("test_case", "text")
ds3 = ds3.rename_column("label_gold", "label")

ds4 = ds4.rename_column("Biased", "label")
ds4 = ds4.rename_column("Text", "text")

ds5 = ds5.rename_column("Comment", "text")
ds5 = ds5.rename_column("Insult", "label")

ds6 = ds6.rename_column("review", "text")
ds6 = ds6.rename_column("sentiment", "label")


ds1 = ds1.remove_columns(['count', 'hate_speech_count', 'offensive_language_count', 'neither_count'])
ds2 = ds2.remove_columns(['label_text'])
ds3 = ds3.remove_columns(['functionality', 'case_id', 'target_ident', 'direction', 'focus_words', 'focus_lemma', 'ref_case_id', 'ref_templ_id', 'case_templ', 'templ_id',] )
ds4 = ds4.remove_columns(['TweetID', 'Username', 'CreateDate', 'Keyword'])
ds6 = ds6.remove_columns(['review_length'])


def map_labels_ds1(example):
    example["label"] = 1 if example["label"] in [0, 1] else 0
    return example

def map_labels_ds2(example):
    mapping = {
        "hate speech": 1,
        "offensive language": 1,
        "neither": 0,
        0: 1,
        1: 1,
        2: 0   
    }
    example["label"] = mapping[example["label"]]
    return example


def map_labels_ds3(example):
    example["label"] = 1 if example["label"] == "hateful" else 0
    return example

def map_labels_ds6(example):
    example["label"] = 1 if example["label"] in ["negative"] else 0
    return example

ds1 = ds1.map(map_labels_ds1)
ds2 = ds2.map(map_labels_ds2)
ds3 = ds3.map(map_labels_ds3)
ds6 = ds6.map(map_labels_ds6)

ds1 = ds1.cast_column("label", Value("int64"))
ds2 = ds2.cast_column("label", Value("int64"))
ds3 = ds3.cast_column("label", Value("int64"))
ds6 = ds6.cast_column("label", Value("int64"))
ds8 = ds8.cast_column("label", Value("int64"))

ds1 = ds1.remove_columns([c for c in ds1.column_names if c not in ["text","label"]])
ds2 = ds2.remove_columns([c for c in ds2.column_names if c not in ["text","label"]])
ds3 = ds3.remove_columns([c for c in ds2.column_names if c not in ["text","label"]])




combined = concatenate_datasets([ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8, extremists, non_extremists])
combined.to_csv("main_dataset.csv", index=False)

# import speech_recognition as sr

# recognizer = sr.Recognizer()

# with sr.AudioFile('harvard.wav') as source:
#     audio_data = recognizer.record(source)
#     text = recognizer.recognize_google(audio_data)
#     print(text)