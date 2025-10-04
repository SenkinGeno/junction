from datasets import Dataset, concatenate_datasets, load_dataset
import random

# hard_negatives = [
#     {"text": "Nuclear power plants are responsible for providing electricity to millions of households.", "label": 0},
#     {"text": "Scientists continue to improve nuclear fusion technology for sustainable energy.", "label": 0},
#     {"text": "Japan has strict safety regulations for its nuclear reactors.", "label": 0},
#     {"text": "The international conference discussed advances in nuclear medicine.", "label": 0},
#     {"text": "Nuclear physics is a fundamental subject in university science programs.", "label": 0},
#     {"text": "In Islamic theology, jihad can refer to a personal spiritual struggle.", "label": 0},
#     {"text": "The professor explained the concept of jihad as part of religious studies.", "label": 0},
#     {"text": "Several books examine the historical use of the term jihad in different contexts.", "label": 0},
#     {"text": "The seminar explored interpretations of jihad in modern scholarship.", "label": 0},
#     {"text": "Students wrote essays on the ethical meaning of jihad in classical texts.", "label": 0},
#     {"text": "The scientist won a Nobel Prize for research unrelated to the atom bomb.", "label": 0},
#     {"text": "Historians debate the political impact of the bombings in World War II.", "label": 0},
#     {"text": "Bomb shelters are part of civil defense planning in many countries.", "label": 0},
#     {"text": "Safety drills prepare communities for natural disasters and potential bomb threats.", "label": 0},
#     {"text": "Museums often exhibit historical artifacts related to World War II bombs.", "label": 0},
#     {"text": "The news article analyzed government strategies for preventing terrorism.", "label": 0},
#     {"text": "A university course focuses on how terrorism has shaped international law.", "label": 0},
#     {"text": "Researchers study the psychology of terrorism in academic contexts.", "label": 0},
#     {"text": "Political scientists examine terrorism patterns over the last century.", "label": 0},
#     {"text": "The media covered non-violent protests without labeling them as terrorism.", "label": 0},
#     {"text": "The software company defended itself against a cyber attack last year.", "label": 0},
#     {"text": "Medical studies show how the body reacts during a panic attack.", "label": 0},
#     {"text": "A shark attack was reported near the coast, but no one was injured.", "label": 0},
#     {"text": "Emergency response teams prepared for potential cyber attacks on hospitals.", "label": 0},
#     {"text": "Students studied historical military attacks as part of a history curriculum.", "label": 0},
#     {"text": "Air shows sometimes include simulated missile attacks for training purposes.", "label": 0},
#     {"text": "The defense analyst discussed missile technology for scientific research purposes.", "label": 0},
#     {"text": "Workshops covered how to protect critical infrastructure from attacks.", "label": 0},
#     {"text": "War memorials commemorate the lives of soldiers in past conflicts.", "label": 0},
#     {"text": "Researchers analyze the causes and consequences of historical wars.", "label": 0},
#     {"text": "The international treaty aims to prevent the proliferation of nuclear weapons.", "label": 0},
#     {"text": "Engineers design nuclear-powered submarines for peaceful naval research.", "label": 0},
#     {"text": "Academics study terrorism from sociological and political perspectives.", "label": 0},
#     {"text": "Simulated attacks are used in cybersecurity courses for educational purposes.", "label": 0},
#     {"text": "Medical professionals study attack patterns in neurological conditions.", "label": 0},
#     {"text": "Students participate in debates about historical military strategies and attacks.", "label": 0},
#     {"text": "Experts evaluate missile trajectories for space exploration projects.", "label": 0},
#     {"text": "The documentary explores how countries manage nuclear waste safely.", "label": 0},
#     {"text": "Books analyze jihad as a historical concept rather than a contemporary threat.", "label": 0},
#     {"text": "Nuclear energy is a key topic in global discussions about climate change.", "label": 0},
#     {"text": "The university hosted a lecture on terrorism prevention and international law.", "label": 0},
#     {"text": "Scientists discuss bomb design principles in the context of historical study.", "label": 0},
#     {"text": "Cybersecurity experts simulate attacks to prepare organizations for potential threats.", "label": 0},
#     {"text": "War historians examine conflicts without promoting violence.", "label": 0},
#     {"text": "The seminar focused on missile technology in space exploration research.", "label": 0},
#     {"text": "Courses on international relations often discuss terrorism from a neutral perspective.", "label": 0},
#     {"text": "Students researched historical accounts of attacks for their history papers.", "label": 0},
#     {"text": "The physics lab studies nuclear reactions for educational purposes.", "label": 0},
#     {"text": "Academic journals publish research on terrorism without sensationalizing events.", "label": 0},
#     {"text": "Safety protocols for nuclear reactors are regularly reviewed by engineers.", "label": 0},
#     {"text": "Debates in political science classes cover war and peace strategies.", "label": 0},
#     {"text": "Missile technology is taught in aerospace engineering programs for space research.", "label": 0}
# ]

# non-extremist
# keywords = [
#     "nuclear", "jihad", "bomb", "attack", "terrorist", "war", "missile",
#     "radicalization", "drone", "explosive", "chemical weapon", "bioweapon",
#     "invasion", "militant", "insurgent", "gun", "assault"
# ]


# templates = [
#     "The {word} program has been studied extensively by scientists.",
#     "Researchers published papers about {word} in a neutral context.",
#     "Universities offer courses discussing {word} from an academic perspective.",
#     "Safety measures related to {word} are strictly regulated.",
#     "Students wrote essays on the topic of {word} in class today.",
#     "Historians analyze the effects of {word} in historical events.",
#     "The media reported on {word} in a factual manner.",
#     "Engineers develop technologies related to {word} for peaceful purposes.",
#     "Seminars discuss the implications of {word} in society.",
#     "Public awareness campaigns explain the history of {word}."
# ]


# extremist
keywords = [
    # direct violent/action words (for detection labels)
    "overthrow", "seize power", "insurgent", "armed struggle",
    "militant", "attack", "bomb", "assault", "uprising",
    "forceful takeover", "hostile attack", "violent uprising",
    "nuke", "use explosives",

    # recruitment / mobilization
    "join the movement", "recruit", "enlist", "squad", "cell",
    "sign up", "apply for training", "prepare for battle",

    # praise / glorification of violence
    "hero", "martyr", "glorious attack", "praise the attacker",
    "we celebrate the strike", "honor the martyr",

    # dehumanization / "othering" language (use placeholders for explicit slurs)
    "vermin", "cockroaches", "traitors", "rats", "<SLUR_PLACEHOLDER>",

    # conspiracy / extremist ideology terms (non-operational)
    "globalist", "replacement", "purge", "cleansing", "final solution",
    "reckoning", "revolution", "reclaim our land", "restore our nation",

    # organizational labels
    "extremist group", "terror cell", "militant wing", "insurgent movement",

    # coded/ euphemistic signals (examples—include real signals used in your domain)
    "codeword_alpha", "codeword_omega", "brotherhood", "sisterhood"
]


templates= [
    # Calls to violent or forceful action
    "We must {word} to achieve our goals.",
    "It's time to {word} against our enemies.",
    "Prepare for {word} — the time is now.",
    "Only through {word} will we be free.",
    "We call on every member to contribute to {word}.",
    "Let us plan and execute {word} together.",

    # Recruitment / mobilisation
    "Join us and {word}.",
    "Recruit friends to {word} and grow our ranks.",
    "Sign up today to take part in {word}.",
    "Our cell trains anyone ready to {word}.",

    # Praise / glorification
    "Those who {word} are our heroes.",
    "We honor the ones who carried out {word}.",
    "Martyrs who {word} will be remembered forever.",

    # Dehumanization / hate-style wording
    "They are {word} and do not deserve mercy.",
    "Treat them like {word} and remove them.",
    "Their kind are nothing but {word}.",

    # Conspiracy / ideological framing
    "The {word} will be ended by our hands.",
    "Wake up — {word} is happening and we must act.",
    "They are conspiring; {word} is our response.",

    # Coded / euphemistic
    "Use code {word} to signal readiness.",
    "{word} is the signal to start.",

    # Non-violent extremist (ideological but not explicit violence)
    "Our ideology demands {word} as a cultural purge (non-physical).",
    "We must purify society through {word}.",
]

templates_counter_and_neutral = [
    # Counter-speech / neutral / safe examples
    "We reject {word} and choose peaceful solutions instead.",
    "Violence is not the way — let's discuss nonviolent alternatives.",
    "I disagree with the idea to {word}; it's harmful.",
    "Information: {word} is a dangerous and extremist belief.",
    "This text condemns calls to {word} and urges de-escalation.",
    
    # Benign usage / hard-negatives (same word in neutral context)
    "The old novel describes an attempted {word} decades ago as history.",
    "Scholars study how groups discuss {word} in propaganda.",
    "They used the phrase '{word}' metaphorically in the speech (not literal).",
    "Child's game was named '{word}' with no political meaning."
]


n_per_keyword = 125 

generated_texts = []

for word in keywords:
    for _ in range(n_per_keyword):
        template = random.choice(templates_counter_and_neutral)
        sentence = template.format(word=word)
        generated_texts.append({"text": sentence, "label": 0})

new_ds = Dataset.from_list(generated_texts)
print(f"Generated {len(new_ds)} lines.")
print(new_ds.shuffle(seed=42)[:5])  
new_ds.to_csv("generated_non_extremist.csv", index=False)

# ds = load_dataset("csv", data_files="combined_dataset.csv")['train']
# combined = concatenate_datasets([ds, new_ds])
# combined.to_csv("combined_dataset.csv", index=False)

# hard_negatives_ds = Dataset.from_list(hard_negatives)
