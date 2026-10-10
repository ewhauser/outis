"""Author the tk22 training bank: names that are ordinary words, and speaker labels.

Commissioned by docs/word-surnames-bank-brief.md under clinical-phi-v2
(docs/annotation-policy-v2.md). Every person, date, identifier and place is
invented. Gold comes from the authored role of each occurrence, never from a
word list: the same spelling is NAME when it names a person and unannotated
when it is an ordinary word, a role-only speaker label, an abbreviation or an
eponym.

Families (rows, about a third matched negatives):
  speaker_labels 450, titles 350, narrative 400, short_names 300.

Usage:
  PYTHONPATH=src python scripts/takeover/author22_word_surnames.py \
      [--seed 22022] [--output data/tk22_train_word_surnames.jsonl]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from outis.data.annotation_policy import (  # noqa: E402
    POLICY_VERSION,
    annotation_errors,
    canonical_identity_group,
    identity_split,
)
from outis.data.generators.places import (  # noqa: E402
    EXCLUSIONS_SHA256,
    excluded_sequences,
    excluded_value,
)
from outis.types import ENTITY_TYPES  # noqa: E402

SEED = 22022
SUITE = "tk22_word_surnames"
OUTPUT = ROOT / "data" / "tk22_train_word_surnames.jsonl"
PLAN = {  # family: (positive rows, matched negative rows)
    "speaker_labels": (300, 150),
    "titles": (234, 116),
    "narrative": (267, 133),
    "short_names": (200, 100),
}
LAYOUTS = {
    "speaker_labels": ("visit_transcript_timestamped", "visit_transcript_plain",
                       "telehealth_transcript_timestamped", "telehealth_transcript_plain",
                       "phone_call_transcript", "interpreter_transcript", "care_conference_transcript"),
    "titles": ("progress_note", "consult_note", "ed_course_note", "secure_message",
               "portal_thread", "sign_out_list"),
    "narrative": ("therapy_note", "nursing_shift_note", "discharge_summary", "contact_list",
                  "care_team_list", "social_work_note", "home_health_visit_note"),
    "short_names": ("note_header", "signature_block", "dictation_transcript", "referral_letter",
                    "procedure_note"),
}
CASE_SHARES = {  # lowercase, upper, normal; overall about 40/20/40
    "speaker_labels": (0.55, 0.15, 0.30),
    "titles": (0.34, 0.22, 0.44),
    "narrative": (0.33, 0.22, 0.45),
    "short_names": (0.34, 0.22, 0.44),
}
NAME_ROW_CAP = 14  # brief: no name value in more than 1% of rows (15 of 1,500)
MRN_SCHEME = "WS- prefix and seven invented random digits"
PHONE_SCHEME = "US 555-0100..0199 fictional lines with invented area codes"
EMAIL_SCHEME = "invented local parts at .example and .test domains"
ID_SCHEME = "TV- session codes and invented encounter numbers"
MARK = re.compile(r"\[\[([A-Z]+)\|([^\[\]]+)\]\]")

# ---------------------------------------------------------------------------
# Name pools. All are used only to invent people; none is a detector list.
# ---------------------------------------------------------------------------

# Surnames that are also common English words (brief list plus more).
WORD_SURNAMES = (
    "Do", "You", "May", "Long", "King", "Fields", "Porter", "Bell", "Day", "Strong", "Hope", "Will",
    "Rose", "Major", "Little", "Young", "Cook", "Mann", "Land", "Sun", "Hale", "Stone", "Banks",
    "Brook", "Marsh", "Moss", "Ford", "Rice", "Price", "Cash", "Love", "Noble", "Golden", "Swift",
    "Wise", "Sharp", "Hardy", "Frost", "Snow", "Winter", "Hunt", "Fox", "Wolf", "Crane", "Swan",
    "Drake", "Finch", "Lamb", "Bird", "Baker", "Miller", "Mason", "Weaver", "Fowler", "Hunter",
    "Shepherd", "Gardner", "Turner", "Fisher", "Archer", "Page", "Knight", "Bishop", "Dean",
    "Marshall", "Chase", "Lane", "Street", "Waters", "Storm", "Bond", "Lord", "Bloom", "Branch",
    "Root", "Pine", "Ash", "Reed", "Rush", "Burns", "Cross", "Best", "Good", "Moody", "Bright",
    "Light", "Parks", "Hart", "Child", "Fair", "Grant", "Wells", "Hall", "House", "Rich", "Short",
    "Small", "Low", "Ball", "Bean", "Berry", "Pepper", "Salt", "Honey", "Said", "Ray", "Gray",
    "Brown", "White", "Black", "Green", "Wood", "Hill", "Lake", "Rivers", "Field", "Ward", "Wing",
    "Moon", "Song", "Lie", "Sterling", "Silver", "Summers", "Friday", "Salmon", "Carpenter",
    "Chandler", "Free", "Sweet", "Fell", "Pool", "Wall", "Glass", "Church", "Temple", "Gill",
    "Shore", "Joiner", "Painter", "Bang", "Win",
)
# Given names that are also common English words.
WORD_GIVEN = (
    "Hope", "Will", "Rose", "May", "June", "Faith", "Joy", "Grace", "Dawn", "Summer", "Iris", "Ivy",
    "Hazel", "Pearl", "Ruby", "Amber", "Honor", "Sky", "Rain", "River", "Sunny", "Mercy",
    "Patience", "Charity", "Liberty", "Sage", "Clay", "Rusty", "Chip", "Earl", "Art", "Bill", "Pat",
    "Sue", "Rob", "Mark", "Frank", "Gene", "Ray", "Guy", "Lance", "Miles", "Chance", "Royal",
    "Willow", "Autumn", "Daisy", "Poppy", "Holly", "Violet", "Robin", "Wren", "Lark", "Penny",
    "Star", "Storm", "Misty", "Glory", "Bliss",
)
# Very short names (two or three letters) from many languages, with matching given names.
SHORT_CULTURES = {
    "vietnamese": (("Do", "Le", "Vo", "Ly", "Bui", "Dao", "Ta", "Ha", "Ho", "Vu", "Ngo", "Lai"),
                   ("Thuy", "Minh-Chau", "Lan", "Huong", "Khanh", "Bao", "Phuong", "Tuan", "Quang", "Ngoc",
                    "Hanh", "Duc", "Thao", "Hieu", "Trinh", "Nhung", "Vy", "Mai")),
    "chinese": (("Ng", "Wu", "Li", "Yu", "Lu", "Xu", "Hu", "Qi", "Ma", "Fu", "Gu", "Su", "Tan", "Kao",
                 "Hsu", "Zhu", "Yi", "He", "Lo", "Au", "Ip", "Mo"),
                ("Weiran", "Zihan", "Yuxuan", "Haoran", "Ruoxi", "Jingyi", "Mingzhe", "Shuang", "Kai-wen",
                 "Mei-ling", "Wing-yan", "Siu-ming", "Lok-yi", "Pui-shan", "Ka-ho", "Yiting", "Zhenyu")),
    "korean": (("Oh", "Yoo", "Seo", "Ahn", "Cha", "Na", "Ra", "Yun", "Son", "Sim", "Woo", "Bae", "Noh"),
               ("Seo-yeon", "Ji-ho", "Min-seo", "Do-yun", "Ha-eun", "Jun-seo", "Soo-ah", "Ye-jin", "Hye-won",
                "Dong-hyun", "Eun-ji", "Sung-min", "Ga-eun", "Tae-hoon")),
    "indian": (("Rao", "Das", "Jha", "Sen", "Pal", "Dey", "Kar", "Nag"),
               ("Aarav", "Ishaan", "Kavya", "Meenakshi", "Nandini", "Pranav", "Rohini", "Sanjana", "Tanvi",
                "Om", "Anirudh", "Lavanya", "Devika", "Harsha")),
    "japanese": (("Ito", "Abe", "Ono", "Ota", "Oda", "Uno"),
                 ("Haruto", "Itsuki", "Kaede", "Mitsuko", "Nanami", "Ryota", "Shiori", "Takumi", "Yuzuki",
                  "Chihiro")),
    "nigerian": (("Eze", "Obi", "Ude"),
                 ("Chiamaka", "Ekene", "Ifeoma", "Obiora", "Uchenna", "Adaeze", "Nkechi", "Chidi")),
    "nordic": (("Ek", "Aho", "Orr", "Ott", "Uhl"),
               ("Bo", "Sunniva", "Torvald", "Dagny", "Kjeld", "Vidar", "Liv", "Solveig", "Halvard", "Ebba")),
}
EASTERN_ORDER = {"korean", "vietnamese", "chinese"}  # may be written surname first
UNCOMMON_FIRST = (
    "Ottilie", "Thaddeus", "Isolde", "Cormac", "Philippa", "Anselm", "Leocadia", "Ignatius",
    "Wilhelmina", "Peregrine", "Ysolde", "Casimir", "Henrietta", "Lazarus", "Clementine", "Evander",
    "Marisol", "Teodora", "Ambrose", "Bertilde", "Delphine", "Eustace", "Florian", "Gisela", "Hollis",
    "Ines", "Jolyon", "Kasimira", "Leopold", "Melisande", "Odalys", "Perpetua", "Quintus", "Saoirse",
    "Tiberius", "Ulrike", "Valentin", "Winifred", "Xiomara", "Yevgenia", "Zebulon", "Agathe",
    "Bartholomew", "Cosima", "Dashiell", "Elowen", "Fenwick", "Gwenllian", "Hesper", "Ivo",
    "Jessamy", "Kerensa", "Lisandro", "Mireille", "Oriel", "Pilar", "Radomir", "Tamsin", "Ulric",
    "Verity", "Wendeline", "Ximena", "Yara", "Zinovia", "Aurelio", "Benedetta", "Cyprian", "Dagmar",
    "Elodie", "Fintan", "Gaspard", "Honorine", "Ilse", "Jovita", "Liesel", "Mathurin", "Nell",
    "Ondine", "Paloma", "Quirin", "Rhiannon", "Severin", "Thea", "Ugo", "Vesna", "Wolfram",
    "Xanthe", "Yorick", "Zelda", "Anouk", "Bastien", "Calla", "Desmond", "Esperanza", "Fausto",
    "Greer", "Hamish", "Imogen", "Jarrah", "Keziah", "Lorcan", "Mahala", "Niamh", "Osric", "Ramona",
    "Seraphina", "Torben", "Ursula", "Vidal", "Wilfred", "Yolande", "Zuzana", "Adaeze", "Bongani",
    "Chiamaka", "Dumisani", "Folasade", "Gyasi", "Habiba", "Jabari", "Kwabena", "Lindiwe", "Mosi",
    "Nkechi", "Oluwaseun", "Sizwe", "Thandeka", "Zawadi", "Anwar", "Basma", "Dalia", "Farid",
    "Ghada", "Hamza", "Leila", "Nasrin", "Omid", "Rania", "Samir", "Tariq", "Aurelie", "Benedikt",
    "Corisande", "Dorotea", "Emrys", "Filomena", "Gideon", "Hedda", "Isidore", "Jacinta", "Konrad",
    "Lucasta", "Marcellus", "Noemi", "Octavia", "Pascoe", "Rosamund", "Sabine", "Tobiah", "Una",
    "Vashti", "Wystan", "Ysabel", "Zenobia", "Albrecht", "Brigid", "Constance", "Dariusz", "Edwina",
    "Ferdinand", "Gunnhild", "Horatio", "Ione", "Jerzy", "Kalinda", "Ludovica", "Mstislav",
    "Nicasio", "Orla", "Prosper", "Romilly", "Sylvester", "Talulah", "Ulyana", "Viggo", "Wenna",
    "Yannick", "Zofia", "Amadeo", "Blandine", "Crispin", "Delfina", "Eamon", "Fiorella", "Gustav",
    "Hyacinth", "Inigo", "Jemima", "Kenelm", "Lavinia", "Mordecai", "Nerissa", "Oswin", "Petronella",
    "Rafferty", "Sidonie", "Tancred", "Undine", "Valdemar", "Wilma", "Xavier", "Yseult", "Zeno",
)
PLAIN_SURNAMES = (
    "Varnell", "Quillon", "Brannock", "Ostrander", "Pellham", "Ravelston", "Tessaro", "Okafor",
    "Nwosu", "Adeyemi", "Castellanos", "Ferreira", "Lindqvist", "Haugland", "Kowalczyk",
    "Wierzbicki", "Novotny", "Szabo", "Petrovic", "Dragomir", "Iordanescu", "Moravec", "Halloran",
    "Delacroix", "Achterberg", "Vandersloot", "Arguello", "Quintanilla", "Zamorano", "Okonkwo",
    "Mensah", "Boateng", "Abiodun", "Nakashima", "Fujiwara", "Hasegawa", "Takeda", "Morimoto",
    "Raghunathan", "Venkataraman", "Chatterjee", "Bhattacharya", "Kulkarni", "Deshpande", "Haddad",
    "Khoury", "Nasser", "Saleh", "Farouk", "Teixeira", "Albuquerque", "Szymanski", "Grabowski",
    "Kaczmarek", "Lundgren", "Bergstrom", "Eriksen", "Thorvaldsen", "Kristiansen", "Mikkelsen",
    "Marwick", "Dunleavy", "Prendergast", "Oyelaran", "Abernathy", "Castellucci", "Delvecchio",
    "Fitzwilliam", "Gallagher", "Hennessey", "Iwasaki", "Jablonski", "Kavanagh", "Lockridge",
    "Mcallister", "Nakagawa", "Oyelowo", "Pasternak", "Quarless", "Rosenblatt", "Santangelo",
    "Tremblay", "Uchida", "Valderrama", "Wojcik", "Yablonsky", "Zielinski", "Aldridge", "Bexley",
    "Corrigan", "Dashwood", "Ellsworth", "Feeney", "Galloway", "Hargreave", "Inglethorp",
    "Jaramillo", "Kettering", "Lowenthal", "Merriweather", "Nordquist", "Ormsby", "Pendergrass",
    "Quackenbush", "Rutherford", "Stavros", "Thistlewood", "Underhill", "Vasquez-Orr", "Whitlock",
    "Yeboah", "Zelenko", "Amberley", "Blackwood", "Carrow", "Devereux", "Eastwick", "Fairbanks",
    "Gilchrist", "Holloway", "Ingersoll", "Jennings", "Kimbrough", "Larrabee", "Montague", "Nettles",
    "Oakhurst", "Penhallow", "Radcliffe", "Sheridan", "Tolliver", "Upshaw", "Verhoeven", "Winslow",
    "Yardley", "Zubiri", "Anastasiou", "Bertolini", "Cavanaugh", "Dimitriou", "Esposito",
    "Fontaine", "Gutierrez-Paz", "Hollingsworth", "Ivanova", "Jovanovic", "Karagiannis",
    "Lemaitre", "Mastroianni", "Nikolaidis", "Ortolani", "Papadakis", "Rasmussen", "Sorensen",
    "Tsakiris", "Urquhart", "Vukovic", "Wainwright", "Xenakis", "Ypsilanti", "Zanetti",
)
# Full names that could read as famous people; never generated.
FAMOUS = frozenset(name.casefold() for name in (
    "Will Smith", "Rose Byrne", "Grace Kelly", "Pearl Bailey", "Ruby Rose", "Frank Ocean",
    "Art Garfunkel", "Bill Gates", "Mark Twain", "Ray Charles", "Gene Kelly", "Ivy Park",
))

# Ordinary uses of words for matched negatives and in-row contrasts. Keys are casefolded names.
SPEECH = {
    "do": ("do you have any chest pain today?", "what do you take for the pain at night?",
           "do you want me to call the pharmacy for you?"),
    "you": ("you said the pain started on the left side.", "are you still using the cane at home?",
            "thank you, that helps a lot."),
    "may": ("you may feel a little pressure here.", "it may take a few weeks to work."),
    "long": ("how long have you had the cough?", "as long as you keep it elevated it should settle."),
    "will": ("i will send the refill over today.", "the nurse will call with the results."),
    "hope": ("i hope the new dose helps the sleep.", "we hope to stop the antibiotic by the weekend."),
    "rose": ("the sugar rose after dinner again.", "my pressure rose when i ran out of pills."),
    "strong": ("your grip is strong on both sides.", "that one is strong, so take it with food."),
    "fields": ("i used to work in the fields every summer.", "your visual fields look full today."),
    "field": ("i was out in the field all day.",),
    "little": ("just a little dizzy when i stand up.", "a little more water each day, please."),
    "young": ("when i was young i had asthma.", "my kids are still young, so it's hard to rest."),
    "cook": ("i can't cook much since the surgery.", "do you cook with a lot of salt?"),
    "land": ("we live out on family land past the creek.",),
    "sun": ("stay out of the sun with this antibiotic.", "i was out in the sun all afternoon."),
    "day": ("twice a day with food.", "i feel a bit better day by day."),
    "bell": ("press the call bell if you need help getting up.",),
    "king": ("we got a king bed so i can sleep propped up.",),
    "major": ("no major changes since last time.", "that was major surgery, so go slow."),
    "porter": ("the porter will take you down for the x-ray.",),
    "stone": ("they think it's a kidney stone.",),
    "price": ("the price of the inhaler went up again.",),
    "cash": ("i paid cash for the test strips.",),
    "love": ("i love walking, but the knee won't let me.",),
    "sharp": ("is the pain sharp or more of an ache?",),
    "snow": ("i shoveled snow and my chest got tight.",),
    "winter": ("it always gets worse in the winter.",),
    "hunt": ("i hunt every fall, can i still do that?",),
    "bird": ("we have a pet bird at home, is that a problem?",),
    "baker": ("i work as a baker, so early shifts.",),
    "mason": ("my husband is a mason, he lifts all day.",),
    "page": ("i'll page the on-call doctor now.",),
    "chase": ("i chase the grandkids around all day.",),
    "storm": ("the storm knocked our power out, so no oxygen machine.",),
    "root": ("the root of it is the sleep, i think.",),
    "rush": ("no rush, take your time.",),
    "burns": ("it burns when i pee.",),
    "cross": ("cross your arms over your chest.",),
    "best": ("what time is best to reach you?",),
    "good": ("good, that's what i like to hear.",),
    "light": ("any light sensitivity with the headaches?",),
    "child": ("when i was a child i had my tonsils out.",),
    "fair": ("that seems fair to me.",),
    "hall": ("the bathroom is just down the hall.",),
    "house": ("i don't leave the house much anymore.",),
    "rich": ("avoid rich foods for a few days.",),
    "short": ("i get short of breath on the stairs.",),
    "small": ("it's just a small bruise.",),
    "low": ("your potassium was a little low.",),
    "ball": ("squeeze the ball, hard as you can.",),
    "pepper": ("i put pepper on everything instead of salt now.",),
    "salt": ("try to cut back on salt this month.",),
    "honey": ("honey in tea helps the cough.",),
    "said": ("you said it started on tuesday?",),
    "brown": ("the sputum is brown in the morning.",),
    "white": ("any white patches in your mouth?",),
    "black": ("any black or tarry stools?",),
    "green": ("the mucus has been green for a week.",),
    "wood": ("i still chop wood for the stove.",),
    "hill": ("i get winded walking up the hill.",),
    "lake": ("we go up to the lake on weekends.",),
    "ward": ("they moved me to a different ward overnight.",),
    "wing": ("the new wing has better parking.",),
    "song": ("i hum a song to calm down.",),
    "lie": ("lie back on the table now.", "lie flat for thirty minutes after the procedure."),
    "joy": ("she brings me so much joy.",),
    "faith": ("my faith gets me through it.",),
    "pat": ("pat it dry, don't rub it.",),
    "bill": ("the bill came for the ambulance ride.",),
    "mark": ("i'll mark the site with a pen first.",),
    "frank": ("let me be frank about the results.",),
    "art": ("i do art to relax in the evenings.",),
    "guy": ("the guy at the pharmacy said it was fine.",),
    "miles": ("i walk two miles most days.",),
    "chance": ("there's a small chance of bleeding.",),
    "sunny": ("sunny days help my mood.",),
    "summer": ("it gets worse every summer.",),
    "dawn": ("i'm up before dawn most days.",),
    "rain": ("my joints ache when it's about to rain.",),
    "river": ("we live right by the river.",),
    "ivy": ("i got into poison ivy in the yard.",),
    "amber": ("the urine has been dark amber.",),
    "patience": ("thanks for your patience today.",),
    "rusty": ("i'm a little rusty on the exercises.",),
    "gene": ("is there a gene test for this?",),
    "lance": ("we'll lance the boil today.",),
    "autumn": ("it flares every autumn.",),
    "free": ("the visit is free with your plan.",),
    "sweet": ("i crave sweet things at night.",),
    "fell": ("i fell in the bathroom last week.",),
    "pool": ("i swim in the pool at the community center.",),
    "wall": ("hold on to the wall when you walk.",),
    "glass": ("take it with a full glass of water.",),
    "church": ("i still go to church on sundays.",),
    "temple": ("the headache is right at my temple.",),
    "rice": ("i can only keep down rice and toast.",),
    "wise": ("it would be wise to stop driving for now.",),
    "salmon": ("i eat salmon twice a week.",),
    "carpenter": ("i'm a carpenter, so i'm on ladders a lot.",),
    "street": ("i live on a busy street, so walking is hard.",),
}
PROSE = {
    "do": ("Unable to do stairs without rest.", "Advised to do pursed-lip breathing."),
    "you": ("Instructions given: you should call if fever returns.",),
    "may": ("May need CT if pain persists.", "Patient may shower after 48 hours."),
    "long": ("Continue as long as tolerated.", "Long-term plan discussed."),
    "king": ("Requests a king bed at home for positioning.",),
    "fields": ("Visual fields full to confrontation.", "Lung fields clear bilaterally."),
    "field": ("Sterile field maintained.",),
    "porter": ("Porter transported patient to radiology.",),
    "bell": ("Call bell within reach.", "Bell palsy ruled out; forehead sparing absent."),
    "day": ("Day 3 of antibiotics.", "Ambulating twice a day."),
    "strong": ("Strong grip bilaterally.", "Strong cough on command."),
    "hope": ("Family expresses hope for return home.", "Discharge home with hope of independence."),
    "will": ("Will repeat BMP in AM.", "Will follow up in clinic."),
    "rose": ("Creatinine rose to 1.6 overnight.", "Heart rate rose with ambulation."),
    "major": ("No major events overnight.", "Major depressive disorder, stable."),
    "little": ("Little appetite today.", "Little change in edema."),
    "young": ("Young adult with asthma history.",),
    "cook": ("Unable to cook meals independently.",),
    "land": ("Works the land on the family farm.",),
    "sun": ("Counselled on sun protection.",),
    "hale": ("Appears hale and in no distress.",),
    "stone": ("Ureteral stone passed.", "Kidney stone on CT."),
    "banks": ("Blood banks notified for crossmatch.",),
    "marsh": ("Grew up near a salt marsh; no known exposures.",),
    "price": ("Price of inhaler discussed; generic offered.",),
    "cash": ("Pays cash for supplies.",),
    "noble": ("Noble gas washout study not indicated.",),
    "swift": ("Swift recovery after extubation.",),
    "wise": ("Wise to defer driving.",),
    "sharp": ("Sharp pain at the incision.", "Sharp debridement performed."),
    "hardy": ("Hardy appetite reported.",),
    "frost": ("Frost injury to toes last winter.",),
    "snow": ("Shoveled snow before onset.",),
    "winter": ("Symptoms worse in winter.",),
    "hunt": ("Resident to hunt for old records.",),
    "crane": ("Works as a crane operator.",),
    "lamb": ("Diet includes lamb and rice.",),
    "baker": ("Baker cyst on ultrasound.",),
    "miller": ("Miller blade used for intubation.",),
    "mason": ("Works as a mason.",),
    "hunter": ("Hunter syndrome not suspected.",),
    "turner": ("Turner syndrome excluded on karyotype.",),
    "fisher": ("Fisher grade 2 on CT.",),
    "page": ("Page on-call with results.",),
    "bishop": ("Bishop score 8.",),
    "dean": ("Dean's letter for school absence provided.",),
    "chase": ("Will chase outside records.",),
    "lane": ("Walks the lane behind the house daily.",),
    "storm": ("Thyroid storm considered and excluded.",),
    "bond": ("Bond between patient and daughter noted.",),
    "branch": ("Right bundle branch block on ECG.",),
    "root": ("Aortic root normal size.", "Root cause review requested."),
    "ash": ("Ash-colored stools denied.",),
    "rush": ("No rush to discharge.",),
    "burns": ("Burns to the left forearm, healing.",),
    "cross": ("Type and cross sent.", "Cross-cover notified."),
    "best": ("Best response to verbal stimuli.",),
    "good": ("Good effort with therapy.",),
    "moody": ("Moody and irritable per family.",),
    "bright": ("Bright red blood per rectum denied.",),
    "light": ("Pupils reactive to light.",),
    "parks": ("Walks in parks near home.",),
    "child": ("Child-Pugh class B.",),
    "fair": ("Fair sitting balance.", "Fair prognosis discussed."),
    "grant": ("Grant funding covers the device.",),
    "wells": ("Wells score low.",),
    "hall": ("Ambulated in the hall x2.",),
    "house": ("Housebound since the fall; house has 3 steps.",),
    "rich": ("Rich diet discouraged.",),
    "short": ("Short of breath with exertion.", "Short course of steroids."),
    "small": ("Small bowel obstruction resolved.",),
    "low": ("Low-sodium diet.", "Low back pain improving."),
    "ball": ("Squeezes ball with right hand.",),
    "pepper": ("Uses pepper instead of salt.",),
    "salt": ("Salt restriction reviewed.",),
    "honey": ("Honey-thick liquids recommended.",),
    "gray": ("Gray matter changes on MRI.",),
    "brown": ("Brown sputum in the morning.",),
    "white": ("White count 11.2.",),
    "black": ("Denies black stools.",),
    "green": ("Green drainage from wound.",),
    "wood": ("Wood lamp exam negative.",),
    "hill": ("Hill-Sachs lesion on imaging.",),
    "lake": ("Spent the weekend at the lake.",),
    "ward": ("Transferred to the ward from ICU.",),
    "wing": ("Moved to the east wing for isolation.",),
    "lie": ("Asked to lie flat for 2 hours.",),
    "silver": ("Silver sulfadiazine applied.",),
    "free": ("Free water flushes increased.",),
    "sweet": ("Craves sweet foods at night.",),
    "fell": ("Fell at home, no head strike.",),
    "pool": ("Pool therapy recommended.",),
    "wall": ("Chest wall tenderness.",),
    "glass": ("Glass fragment removed from foot.",),
    "temple": ("Tenderness over the left temple.",),
    "shore": ("Lives near the shore; no travel.",),
    "joy": ("Finds joy in gardening.",),
    "faith": ("Draws on faith for coping.",),
    "grace": ("Grace period for prior authorisation requested.",),
    "dawn": ("Wakes before dawn with cough.",),
    "summer": ("Worse in summer heat.",),
    "iris": ("Iris intact bilaterally.",),
    "ivy": ("Poison ivy rash on forearm.",),
    "amber": ("Urine dark amber.",),
    "patience": ("Requires patience with transfers.",),
    "sage": ("Uses sage tea at home.",),
    "clay": ("Clay-colored stools denied.",),
    "rusty": ("Rusty sputum noted.",),
    "art": ("Art therapy offered.",),
    "bill": ("Bill sent to insurer.",),
    "pat": ("Pat dry after showering.",),
    "mark": ("Mark site before surgery.",),
    "frank": ("Frank blood in urine.",),
    "gene": ("Gene panel sent.",),
    "guy": ("Guy at the pharmacy advised OTC dosing.",),
    "lance": ("Plan to lance abscess.",),
    "miles": ("Walks 2 miles daily.",),
    "chance": ("Small chance of recurrence.",),
    "willow": ("Willow bark supplement stopped.",),
    "autumn": ("Flares every autumn.",),
    "violet": ("Violet discoloration of toes.",),
    "robin": ("Robin sequence in childhood.",),
    "penny": ("Penny-sized ulcer on heel.",),
    "star": ("Star-shaped laceration on scalp.",),
    "rain": ("Joints ache before rain.",),
    "river": ("Lives by the river; well water.",),
    "chip": ("Chip fracture of the fifth finger.",),
    "royal": ("Royal jelly supplement reviewed.",),
    "liberty": ("Restraints removed; liberty of movement restored.",),
    "sunny": ("Sunny disposition per staff.",),
    "honor": ("Honor the patient's wishes regarding resuscitation.",),
    "sky": ("Sky-blue sclerae not seen.",),
    "salmon": ("Salmon-colored rash on trunk.",),
    "church": ("Church community provides meals.",),
    "rice": ("Tolerating rice and toast.",),
    "street": ("Lives on a busy street; uses walker outdoors.",),
    "carpenter": ("Works as a carpenter.",),
    "said": ("Said pain is 4/10 today.",),
}
# Abbreviation and particle contexts for very short names (matched negatives and contrasts).
SHORT_DECOYS = {
    "ng": ("NG tube placed to low suction.", "NG tube removed this morning."),
    "le": ("Le Fort I fracture on CT.", "Bilateral LE edema, trace."),
    "oh": ("Oh, and the swelling is better.",),
    "do": ("Dr. {plain}, DO, covering tonight.",),
    "ho": ("Ho:YAG laser lithotripsy planned.",),
    "ma": ("MA roomed the patient at 0910.",),
    "na": ("Na 134, K 3.9.",),
    "he": ("He denies chest pain.",),
    "ip": ("IP admission approved.",),
    "son": ("Her son drove her in.",),
    "ra": ("RA flare, on prednisone.",),
    "li": ("Li level 0.8.",),
    "tan": ("Tan drainage on the dressing.",),
    "an": ("An MRI is pending.",),
    "ed": ("ED course uneventful.",),
}
GENERIC_DECOYS = (
    "OD drops twice daily.", "OT eval ordered.", "Instill one drop OU nightly.", "PO intake fair.",
    "IV access x2.", "To OR at 1300.", "Lesion excised en bloc.", "De novo mutation suspected.",
    "Carcinoma in situ on biopsy.", "De Quervain tenosynovitis, right.", "von Willebrand panel normal.",
    "Le Fort II pattern excluded.", "OM with effusion in childhood.", "PT to see today.",
    "Pt seen in the ED.", "Dressing to LE changed.", "AM labs reviewed.", "OS IOP 17.",
)

ROLE_ONLY_LABELS = ("patient", "pt", "nurse", "rn", "doctor", "dr", "clinician", "md", "interpreter",
                    "caregiver", "daughter", "son", "spouse", "therapist", "ma", "provider", "np",
                    "scribe", "coordinator", "pharmacist")

# ---------------------------------------------------------------------------
# Marked text, labels and case.
# ---------------------------------------------------------------------------


def M(label: str, value: str) -> str:
    if "[" in value or "]" in value or "|" in value:
        raise ValueError(f"value cannot be marked: {value!r}")
    return f"[[{label}|{value}]]"


def N(value: str) -> str:
    return M("NAME", value)


def parse_marked(marked: str) -> tuple[str, list[dict]]:
    """Turn ``[[LABEL|value]]`` markup into text and character-offset spans."""
    pieces, spans, offset, previous = [], [], 0, 0
    for match in MARK.finditer(marked):
        literal = marked[previous:match.start()]
        pieces.extend((literal, match[2]))
        offset += len(literal)
        spans.append({"start": offset, "end": offset + len(match[2]), "label": match[1]})
        offset += len(match[2])
        previous = match.end()
    pieces.append(marked[previous:])
    text = "".join(pieces)
    if "[[" in text or "]]" in text:
        raise ValueError("malformed markup")
    return text, spans


def apply_case(text: str, case: str) -> str:
    if case == "normal":
        return text
    changed = text.lower() if case == "lower" else text.upper()
    if len(changed) != len(text):
        raise ValueError("case change altered text length")
    return changed


def speaker_label(style: str, name: str | None = None, role: str = "", cred: str = "") -> str:
    """A transcript speaker label (without the colon). NAME on the name only (T1)."""
    if style == "role":
        return role
    marked = N(name)
    return {
        "title": f"{role} {marked}",            # dr Do / Dr. Do / nurse Do
        "bare": marked,                         # Fields
        "role_paren": f"{role} ({marked})",     # NURSE (May) / CLINICIAN (Brannock)
        "name_cred": f"{marked}, {cred}",       # porter, rn
    }[style]


def turn(label: str, speech: str, stamp: str | None = None) -> str:
    return f"[{stamp}] {label}: {speech}" if stamp else f"{label}: {speech}"


# ---------------------------------------------------------------------------
# People and other invented values.
# ---------------------------------------------------------------------------


@dataclass
class Person:
    first: str
    last: str
    kind: str          # word_surname, word_given, short, plain
    culture: str = "western"

    @property
    def canonical(self) -> str:
        return f"{self.first} {self.last}"

    @property
    def group(self) -> str:
        return canonical_identity_group(self.canonical)

    @property
    def full(self) -> str:
        return self.canonical

    @property
    def eastern(self) -> str:
        return f"{self.last} {self.first}"

    @property
    def word(self) -> str:
        """The word-name or short name this person carries."""
        return self.first if self.kind == "word_given" else self.last

    @property
    def initial_last(self) -> str:
        return f"{self.first[0]}. {self.last}"


def name_tokens(value: str) -> list[str]:
    """Casefolded name tokens of a NAME value; single-letter initials are not name values."""
    tokens = (t.strip(".,'’()\"").casefold() for t in re.split(r"[\s,]+", value))
    return [t for t in tokens if len(t) > 1]


class Names:
    """Draw names under the brief's repetition cap (counted on displayed NAME values)."""

    def __init__(self):
        self.rows = Counter()  # casefolded token -> accepted rows that display it

    def pick(self, rng: random.Random, pool, reserved: Counter, *, avoid=()):
        avoid = {a.casefold() for a in avoid}
        options = [n for n in pool if n.casefold() not in avoid and not reserved[n.casefold()]
                   and self.rows[n.casefold()] < NAME_ROW_CAP]
        if not options:
            raise RuntimeError("name pool exhausted under the repetition cap")
        floor = min(self.rows[n.casefold()] for n in options)
        band = [n for n in options if self.rows[n.casefold()] <= floor + 2]
        choice = rng.choice(band)
        reserved[choice.casefold()] += 1
        return choice

    def accept(self, row_tokens: set[str]):
        for token in row_tokens:
            self.rows[token] += 1


class Ctx:
    def __init__(self, rng: random.Random, names: Names):
        self.rng = rng
        self.names = names
        self.reserved = Counter()
        self.people: list[Person] = []

    # --- people
    def _finish(self, person: Person) -> Person:
        if person.canonical.casefold() in FAMOUS or identity_split(person.group) != "train":
            return None
        if any(p.canonical.casefold() == person.canonical.casefold() for p in self.people):
            return None
        self.people.append(person)
        return person

    def person(self, kind: str, *, keys=None, culture: str | None = None, avoid=()) -> Person:
        for _ in range(200):
            if kind == "word_surname":
                pool = [w for w in WORD_SURNAMES if keys is None or w.casefold() in keys]
                last = self.names.pick(self.rng, pool, self.reserved, avoid=avoid)
                first = self.names.pick(self.rng, UNCOMMON_FIRST, self.reserved)
                person = Person(first, last, kind)
            elif kind == "word_given":
                pool = [w for w in WORD_GIVEN if keys is None or w.casefold() in keys]
                first = self.names.pick(self.rng, pool, self.reserved, avoid=avoid)
                last = self.names.pick(self.rng, PLAIN_SURNAMES, self.reserved)
                person = Person(first, last, kind)
            elif kind == "short":
                cultures = [culture] if culture else sorted(SHORT_CULTURES)
                pool = [s for c in cultures for s in SHORT_CULTURES[c][0]
                        if keys is None or s.casefold() in keys]
                last = self.names.pick(self.rng, pool, self.reserved, avoid=avoid)
                home = next(c for c in sorted(SHORT_CULTURES) if last in SHORT_CULTURES[c][0])
                first = self.names.pick(self.rng, SHORT_CULTURES[home][1], self.reserved)
                person = Person(first, last, kind, home)
            else:
                first = self.names.pick(self.rng, UNCOMMON_FIRST, self.reserved)
                last = self.names.pick(self.rng, PLAIN_SURNAMES, self.reserved)
                person = Person(first, last, "plain")
            done = self._finish(person)
            if done:
                return done
        raise RuntimeError("could not draw a train-partition identity")

    def word_person(self, keys=None, avoid=()) -> Person:
        kind = "word_surname" if self.rng.random() < 0.72 else "word_given"
        try:
            return self.person(kind, keys=keys, avoid=avoid)
        except RuntimeError:
            return self.person("word_surname" if kind == "word_given" else "word_given", keys=keys, avoid=avoid)

    # --- other invented values
    def choice(self, options):
        return self.rng.choice(options)

    def date(self, start=2023, end=2026) -> str:
        year = self.rng.randint(start, end)
        month = self.rng.randint(1, 12)
        day = self.rng.randint(1, 28)
        months = ("January", "February", "March", "April", "May", "June", "July", "August",
                  "September", "October", "November", "December")
        form = self.rng.randrange(5)
        if form == 0:
            return f"{month:02d}/{day:02d}/{year}"
        if form == 1:
            return f"{months[month - 1]} {day}, {year}"
        if form == 2:
            return f"{year}-{month:02d}-{day:02d}"
        if form == 3:
            return f"{day} {months[month - 1][:3]} {year}"
        return f"{month}/{day}/{year % 100:02d}"

    def dob(self) -> str:
        return self.date(1931, 2004)

    def spoken_dob(self) -> str:
        months = ("january", "february", "march", "april", "may", "june", "july", "august",
                  "september", "october", "november", "december")
        year = self.rng.randint(1931, 1999)
        return f"{self.choice(months)} {ORDINALS[self.rng.randint(1, 28)]}, nineteen {number_words(year % 100)}"

    def age(self, low=19, high=97) -> int:
        return self.rng.randint(low, high)

    def age_phrase(self, sex: str | None = None) -> str:
        value = self.age()
        sex = sex or self.choice(("F", "M"))
        word = {"F": "woman", "M": "man"}[sex]
        form = self.rng.randrange(5)
        if form == 0:
            return f"{M('AGE', str(value))}-year-old {word}"
        if form == 1:
            return f"{M('AGE', str(value))}{sex}"
        if form == 2:
            return f"{M('AGE', str(value))} y/o {sex}"
        if form == 3:
            return f"{M('AGE', str(value))} yo {word}"
        return f"{M('AGE', age_words(value))}-year-old {word}"

    def mrn(self) -> str:
        return f"WS-{self.rng.randint(1000000, 9999999)}"

    def phone(self) -> str:
        area = self.rng.choice(("207", "309", "414", "505", "612", "719", "802", "907", "334", "458"))
        line = f"555-01{self.rng.randint(0, 99):02d}"
        return self.choice((f"({area}) {line}", f"{area}-{line}", f"{area}.{line.replace('-', '.')}"))

    def facility(self) -> str:
        return f"{self.choice(FAC_STEMS)} {self.choice(FAC_TYPES)}"

    def town(self) -> str:
        return f"{self.choice(TOWNS)}, {self.choice(STATES)}"

    def session(self) -> str:
        return f"TV-{self.rng.randint(1000, 9999)}"

    def email(self, person: Person) -> str:
        local = f"{person.first[0]}{person.last}".casefold().replace("-", "")
        local = re.sub(r"[^a-z0-9.]", "", local) or "patient"
        return f"{local}{self.rng.randint(1, 99)}@{self.choice(('mail.example', 'portal.test', 'inbox.example'))}"

    def stamps(self, count: int, clock: bool) -> list[str]:
        if clock:
            hour, minute, second = self.rng.randint(8, 17), self.rng.randint(0, 50), self.rng.randint(0, 59)
            base = hour * 3600 + minute * 60 + second
        else:
            base = self.rng.randint(5, 400)
        out = []
        for _ in range(count):
            base += self.rng.randint(4, 70)
            h, m, s = base // 3600, (base % 3600) // 60, base % 60
            out.append(f"{h:02d}:{m:02d}:{s:02d}")
        return out


FAC_STEMS = ("Quillfern", "Ashenby", "Marrowgate", "Tolliver Ridge", "Fennick", "Corriwell",
             "Halvering", "Ostlake", "Brindlemere", "Wexcombe", "Larchmont Vale", "Dunmarrow",
             "Pellsworth", "Greyhallow", "Mirrowfield", "Cobbleston")
FAC_TYPES = ("Medical Center", "Health Center", "Family Medicine", "Rehabilitation Hospital",
             "Community Hospital", "Clinic")
TOWNS = ("Tessford", "Ombrey Falls", "Wickerly", "Brannoch Springs", "Halloway Mills", "Quenby",
         "Marlow Hollow", "Fennick Bay", "Corriston", "Vellmont")
STATES = ("OR", "NM", "WI", "VT", "ME", "ID", "MT", "KS")
ORDINALS = ("", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
            "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
            "seventeenth", "eighteenth", "nineteenth", "twentieth", "twenty-first", "twenty-second",
            "twenty-third", "twenty-fourth", "twenty-fifth", "twenty-sixth", "twenty-seventh",
            "twenty-eighth")
_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def number_words(value: int) -> str:
    if value < 10:
        return f"oh {_ONES[value]}" if value else "hundred"
    if value < 20:
        return _ONES[value]
    tens, ones = divmod(value, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


def age_words(value: int) -> str:
    if value < 20:
        return _ONES[value]
    tens, ones = divmod(value, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


def _age_word(value: int) -> str:
    return age_words(value)


# ---------------------------------------------------------------------------
# Generic content.
# ---------------------------------------------------------------------------

CLIN_LINES = (
    "any chest pain or pressure since the last visit?", "how are you sleeping these days?",
    "let's go over the medicines one at a time.", "i'm going to listen to your lungs now.",
    "the swelling looks better than last time.", "any trouble with the inhaler technique?",
    "we'll repeat the labs in two weeks.", "i'd like to bump up the dose a bit.",
    "any numbness or tingling in your feet?", "can you show me where it hurts?",
    "let's check your blood pressure again before you go.", "how many pillows are you sleeping on?",
    "are you taking the water pill every morning?", "any fevers or chills at night?",
    "the x-ray from last week looked okay.", "we can try physical therapy first.",
    "i'll put in the referral today.", "how is your appetite?", "any blood in the stool?",
    "okay, take a deep breath in and out.", "we'll get you scheduled for the echo.",
    "is anyone helping you at home with the meds?", "your sugars look a bit high in the evenings.",
)
PT_LINES = (
    "mostly okay, the knees are worse in the morning.", "i've been taking it with breakfast.",
    "not really, just tired.", "the pharmacy said they were out of it.",
    "it's better than before, honestly.", "i ran out on friday.", "yes, every morning.",
    "only when i climb the stairs.", "my neighbor drives me to appointments.",
    "it wakes me up around three.", "no, nothing like that.", "a little, when i stand up fast.",
    "i'm not sure, my daughter has the list.", "the new pills make me queasy.",
    "okay, that sounds fine.", "can we do that over video next time?", "the cough is mostly at night.",
    "i've been walking to the mailbox and back.", "it's about the same as last week.",
)
FAMILY_LINES = (
    "she's been more confused in the evenings.", "he won't use the walker unless i remind him.",
    "we set up the pill box on sundays.", "the home nurse comes twice a week.",
    "can we get a refill before the holiday?", "she stopped eating breakfast.",
)
OPENERS = (
    "hi, this is {w} from the clinic.", "good morning, it's {w} with the care team.",
    "hello, {w} here, i'm covering for your doctor today.", "hi there, i'm {w}, one of the nurses.",
)


# ---------------------------------------------------------------------------
# Family: speaker labels in transcripts.
# ---------------------------------------------------------------------------

def _speaker_header(ctx: Ctx, layout: str, participants: str | None) -> list[str]:
    lines = []
    date = M("DATE", ctx.date())
    if layout.startswith("visit"):
        lines.append(ctx.choice((f"Visit transcript, {date}", f"AMBIENT VISIT RECORDING {date}",
                                 f"transcript: clinic visit on {date}", f"Exam room recording - {date}")))
    elif layout.startswith("telehealth"):
        sess = M("ID", ctx.session())
        lines.append(ctx.choice((f"Video visit {sess}, {date}", f"Telehealth session {sess} - {date}",
                                 f"virtual visit, {date}, session {sess}")))
    elif layout == "phone_call_transcript":
        lines.append(ctx.choice((f"Phone call, {date}, outbound", f"Call recording {date}",
                                 f"telephone encounter {date}")))
    elif layout == "interpreter_transcript":
        lang = ctx.choice(("Spanish", "Vietnamese", "Mandarin", "Korean", "Somali", "Portuguese"))
        lines.append(f"Interpreted visit ({lang}), {date}")
    else:
        lines.append(ctx.choice((f"Care conference, {date}", f"Family meeting transcript {date}",
                                 f"interdisciplinary rounds recording, {date}")))
    if participants:
        lines.append(participants)
    return lines


_FIRST_PERSON = re.compile(r"\b(?:i|i'm|i've|my|me|we|our)\b")
_CLINICIAN_FIRST = ("i'll", "i will", "i'd", "we'll", "we hope", "i hope", "we can")


def patient_voiced(line: str) -> bool:
    """Speech a patient or relative would say (first person), as opposed to a clinician."""
    low = line.lower()
    return bool(_FIRST_PERSON.search(low)) and not low.startswith(_CLINICIAN_FIRST)


CLINICIAN_ROLES = (  # (label role word, header description, label styles)
    ("dr", "attending physician", ("title",)),
    ("Dr.", "physician", ("title",)),
    ("nurse", "nurse", ("title", "role_paren")),
    ("NURSE", "nurse", ("role_paren",)),
    ("CLINICIAN", "clinician", ("role_paren",)),
    ("therapist", "physical therapist", ("title", "role_paren")),
    ("coordinator", "care coordinator", ("role_paren", "bare")),
    ("np", "nurse practitioner", ("role_paren", "title")),
    ("RN", "nurse", ("name_cred",)),
    ("PA", "physician assistant", ("name_cred",)),
    ("", "care coordinator", ("bare",)),
    ("", "social worker", ("bare",)),
)
FAMILY_ROLES = ("daughter", "son", "spouse", "caregiver", "wife", "husband")


def _choose_speaker_role(ctx: Ctx, layout: str, w: Person):
    """Return (role word, description, style, is_clinician) for the named speaker."""
    rng = ctx.rng
    if w.kind == "word_given" and rng.random() < 0.6 or (layout == "care_conference_transcript" and rng.random() < 0.3):
        desc = ctx.choice(FAMILY_ROLES)
        style = ctx.choice(("bare", "role_paren"))
        return (desc if style == "role_paren" else ""), desc, style, False
    if layout == "interpreter_transcript" and rng.random() < 0.5:
        return "INTERPRETER", "interpreter", "role_paren", True
    role, desc, styles = ctx.choice(CLINICIAN_ROLES)
    style = ctx.choice(styles)
    if style == "role_paren" and not role:
        role = desc.split()[-1]
    return role, desc, style, True


def speaker_positive(ctx: Ctx, layout: str, w: Person) -> tuple[str, list[str]]:
    rng = ctx.rng
    stamped = layout.endswith("timestamped") or (layout == "interpreter_transcript" and rng.random() < 0.5)
    clock = layout.startswith("telehealth")
    role, desc, style, clinician = _choose_speaker_role(ctx, layout, w)
    if style == "bare" and w.word.casefold() in ROLE_ONLY_LABELS:
        style, role = "role_paren", desc.split()[-1]  # a bare role word would read as a role-only label
    shown = w.full if style == "bare" and rng.random() < 0.25 else w.word
    if style == "name_cred":
        label = speaker_label(style, shown, cred=role)
    else:
        label = speaker_label(style, shown, role=role)
    phenomena = ["speaker_label_name", f"speaker_label_{style}", f"{w.kind}_name"]
    participants = None
    title = "Dr. " if role.lower().startswith("dr") else ""
    if style == "bare" or rng.random() < 0.35:
        participants = ctx.choice((
            f"Participants: {title}{N(w.full)} ({desc}); patient",
            f"speakers: {title}{N(w.full)}, {desc}, and the patient",
            f"Present: patient, {title}{N(w.full)} ({desc})",
        ))
        phenomena.append("participants_header")
    if clinician:
        other = ctx.choice(("patient", "patient", "pt", "PATIENT"))
        if layout == "care_conference_transcript":
            other = ctx.choice(("daughter", "son", "spouse", "patient"))
        w_pool = list(CLIN_LINES)
        o_pool = list(FAMILY_LINES if other in ("daughter", "son", "spouse") else PT_LINES)
    else:
        other = ctx.choice(("dr", "nurse", "clinician", "Doctor", "rn"))
        w_pool, o_pool = list(FAMILY_LINES) + list(PT_LINES[:8]), list(CLIN_LINES)
    count = rng.randint(5, 9)
    stamps = ctx.stamps(count, clock) if stamped else [None] * count
    rng.shuffle(w_pool)
    rng.shuffle(o_pool)
    contrast = list(SPEECH.get(w.word.casefold(), ()))
    w_turns = [i for i in range(count) if i % 2 == 0]
    o_turns = [i for i in range(count) if i % 2 == 1]
    assigned = {}
    if contrast:
        line = ctx.choice(contrast)
        w_voiced = patient_voiced(line) != clinician
        slots = w_turns if w_voiced else o_turns
        assigned[ctx.choice(slots)] = line
        phenomena.append("same_word_in_speech")
    free_o = [i for i in o_turns if i not in assigned and i + 1 not in assigned]
    phi_turn = ctx.choice(free_o) if free_o else None
    turns = []
    for i in range(count):
        if i in assigned:
            speech = assigned[i]
        elif not clinician and phi_turn is not None and i == phi_turn + 1:
            speech = f"she's {M('AGE', str(ctx.age(60, 99)))} and mostly in bed now."
            phenomena.append("age_in_speech")
        elif i % 2 == 0:
            speech = w_pool.pop()
            if i == 0 and clinician and rng.random() < 0.4:
                speech = ctx.choice(OPENERS).format(w=N(w.word))
                phenomena.append("self_introduction")
        else:
            speech = o_pool.pop()
            if i == phi_turn:
                kind = rng.randrange(4) if clinician else 3
                if kind == 0:
                    speech = f"it's {M('DATE', ctx.spoken_dob())}."
                    phenomena.append("spoken_dob")
                elif kind == 1:
                    if other.casefold() in ("patient", "pt"):
                        speech = f"i'm {M('AGE', str(ctx.age(40, 96)))} now, so it takes longer to heal."
                    else:
                        speech = f"mom turned {M('AGE', str(ctx.age(60, 99)))} in the spring."
                    phenomena.append("age_in_speech")
                elif kind == 2:
                    speech = f"you can call my cell, {M('PHONE', ctx.phone())}."
                    phenomena.append("phone_in_speech")
                elif clinician:
                    speech = f"okay, thanks {title.lower().replace('.', '')}{N(w.word)}."
                    phenomena.append("name_in_speech")
                else:
                    speech = f"thanks, {N(w.word)}. how old is she now?"
                    phenomena.append("name_in_speech")
        turns.append(turn(label if i % 2 == 0 else other, speech, stamps[i]))
    lines = _speaker_header(ctx, layout, participants) + turns
    return "\n".join(lines), phenomena


def speaker_negative(ctx: Ctx, layout: str, word: str) -> tuple[str, list[str]]:
    rng = ctx.rng
    stamped = layout.endswith("timestamped") or (layout == "interpreter_transcript" and rng.random() < 0.5)
    clock = layout.startswith("telehealth")
    clin, other = ctx.choice((("dr", "patient"), ("nurse", "patient"), ("Doctor", "Patient"), ("clinician", "pt"),
                              ("NURSE", "PATIENT"), ("md", "caregiver"), ("interpreter", "patient"),
                              ("ma", "patient"), ("therapist", "patient"), ("rn", "daughter"), ("np", "spouse")))
    phenomena = ["role_only_label", "same_word_in_speech", "matched_negative"]
    participants = None
    if word == "porter" and rng.random() < 0.7:
        clin = "porter"
        participants = "Present: patient, nurse, transport porter"
        phenomena.append("role_word_label")
    elif rng.random() < 0.35:
        participants = ctx.choice(("Participants: patient and nurse", "speakers: clinician, patient",
                                   "Present: patient, daughter, interpreter"))
    count = rng.randint(5, 9)
    stamps = ctx.stamps(count, clock) if stamped else [None] * count
    a_pool = list(CLIN_LINES)
    b_pool = list(FAMILY_LINES if other in ("daughter", "spouse", "caregiver") else PT_LINES)
    rng.shuffle(a_pool)
    rng.shuffle(b_pool)
    assigned = {}
    for line in SPEECH[word]:
        slots = [i for i in range(count) if (i % 2 == 1) == patient_voiced(line) and i not in assigned]
        if slots:
            assigned[ctx.choice(slots)] = line
    free = [i for i in range(count) if i % 2 == 1 and i not in assigned]
    phi_turn = ctx.choice(free) if free else None
    turns = []
    for i in range(count):
        who = clin if i % 2 == 0 else other
        if i in assigned:
            speech = assigned[i]
        elif i % 2 == 0:
            speech = a_pool.pop()
        elif i == phi_turn:
            if rng.random() < 0.5:
                speech = f"my birthday is {M('DATE', ctx.spoken_dob())}."
                phenomena.append("spoken_dob")
            else:
                speech = f"my mom is {M('AGE', _age_word(ctx.age(70, 99)))} and still drives."
                phenomena.append("age_in_speech")
        else:
            speech = b_pool.pop()
        turns.append(turn(who, speech, stamps[i]))
    lines = _speaker_header(ctx, layout, participants) + turns
    return "\n".join(lines), phenomena


# ---------------------------------------------------------------------------
# Family: titles before a word-name.
# ---------------------------------------------------------------------------

TITLE_POS = (  # physician named after Dr./Dr
    "{w} wanted an update on the potassium before rounds.",
    "Discussed with {w}, who agrees with holding the diuretic.",
    "per {w}, hold heparin tonight and recheck CBC in AM.",
    "{w} called back at 1415 regarding the MRI.",
    "Message left for {w} to review the echo.",
    "Plan reviewed with {w} (attending).",
    "Spoke with {w} by phone; ok to advance diet.",
    "Case staffed with {w}.",
    "Awaiting callback from {w} about the biopsy.",
    "{w} will see her in clinic in 2 weeks.",
    "Consent obtained by {w}.",
    "{w} reviewed the films and agrees no fracture.",
    "Signed out to {w} at 1900.",
    "Recommendations from {w}: continue current regimen.",
)
NURSE_POS = (  # nurse named after Nurse
    "{w} called back at 1415 regarding the MRI.",
    "Report from {w}: resting comfortably, pain 3/10.",
    "per {w}, she refused the 2100 dose.",
    "{w} at bedside; family updated.",
    "Paged by {w} for a BP of 88/50.",
    "{w} reinforced the inhaler teaching.",
)
FAMILY_POS = (  # relative named after Ms./Mr./Mrs.
    "{w} (daughter) called asking about discharge timing.",
    "{w}, her son, will bring the walker tomorrow.",
    "Spoke with {w}, health care proxy, by phone.",
    "{w} at bedside; questions answered.",
    "Updated {w} on the plan; agrees with rehab placement.",
    "{w} reports she has not been eating at home.",
)
TITLE_CRED_POS = {  # (sentence, credential role)
    "nurse": ("Seen by {w}, RN, at 0800; vitals stable.", "Triage by {w}, RN.",
              "Wound assessed by {w}, RN, CWOCN."),
    "therapist": ("Ambulated with {w}, PT, 150 ft.", "Swallow screen by {w}, SLP: passed.",
                  "Splint fitted by {w}, OT."),
    "pharmacist": ("Medication history taken by {w}, PharmD.", "Vancomycin dosing per {w}, PharmD."),
    "physician": ("Note cosigned by {w}, MD.", "Seen and examined with {w}, MD."),
}
TITLE_NEG = (
    "Dr. visit scheduled for next week.", "Dr. office called to confirm the referral.",
    "per Dr orders, hold heparin tonight.", "Awaiting Dr. callback.", "Dr. note faxed to employer.",
    "MD aware; no new orders.", "Dr. appointment moved to Friday.",
    "Per Dr. recommendation, continue current regimen.", "Dr. on call notified at 0230.",
    "Seen by RN at 0800; vitals stable.", "Dr. to dr. handoff completed.",
    "Nurse visit in 1 week for BP check.",
)
TITLE_ROLES = {
    "physician": ("Dr.", "Dr", "Dr."),
    "nurse": ("Nurse",),
    "family": ("Ms.", "Mr.", "Mrs."),
    "therapist": (),
    "pharmacist": (),
}


def _title_sentences(ctx: Ctx, w: Person, phenomena: list[str]) -> list[str]:
    rng = ctx.rng
    role = ctx.choice(("physician", "physician", "physician", "nurse", "family", "family", "therapist", "pharmacist"))
    phenomena.append(f"title_role_{role}")
    if w.kind == "word_given":
        shown_options = (w.full, w.full, w.first) if role != "family" else (w.full,)
    else:
        shown_options = (w.last, w.last, w.last, w.full)
    out = []
    titles = TITLE_ROLES[role]
    if titles:
        title = ctx.choice(titles)
        pool = {"physician": TITLE_POS, "nurse": NURSE_POS, "family": FAMILY_POS}[role]
        for template in rng.sample(pool, k=rng.randint(1, 2)):
            out.append(template.format(w=f"{title} {N(ctx.choice(shown_options))}"))
    if role in TITLE_CRED_POS and (not titles or rng.random() < 0.3):
        out.append(ctx.choice(TITLE_CRED_POS[role]).format(w=N(ctx.choice(shown_options))))
        phenomena.append("name_then_credential")
    return out


def _clinical_filler(ctx: Ctx, count: int) -> list[str]:
    pool = ("Afebrile overnight.", "Tolerating diet.", "Pain controlled on current regimen.",
            "Lungs with scattered crackles at the bases.", "Abdomen soft, nontender.",
            "Telemetry without events.", "Ambulating in the hall with a walker.",
            "Urine output adequate.", "Mild bilateral ankle edema.", "Mood appropriate, engaged.",
            "No acute distress.", "BP 132/78, HR 84, SpO2 95% on room air.",
            "Incision clean, dry and intact.", "Glucose 140s to 180s.")
    return ctx.rng.sample(pool, k=count)


def _patient_line(ctx: Ctx, patient: Person | None) -> str:
    if patient is None:
        return ctx.choice((f"{ctx.age_phrase()} with CHF exacerbation.",
                           f"{ctx.age_phrase()} admitted with pneumonia.",
                           f"{ctx.age_phrase()} with COPD, here for follow-up."))
    return ctx.choice((
        f"Patient: {N(patient.full)}  MRN: {M('MRN', ctx.mrn())}  DOB: {M('DATE', ctx.dob())}",
        f"{N(patient.full)} is a {ctx.age_phrase()} with CHF exacerbation.",
        f"Pt {N(patient.full)}, {M('AGE', str(ctx.age()))}, admitted {M('DATE', ctx.date())}.",
    ))


def titles_build(ctx: Ctx, layout: str, w: Person | None, word: str, patient: Person | None,
                 positive: bool) -> tuple[str, list[str]]:
    rng = ctx.rng
    phenomena = ["title_before_word_name" if positive else "ordinary_word_after_title"]
    if not positive:
        phenomena.append("matched_negative")

    prose = list(PROSE.get(word, ()))
    if positive:
        body = _title_sentences(ctx, w, phenomena)
    else:
        body = rng.sample(TITLE_NEG, k=rng.randint(1, 2))
    if prose and (not positive or rng.random() < 0.45):
        body.append(ctx.choice(prose))
        phenomena.append("same_word_ordinary_use")
    rng.shuffle(body)
    date = M("DATE", ctx.date())
    if layout == "progress_note":
        head = ctx.choice((f"PROGRESS NOTE {date}", f"Hospital day 4 progress note, {date}", f"Daily note - {date}"))
        lines = [head, _patient_line(ctx, patient), "S: " + ctx.choice(PT_LINES).capitalize(),
                 "O: " + " ".join(_clinical_filler(ctx, 2)), "A/P: " + " ".join(body)]
    elif layout == "consult_note":
        service = ctx.choice(("Cardiology", "Nephrology", "Infectious Disease", "Palliative Care", "Hematology"))
        lines = [f"{service} consult, {date}", _patient_line(ctx, patient),
                 "Reason for consult: " + ctx.choice(("AKI on CKD.", "new atrial fibrillation.", "goals of care.",
                                                      "persistent bacteremia.", "anemia work-up.")),
                 " ".join(_clinical_filler(ctx, 2)), "Recommendations: " + " ".join(body)]
    elif layout == "ed_course_note":
        arrival = f"{rng.randint(0, 23):02d}{rng.choice(('05', '20', '35', '50'))}"
        lines = [f"ED COURSE ({date})", _patient_line(ctx, patient), f"{arrival}: arrived by EMS.",
                 " ".join(body), "Disposition: " + ctx.choice(("admit to medicine.", "discharge home.", "observation."))]
    elif layout == "secure_message":
        sender = ctx.person("plain")
        lines = [f"Secure message from {N(sender.full)}, RN, {date}",
                 "Re: " + ctx.choice(("bed 14", "room 3 overnight", "your patient on 5 West", "AM labs")),
                 " ".join(body), ctx.choice(("Thanks.", "Please advise.", "Will update.", "FYI only."))]
        phenomena.append("secure_message")
    elif layout == "portal_thread":
        pt = patient or ctx.person("plain")
        lines = [f"From: {N(pt.full)}  Sent: {date}",
                 ctx.choice(("I have a question about my new medicine.", "My refill hasn't gone through.",
                             "Following up on my test results.")),
                 "Reply from clinic: " + " ".join(body),
                 f"Call us at {M('PHONE', ctx.phone())} with questions."]
    else:  # sign_out_list
        items = []
        for i in range(rng.randint(2, 3)):
            items.append(f"{i + 1}. Bed {rng.randint(1, 30)} - {ctx.age_phrase()} - "
                         + ctx.choice(("CHF, diuresing.", "pneumonia day 3.", "post-op day 1.", "GI bleed, stable Hgb.")))
        items.append(f"{len(items) + 1}. Bed {rng.randint(31, 40)} - " + " ".join(body))
        lines = [f"Sign-out {date}"] + items
    return "\n".join(lines), phenomena


# ---------------------------------------------------------------------------
# Family: word-names in narrative and lists.
# ---------------------------------------------------------------------------

RELATIONS = ("daughter", "son", "wife", "husband", "sister", "brother", "niece", "nephew", "partner",
             "granddaughter", "grandson", "neighbor", "friend")
STAFF_ROLES = (("PT", "physical therapist"), ("OT", "occupational therapist"), ("RN", "nurse"),
               ("SW", "social worker"), ("CM", "case manager"), ("Chaplain", "chaplain"),
               ("RD", "dietitian"), ("SLP", "speech therapist"), ("NP", "nurse practitioner"))


def narrative_build(ctx: Ctx, layout: str, w: Person | None, word: str, patient: Person | None,
                    positive: bool) -> tuple[str, list[str]]:
    rng = ctx.rng
    in_list = layout in ("contact_list", "care_team_list")
    phenomena = [("word_name_in_list" if in_list else "word_name_in_narrative") if positive else "same_word_same_shape"]
    if not positive:
        phenomena.append("matched_negative")
    prose = list(PROSE.get(word, ()))
    rel = ctx.choice(RELATIONS)
    date = M("DATE", ctx.date())
    pt_line = _patient_line(ctx, patient) if patient else f"Patient is a {ctx.age_phrase()}."
    sentences = []
    if layout == "therapy_note":
        head = ctx.choice((f"PT daily note {date}", f"Occupational therapy treatment note, {date}",
                           f"Therapy session {date}"))
        if positive:
            helper = ctx.person(ctx.choice(("plain", "word_surname")))
            first = N(helper.initial_last)
            sentences.append(ctx.choice((
                f"Transfers with {first} and {N(w.word)}, mod assist x2.",
                f"Gait training 80 ft with {N(w.word)} guarding; {first} managed lines.",
                f"Co-treated with {N(w.full)}, OT.",
                f"Sit to stand x5 with {N(w.word)} cueing hand placement.",
            )))
            phenomena.append("word_name_in_list" if " and " in sentences[-1] else "word_name_in_narrative")
        else:
            sentences.append(ctx.choice(("Transfers with mod assist x2 and strong effort.",
                                         "Gait training 80 ft with contact guard.",
                                         "Sit to stand x5 with cues for hand placement.")))
        sentences += _clinical_filler(ctx, 1)
        sentences.append(ctx.choice(("Plan: progress to cane.", "Continue POC 3x/week.", "Goals updated.")))
    elif layout == "nursing_shift_note":
        head = ctx.choice((f"Nursing note {date} 0700-1900", f"Shift summary, {date}", f"RN note - {date}"))
        if positive:
            sentences.append(ctx.choice((
                f"Report given to {N(w.word)}, oncoming RN.",
                f"{N(w.full)} ({rel}) at bedside most of the shift.",
                f"Turned q2h with help from {N(w.word)}.",
                f"Patient asked for {N(w.word)}, her {rel}, to be called.",
            )))
        else:
            sentences.append(ctx.choice(("Report given to oncoming RN.", f"{rel.capitalize()} at bedside most of the shift.",
                                         "Turned q2h with assistance.")))
        sentences += _clinical_filler(ctx, 2)
    elif layout == "discharge_summary":
        head = ctx.choice((f"DISCHARGE SUMMARY  Discharged: {date}", f"Discharge summary, {date}"))
        sentences.append(f"Admitted with {ctx.choice(('pneumonia', 'CHF exacerbation', 'cellulitis', 'hip fracture'))}.")
        if positive:
            if w.kind == "word_given":
                sentences.append(ctx.choice((
                    f"Discharge home with {N(w.first)} and home health; {N(w.first)} is her {rel} and primary caregiver.",
                    f"Lives with her {rel}, {N(w.full)}, who will manage medications.",
                )))
            else:
                sentences.append(ctx.choice((
                    f"Follow up with Dr. {N(w.last)} in 2 weeks.",
                    f"Discharged to the care of her {rel}, {N(w.full)}.",
                    f"PCP: {N(w.full)}, MD; summary faxed.",
                )))
        else:
            sentences.append(ctx.choice(("Discharge home with home health.", "Follow up with PCP in 2 weeks.",
                                         f"Lives with her {rel}, who will manage medications.")))
        sentences.append(ctx.choice(("Condition at discharge: stable.", "Diet: low sodium.", "Activity as tolerated.")))
    elif layout == "contact_list":
        head = ctx.choice(("CONTACTS", "Emergency contacts", "Contact list (verified at admission)"))
        other = ctx.person("plain")
        entries = []
        if positive:
            entries.append(f"{N(w.full)} ({rel}) {M('PHONE', ctx.phone())}")
        entries.append(f"{N(other.full)} ({ctx.choice(RELATIONS)}) {M('PHONE', ctx.phone())}")
        if positive and rng.random() < 0.4:
            entries.append(f"Health care proxy: {N(w.word)}")
        if rng.random() < 0.4:
            entries.append(f"Email: {M('EMAIL', ctx.email(other))}")
        rng.shuffle(entries)
        sentences = [f"{i + 1}. {e}" for i, e in enumerate(entries)]
        if not positive:
            sentences.append(ctx.choice(("Pharmacy: none on file.", "No other contacts listed.")))
    elif layout == "care_team_list":
        head = ctx.choice(("Care team:", "CARE TEAM", "Treatment team this admission"))
        roles = rng.sample(STAFF_ROLES, k=3)
        entries = [f"Attending - Dr. {N(ctx.person('plain').last)}"]
        for index, (abbr, _) in enumerate(roles):
            if positive and index == 0:
                entries.append(f"{abbr} - {N(w.word if rng.random() < 0.5 else w.full)}")
            else:
                entries.append(f"{abbr} - {N(ctx.person('plain').full)}")
        if not positive:
            entries.append("Case manager - pending assignment")
        sentences = entries
    elif layout == "social_work_note":
        head = ctx.choice((f"Social work assessment {date}", f"SW note, {date}"))
        if positive:
            sentences.append(ctx.choice((
                f"Patient lives with her {rel}, {N(w.full)}, in a two-story house.",
                f"Neighbor {N(w.full)} checks on him daily and has a key.",
                f"{rel.capitalize()} {N(w.word)} will provide transport to dialysis.",
            )))
        else:
            sentences.append(ctx.choice((f"Patient lives with her {rel} in a two-story house.",
                                         "A neighbor checks on him daily and has a key.")))
        sentences.append(ctx.choice((f"Address on file: {M('ADDRESS', f'{rng.randint(10, 999)} Wenlow Lane, {ctx.town()} {rng.randint(10000, 99999)}')}.",
                                     f"Moved from {M('LOCATION', ctx.town())} last year.",
                                     "Insurance verified.")))
    else:  # home_health_visit_note
        head = ctx.choice((f"Home health skilled nursing visit {date}", f"HH visit note - {date}"))
        if positive:
            sentences.append(ctx.choice((
                f"Visit with patient and caregiver {N(w.full)}.",
                f"{N(w.word)} ({rel}) reports good adherence with the pill box.",
                f"Medications reviewed with {N(w.word)}, her {rel}.",
            )))
        else:
            sentences.append(ctx.choice(("Visit with patient and caregiver.",
                                         f"{rel.capitalize()} reports good adherence with the pill box.")))
        sentences += _clinical_filler(ctx, 1)
    if prose and (not positive or rng.random() < 0.45):
        sentences.append(ctx.choice(prose))
        phenomena.append("same_word_ordinary_use")
    lines = [head, pt_line] + sentences
    return "\n".join(lines), phenomena


# ---------------------------------------------------------------------------
# Family: very short names.
# ---------------------------------------------------------------------------


def _short_display(ctx: Ctx, s: Person) -> str:
    if s.culture in EASTERN_ORDER and ctx.rng.random() < 0.45:
        return s.eastern
    return s.full


def short_build(ctx: Ctx, layout: str, s: Person | None, word: str, patient: Person | None,
                positive: bool) -> tuple[str, list[str]]:
    rng = ctx.rng
    phenomena = ["very_short_name" if positive else "two_letter_abbreviation", ]
    if not positive:
        phenomena.append("matched_negative")
    decoys = list(SHORT_DECOYS.get(word, ())) + list(rng.sample(GENERIC_DECOYS, k=2))
    plain = ctx.person("plain")
    decoy_lines = [d.format(plain=N(plain.last)) for d in rng.sample(decoys, k=min(len(decoys), 2 if not positive else 1))]
    date = M("DATE", ctx.date())
    if layout == "note_header":
        if positive and rng.random() < 0.6:
            pt_name = N(_short_display(ctx, s))
            lines = [ctx.choice(("H&P", "ADMISSION NOTE", "Clinic note")),
                     f"Name: {pt_name}   Age: {M('AGE', str(ctx.age()))}   MRN: {M('MRN', ctx.mrn())}",
                     f"Visit date: {date}"]
            phenomena.append("short_name_patient_header")
        else:
            pt = patient or plain
            lines = [ctx.choice(("H&P", "ADMISSION NOTE", "Clinic note")),
                     f"Name: {N(pt.full)}   DOB: {M('DATE', ctx.dob())}   MRN: {M('MRN', ctx.mrn())}",
                     f"Visit date: {date}"]
            if positive:
                lines.append(ctx.choice((f"Attending: Dr. {N(s.last)}", f"PCP: {N(s.full)}, MD",
                                         f"Referring: Dr. {N(s.last)}")))
                phenomena.append("short_name_clinician_header")
        lines.append("CC: " + ctx.choice(("shortness of breath", "fall at home", "abdominal pain", "rash")))
    elif layout == "signature_block":
        lines = [ctx.choice((f"Follow-up note {date}", f"Clinic visit {date}")),
                 " ".join(_clinical_filler(ctx, 2))] + decoy_lines + [
                 ctx.choice(("Plan: continue current therapy.", "Return in 3 months.", "Labs next visit."))]
        decoy_lines = []
        if positive:
            lines.append(ctx.choice((
                f"Electronically signed by {N(s.full)}, RN on {M('DATE', ctx.date())}",
                f"-- {N(s.initial_last)}, PT",
                f"Dictated by: Dr. {N(s.last)}; transcribed by: {N(plain.initial_last)}",
                f"Signed: {N(_short_display(ctx, s))}, NP",
                f"Cosigned: {N(s.last)}, MD",
            )))
            phenomena.append("short_name_signature")
        else:
            lines.append(ctx.choice((f"Electronically signed by {N(plain.full)}, RN on {M('DATE', ctx.date())}",
                                     f"-- {N(plain.initial_last)}, PT", "Signed: on file")))
    elif layout == "dictation_transcript":
        stamps = ctx.stamps(6, False) if rng.random() < 0.5 else [None] * 6
        lines = [ctx.choice((f"Dictation transcript {date}", f"voice note {date}", f"Recorded consult {date}"))]
        if positive:
            style = ctx.choice(("title", "bare", "role_paren"))
            if style == "bare" and s.last.casefold() in ROLE_ONLY_LABELS:
                style = "title"  # a bare `son:` or `ma:` would read as a role-only label
            role = {"title": ctx.choice(("dr", "Dr.")), "bare": "", "role_paren": ctx.choice(("INTERPRETER", "NURSE", "dr"))}[style]
            if style == "bare":
                lines.append(f"Participants: {N(_short_display(ctx, s))} and the patient")
            label = speaker_label(style, s.last, role=role)
            phenomena += ["speaker_label_name", f"speaker_label_{style}"]
            other = "patient"
        else:
            label, other = ctx.choice((("dr", "patient"), ("nurse", "pt"), ("interpreter", "patient"), ("MA", "patient")))
            phenomena.append("role_only_label")
        speech_a = rng.sample(CLIN_LINES, k=3)
        speech_b = rng.sample(PT_LINES, k=3)
        if word == "oh" or rng.random() < 0.3:
            speech_b[0] = "oh, okay. " + speech_b[0]
            phenomena.append("interjection_oh")
        for index, decoy in enumerate(decoy_lines[:2]):
            speech_a[index + 1] = decoy
        decoy_lines = []
        for i in range(3):
            lines.append(turn(label, speech_a[i], stamps[2 * i]))
            lines.append(turn(other, speech_b[i], stamps[2 * i + 1]))
    elif layout == "referral_letter":
        signer = s if positive and rng.random() < 0.5 else ctx.person("plain")
        if positive and signer is not s:
            greeting = f"Dear Dr. {N(s.last)},"
        else:
            greeting = ctx.choice(("Dear Colleague,", f"Dear Dr. {N(plain.last)},"))
        pt = patient or ctx.person("plain")
        lines = [M("DATE", ctx.date()), greeting,
                 f"Thank you for seeing {N(pt.full)}, a {ctx.age_phrase()}, for "
                 + ctx.choice(("evaluation of anemia.", "a thyroid nodule.", "chronic cough.", "knee pain.")),
                 " ".join(_clinical_filler(ctx, 1))] + decoy_lines + ["Sincerely,"]
        decoy_lines = []
        lines.append(f"{N(_short_display(ctx, signer) if signer is s else signer.full)}, MD")
        lines.append(f"{M('FACILITY', ctx.facility())}  Tel {M('PHONE', ctx.phone())}")
        if positive and signer is s:
            phenomena.append("short_name_signature")
    else:  # procedure_note
        lines = [f"PROCEDURE NOTE {date}",
                 "Procedure: " + ctx.choice(("cystoscopy with stent placement", "EGD with biopsy",
                                             "cataract extraction OD", "closed reduction, left wrist")),
                 f"Indication: {ctx.age_phrase()} with symptoms for 3 months."]
        if positive:
            lines.append(ctx.choice((f"Operator: Dr. {N(s.last)}; Assistant: {N(plain.full)}, RN",
                                     f"Proceduralist: {N(s.full)}, MD", f"Anesthesia: Dr. {N(s.last)}")))
        else:
            lines.append(f"Operator: Dr. {N(plain.last)}; Assistant: RN on duty")
        lines += decoy_lines + ["Complications: none. EBL minimal."]
        decoy_lines = []
    lines += decoy_lines
    if not positive:
        phenomena.append("abbreviation_or_particle")
    return "\n".join(lines), phenomena


# ---------------------------------------------------------------------------
# Assembly.
# ---------------------------------------------------------------------------


def quotas(rng: random.Random, count: int, shares) -> list[str]:
    lower = round(count * shares[0])
    upper = round(count * shares[1])
    items = ["lower"] * lower + ["upper"] * upper + ["normal"] * (count - lower - upper)
    rng.shuffle(items)
    return items


def spread(rng: random.Random, count: int, options) -> list[str]:
    items = [options[i % len(options)] for i in range(count)]
    rng.shuffle(items)
    return items


def finish_row(marked: str, *, case: str, family: str, layout: str, number: int, role: str,
               pair_id: str, identity_group: str, people: list[Person], phenomena: list[str],
               cohort: str, word: str) -> dict:
    text, spans = parse_marked(marked)
    text = apply_case(text, case)
    for span in spans:
        span["text"] = text[span["start"]:span["end"]]
    present = [p for p in people if any(
        any(tok.casefold() in name_tokens(s["text"]) for tok in (p.first.casefold(), p.last.casefold()))
        for s in spans if s["label"] == "NAME")]
    groups = [identity_group] + [p.group for p in present if p.group != identity_group]
    groups = list(dict.fromkeys(groups))
    ph = list(dict.fromkeys(phenomena + [f"case_{case}"]))
    return {
        "id": f"tk22-{family}-{number:04d}",
        "text": text,
        "spans": spans,
        "split": "train",
        "suite": SUITE,
        "annotation_policy": POLICY_VERSION,
        "cohort": cohort,
        "family": f"tk22_{family}",
        "template_family": f"tk22_{family}",
        "layout": layout,
        "case": case,
        "role": role,
        "phenomena": ph,
        "identity_group": identity_group,
        "identity_groups": groups,
        "pair_id": pair_id,
        "word_name": word,
    }


BUILDERS = {"titles": titles_build, "narrative": narrative_build, "short_names": short_build}


def exclusion_hits(row: dict) -> int:
    hits = excluded_sequences(row["text"])
    hits += sum(excluded_value(s["text"]) for s in row["spans"] if s["label"] in ("NAME", "FACILITY", "LOCATION"))
    return hits


def row_name_tokens(row: dict) -> set[str]:
    return {t for s in row["spans"] if s["label"] == "NAME" for t in name_tokens(s["text"])}


_POOL_NAMES = ("WORD_SURNAMES", "WORD_GIVEN", "UNCOMMON_FIRST", "PLAIN_SURNAMES", "FAC_STEMS", "TOWNS",
               "CLIN_LINES", "PT_LINES", "FAMILY_LINES", "OPENERS", "TITLE_POS", "NURSE_POS",
               "FAMILY_POS", "TITLE_NEG", "GENERIC_DECOYS")
_FILTERED: dict | None = None


def filter_pools() -> dict:
    """Drop every pool value with a 1-8 word sequence on the exclusion list (counts only, values unseen).

    Idempotent; rebinds the module pools once so later draws never produce an excluded value.
    """
    global _FILTERED, SHORT_CULTURES
    if _FILTERED is not None:
        return _FILTERED
    removed = {}
    module = globals()
    for name in _POOL_NAMES:
        pool = module[name]
        kept = tuple(v for v in pool if not excluded_value(v.replace("{w}", "").replace("{plain}", "")))
        removed[name] = len(pool) - len(kept)
        module[name] = kept
    short_removed = 0
    cultures = {}
    for culture, (surnames, given) in SHORT_CULTURES.items():
        s_kept = tuple(v for v in surnames if not excluded_value(v))
        g_kept = tuple(v for v in given if not excluded_value(v))
        short_removed += len(surnames) - len(s_kept) + len(given) - len(g_kept)
        cultures[culture] = (s_kept, g_kept)
    SHORT_CULTURES = cultures
    removed["SHORT_CULTURES"] = short_removed
    for name in ("SPEECH", "PROSE", "SHORT_DECOYS", "TITLE_CRED_POS"):
        table = module[name]
        out, dropped = {}, 0
        for key, values in table.items():
            kept = tuple(v for v in values
                         if not excluded_sequences(v.replace("{plain}", "").replace("{w}", "")))
            dropped += len(values) - len(kept)
            if kept:
                out[key] = kept
        module[name] = out
        removed[name] = dropped
    _FILTERED = {"pool_values_removed": removed, "pool_values_removed_total": sum(removed.values())}
    return _FILTERED


def generate(seed: int = SEED) -> tuple[list[dict], dict]:
    pool_stats = filter_pools()
    rng = random.Random(seed)
    names = Names()
    rows: list[dict] = []
    stats = {**pool_stats, "exclusion_rejected_attempts": 0, "exclusion_rejected_by_family": Counter(),
             "duplicate_text_retries": 0, "rows_checked": 0}
    seen_texts: set[str] = set()
    number = 0
    for family in PLAN:
        positives, negatives = PLAN[family]
        layouts = spread(rng, positives, LAYOUTS[family])
        cases = quotas(rng, positives + negatives, CASE_SHARES[family])
        case_iter = iter(cases)
        paired = set(rng.sample(range(positives), k=negatives))
        if family == "speaker_labels":
            keys = set(SPEECH)
        elif family == "short_names":
            keys = None
        else:
            keys = set(PROSE)
        for index in range(positives):
            layout = layouts[index]
            case_pos = next(case_iter)
            case_neg = next(case_iter) if index in paired else None
            for _attempt in range(60):
                ctx = Ctx(random.Random(rng.getrandbits(64)), names)
                need_keys = keys if index in paired else None
                if family == "short_names":
                    s_keys = set(SHORT_DECOYS) if (index in paired and ctx.rng.random() < 0.5) else None
                    w = ctx.person("short", keys=s_keys)
                else:
                    w = ctx.word_person(keys=need_keys)
                word = w.word.casefold()
                patient = ctx.person("plain") if ctx.rng.random() < 0.55 else None
                if family == "speaker_labels":
                    marked, ph = speaker_positive(ctx, layout, w)
                else:
                    marked, ph = BUILDERS[family](ctx, layout, w, word, patient, True)
                number += 1
                pair_id = f"tk22-pair-{family}-{index:04d}"
                row = finish_row(marked, case=case_pos, family=family, layout=layout, number=number,
                                 role="positive", pair_id=pair_id, identity_group=w.group,
                                 people=ctx.people, phenomena=ph, cohort="unusual_name", word=w.word)
                batch = [row]
                if case_neg is not None:
                    if family == "speaker_labels":
                        neg_marked, neg_ph = speaker_negative(ctx, layout, word)
                    else:
                        neg_marked, neg_ph = BUILDERS[family](ctx, layout, None, word, patient, False)
                    number += 1
                    neg = finish_row(neg_marked, case=case_neg, family=family, layout=layout, number=number,
                                     role="matched_negative", pair_id=pair_id, identity_group=w.group,
                                     people=ctx.people, phenomena=neg_ph, cohort="realistic", word=w.word)
                    batch.append(neg)
                stats["rows_checked"] += len(batch)
                if any(exclusion_hits(r) for r in batch):
                    stats["exclusion_rejected_attempts"] += 1
                    stats["exclusion_rejected_by_family"][family] += 1
                    continue
                if any(r["text"] in seen_texts for r in batch) or (len(batch) == 2 and batch[0]["text"] == batch[1]["text"]):
                    stats["duplicate_text_retries"] += 1
                    continue
                if any(annotation_errors(r["text"], r["spans"]) for r in batch):
                    raise ValueError(f"structural error in {row['id']}")
                for r in batch:
                    seen_texts.add(r["text"])
                    names.accept(row_name_tokens(r))
                    rows.append(r)
                break
            else:
                raise RuntimeError(f"could not author {family} row {index}: {stats}")
    # Stable ids in final order.
    for i, r in enumerate(rows, 1):
        fam = r["family"].removeprefix("tk22_")
        r["id"] = f"tk22-{fam}-{i:04d}"
    stats["exclusion_rejected_by_family"] = dict(stats["exclusion_rejected_by_family"])
    stats["final_rows_with_exclusion_hits"] = sum(1 for r in rows if exclusion_hits(r))
    return rows, stats


def support(rows: list[dict]) -> dict:
    counts = Counter(s["label"] for r in rows for s in r["spans"])
    return {label: counts[label] for label in ENTITY_TYPES}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_for(rows: list[dict], payload: bytes, seed: int, stats: dict) -> dict:
    labels = support(rows)
    name_rows = Counter(t for r in rows for t in row_name_tokens(r))
    policy_modules = ("annotation_policy.py", "annotation_policy_places.py", "annotation_policy_care_sites.py",
                      "annotation_policy_targeted.py", "annotation_policy_transcripts.py", "annotation_policy_ages.py")
    return {
        "name": "tk22_train_word_surnames",
        "suite": SUITE,
        "annotation_policy": POLICY_VERSION,
        "annotation_policy_documents": ["docs/annotation-policy-v2.md", "docs/annotation-policy-v2-places.md"],
        "brief": "docs/word-surnames-bank-brief.md",
        "split": "train",
        "role": "train",
        "independent_final": False,
        "rows": len(rows),
        "documents": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "label_support": labels,
        "entity_support": labels,
        "label_counts": labels,
        "cohort_counts": dict(sorted(Counter(r["cohort"] for r in rows).items())),
        "family_counts": dict(sorted(Counter(r["family"] for r in rows).items())),
        "role_counts": dict(sorted(Counter(r["role"] for r in rows).items())),
        "layout_counts": dict(sorted(Counter(r["layout"] for r in rows).items())),
        "case_counts": dict(sorted(Counter(r["case"] for r in rows).items())),
        "max_rows_per_name_value": max(name_rows.values()) if name_rows else 0,
        "phi_free_rows": sum(1 for r in rows if not r["spans"]),
        "exclusion_filter": {
            "list": "data/places_exclusions.sha256.json",
            "list_sha256": EXCLUSIONS_SHA256,
            "method": "reject and redraw any row pair with a 1-8 word sequence or NAME/FACILITY/LOCATION value on the hash list",
            **stats,
        },
        "synthetic_schemes": {"MRN": MRN_SCHEME, "PHONE": PHONE_SCHEME,
                              "EMAIL": EMAIL_SCHEME, "ID": ID_SCHEME,
                              "places": "invented facility and town names only; no real place values"},
        "author_provenance": {
            "generator": "scripts/takeover/author22_word_surnames.py",
            "generator_sha256": sha256_file(Path(__file__).resolve()),
            "seed": seed,
            "policy_module_sha256": {m: sha256_file(ROOT / "src" / "outis" / "data" / m) for m in policy_modules},
            "types_sha256": sha256_file(ROOT / "src" / "outis" / "types.py"),
            "isolation": "authored from the brief, the policy and addendum, types.py, policy fixture modules, "
                         "validate_bank.py, AGENTS.md and CLEANUP.md only; no corpus, bank or report read",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    rows, stats = generate(args.seed)
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    manifest = manifest_for(rows, payload, args.seed, stats)
    out.with_suffix(out.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "sha256", "label_support", "cohort_counts",
                                               "layout_counts", "case_counts", "max_rows_per_name_value",
                                               "exclusion_filter")}, indent=1))


if __name__ == "__main__":
    main()
