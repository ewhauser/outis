"""Authored contexts for the clinical-phi-v2-places generator.

Original prose for this project (CC0-1.0). Each template belongs to exactly one
split; train and validation families never share wording. Slots carry their
authored role, which alone decides gold labels (never a dictionary lookup):

* labelled slots: ``loc``/``site_loc`` LOCATION, ``site`` FACILITY,
  ``clin``/``clin_last``/``pt``/``person``/``person_first`` NAME, ``date``/``year``
  DATE, ``mrn`` MRN, ``phone`` PHONE;
* public-reference slots (never labelled): ``cite`` (N3), ``finding``/``device``/
  ``cohort_study`` (N1), ``dz`` (N2), ``state``/``country``/``region``/
  ``loc_state`` (N4), ``word`` (N5);
* neutral fillers: ``age`` (18-89), ``drug``, ``symptom``, ``n``, ``days``.

The real-value lists below are public reference names, not a detector
dictionary; every value is filtered through ``place_value_pool`` before use.
"""

from __future__ import annotations

LAYOUTS = ("query", "shorthand", "narrative", "field")
LABELLED_SLOTS = {
    "loc": "LOCATION", "site_loc": "LOCATION", "site": "FACILITY",
    "clin": "NAME", "clin_last": "NAME", "pt": "NAME", "person": "NAME", "person_first": "NAME",
    "date": "DATE", "year": "DATE", "mrn": "MRN", "phone": "PHONE",
}
REFERENCE_SLOTS = ("cite", "finding", "device", "cohort_study", "dz", "state", "country", "region", "loc_state", "word")
FILLER_SLOTS = ("age", "drug", "symptom", "n", "days")

DRUGS = ("apixaban", "metoprolol", "lisinopril", "metformin", "atorvastatin", "levothyroxine", "sertraline",
         "gabapentin", "amlodipine", "furosemide", "prednisone", "insulin glargine", "albuterol", "omeprazole",
         "warfarin", "tamsulosin", "losartan", "duloxetine", "montelukast", "rosuvastatin")
SYMPTOMS = ("chest pain", "shortness of breath", "dizziness", "a cough", "abdominal pain", "fatigue", "headaches",
            "back pain", "palpitations", "leg swelling", "fever", "nausea", "a rash", "joint pain", "blurred vision")

# N1: place inside a named instrument, criterion, classification, score, rule,
# device or cohort study. (place key, surface, result phrases containing it)
FINDINGS = (
    ("Chicago", "Chicago Classification", ("type II achalasia per the Chicago Classification", "EGJ outflow obstruction by Chicago Classification v4.0", "ineffective esophageal motility per the Chicago Classification")),
    ("Boston", "Boston criteria", ("probable cerebral amyloid angiopathy by the Boston criteria", "possible CAA per the modified Boston criteria")),
    ("Boston", "Boston Bowel Preparation Scale", ("Boston Bowel Preparation Scale of {n}", "adequate prep, Boston Bowel Preparation Scale {n}")),
    ("Seattle", "Seattle Heart Failure Model", ("Seattle Heart Failure Model estimated 1-year survival of 9{n}%", "Seattle Heart Failure Model mean survival {n} years")),
    ("Seattle", "Seattle Angina Questionnaire", ("Seattle Angina Questionnaire summary score of 6{n}",)),
    ("Ottawa", "Ottawa ankle rules", ("negative Ottawa ankle rules", "Ottawa ankle rules negative, no films needed")),
    ("Ottawa", "Ottawa knee rules", ("negative Ottawa knee rules",)),
    ("Ottawa", "Ottawa subarachnoid hemorrhage rule", ("low risk by the Ottawa subarachnoid hemorrhage rule",)),
    ("San Francisco", "San Francisco Syncope Rule", ("low risk by the San Francisco Syncope Rule", "San Francisco Syncope Rule negative")),
    ("Framingham", "Framingham Risk Score", ("Framingham Risk Score of 1{n}%", "10-year Framingham Risk Score {n}%")),
    ("Framingham", "Framingham criteria", ("heart failure meeting Framingham criteria", "two major Framingham criteria")),
    ("Rochester", "Rochester criteria", ("low risk by the Rochester criteria",)),
    ("Philadelphia", "Philadelphia criteria", ("low risk per the Philadelphia criteria",)),
    ("Atlanta", "revised Atlanta classification", ("moderately severe pancreatitis per the revised Atlanta classification", "mild acute pancreatitis by the revised Atlanta classification")),
    ("Los Angeles", "Los Angeles classification", ("grade B esophagitis by the Los Angeles classification", "Los Angeles classification grade A esophagitis")),
    ("Dallas", "Dallas criteria", ("borderline myocarditis by the Dallas criteria",)),
    ("Bethesda", "Bethesda System", ("Bethesda System category III nodule", "Bethesda System category II, benign")),
    ("Bethesda", "revised Bethesda guidelines", ("tumor meeting the revised Bethesda guidelines",)),
    ("Amsterdam", "Amsterdam II criteria", ("family history meeting Amsterdam II criteria",)),
    ("Berlin", "Berlin definition", ("moderate ARDS by the Berlin definition", "mild ARDS per the Berlin definition")),
    ("Berlin", "Berlin Questionnaire", ("high-risk Berlin Questionnaire for sleep apnea",)),
    ("Milan", "Milan criteria", ("single lesion within the Milan criteria", "tumor burden beyond the Milan criteria")),
    ("Ann Arbor", "Ann Arbor staging", ("Ann Arbor stage IIA disease", "stage IIIB by Ann Arbor staging")),
    ("Paris", "Paris classification", ("Paris 0-IIa polyp", "flat lesion, Paris classification 0-IIb")),
    ("Montreal", "Montreal classification", ("Crohn disease, Montreal classification A2 L3 B1",)),
    ("Glasgow", "Glasgow Coma Scale", ("Glasgow Coma Scale of 1{n}", "Glasgow Coma Scale 15 on arrival")),
    ("Glasgow", "Glasgow-Blatchford score", ("Glasgow-Blatchford score of {n}",)),
    ("Richmond", "Richmond Agitation-Sedation Scale", ("Richmond Agitation-Sedation Scale of -{n}", "Richmond Agitation-Sedation Scale 0")),
    ("Kansas City", "Kansas City Cardiomyopathy Questionnaire", ("Kansas City Cardiomyopathy Questionnaire score of 5{n}",)),
    ("Pittsburgh", "Pittsburgh Sleep Quality Index", ("Pittsburgh Sleep Quality Index of 1{n}",)),
    ("Pittsburgh", "Pittsburgh knee rules", ("negative Pittsburgh knee rules",)),
    ("Edinburgh", "Edinburgh Postnatal Depression Scale", ("Edinburgh Postnatal Depression Scale score of 1{n}",)),
    ("Cincinnati", "Cincinnati Prehospital Stroke Scale", ("positive Cincinnati Prehospital Stroke Scale", "Cincinnati Prehospital Stroke Scale negative")),
    ("Vancouver", "Vancouver Scar Scale", ("Vancouver Scar Scale of {n}",)),
    ("Michigan", "Michigan Neuropathy Screening Instrument", ("Michigan Neuropathy Screening Instrument score of {n}",)),
    ("St. Louis", "St. Louis University Mental Status exam", ("St. Louis University Mental Status exam score of 2{n}",)),
    ("New York", "New York Heart Association class", ("New York Heart Association class III symptoms", "New York Heart Association class II dyspnea")),
)
DEVICES = (
    ("Philadelphia", "Philadelphia collar"), ("Miami", "Miami J collar"), ("Aspen", "Aspen collar"),
    ("Milwaukee", "Milwaukee brace"), ("Boston", "Boston brace"), ("Denver", "Denver shunt"),
    ("Minnesota", "Minnesota tube"),
)
COHORT_STUDIES = (
    ("Framingham", "Framingham Heart Study"), ("Jackson", "Jackson Heart Study"),
    ("Bogalusa", "Bogalusa Heart Study"), ("Framingham", "Framingham Offspring Study"),
)
# N2: place inside a disease, pathogen or eponymous condition.
DISEASES = (
    ("Lyme", "Lyme disease"), ("Lyme", "Lyme arthritis"), ("Lyme", "Lyme carditis"),
    ("West Nile", "West Nile virus"), ("Rocky Mountain", "Rocky Mountain spotted fever"),
    ("Norwalk", "Norwalk virus"), ("Coxsackie", "coxsackievirus"), ("Pontiac", "Pontiac fever"),
    ("St. Louis", "St. Louis encephalitis"), ("Colorado", "Colorado tick fever"),
    ("Jamestown", "Jamestown Canyon virus"), ("La Crosse", "La Crosse encephalitis"),
    ("Bourbon", "Bourbon virus"), ("Haverhill", "Haverhill fever"),
)
COUNTRIES = ("Mexico", "Canada", "Guatemala", "Honduras", "El Salvador", "India", "China", "Vietnam", "Nigeria",
             "Ethiopia", "Somalia", "Brazil", "Peru", "Haiti", "Cuba", "Ukraine", "Poland", "Italy", "Germany",
             "Ireland", "Japan", "South Korea", "Thailand", "Kenya", "Ghana", "Colombia", "Pakistan", "Egypt")
REGIONS = ("the Midwest", "New England", "the Pacific Northwest", "the Southwest", "the Gulf Coast",
           "Central America", "Southeast Asia", "West Africa", "the Caribbean", "Puerto Rico", "East Africa")
STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}
# States alone that are also a common city reading are not used as N4 values.
AMBIGUOUS_STATES = ("New York", "Washington", "District of Columbia")
# N5: a place-name word used as an ordinary word. (place word, third-person
# predicate fragments containing it). Words are kept only if the Gazetteer
# confirms a place of exactly that name.
PLACE_WORDS = (
    ("Reading", ("needs new reading glasses", "has trouble reading small print")),
    ("Mobile", ("uses a mobile phone for video visits", "was seen by the mobile crisis team")),
    ("Normal", ("received a liter of normal saline", "has normal sinus rhythm on the monitor")),
    ("Orange", ("drinks orange juice every morning", "noticed orange urine after starting rifampin")),
    ("Media", ("has recurrent otitis media", "had otitis media with effusion")),
    ("Walker", ("walks with a walker", "keeps a walker by the bed")),
    ("Accident", ("was in a car accident {n} years ago", "had a workplace accident last winter")),
    ("Independence", ("values independence at home", "wants to keep some independence")),
    ("Bath", ("needs help getting in and out of the bath", "slipped getting out of the bath")),
    ("Comfort", ("is now on comfort care", "prefers comfort measures only")),
    ("Energy", ("reports low energy since the infection", "has more energy on the new dose")),
    ("Early", ("reports early satiety", "wakes up early with headaches")),
    ("Marathon", ("is training for a marathon", "ran a marathon last fall")),
    ("Protection", ("uses sun protection on the new graft", "wears hearing protection at work")),
    ("Hurricane", ("lost medications in the hurricane", "was displaced by the hurricane")),
)
# N5: place-name words used as given names or surnames of invented people.
PLACE_GIVEN = ("Florence", "Austin", "Dallas", "Madison", "Savannah", "Charlotte", "Aurora", "Phoenix", "Eugene",
               "Tyler", "Marion", "Chester", "Sharon", "Shelby", "Augusta", "Beatrice", "Camden", "Victoria",
               "Raleigh", "Orlando", "Lafayette", "Brooklyn", "Denver", "Houston")
PLACE_SURNAMES = ("Jackson", "Lincoln", "Cleveland", "Houston", "Denver", "Madison", "Monroe", "Warren",
                  "Clinton", "Franklin", "Hamilton", "Columbus", "Springfield", "Austin", "Sheridan")


def _t(*items):
    return tuple(items)


# Templates: TEMPLATES[split][group][layout] -> texts. Groups:
# p1 (locality), care_site (P2/P3/P4 by value form), p5, p6, mixed,
# n1_finding, n1_device, n1_study, n2, n3, n3_clinician, n4_state, n4_country,
# n4_region, n5_word, n5_person.
TEMPLATES = {
    "train": {
        "p1": {
            "query": _t(
                "pt from {loc} with new afib, anticoag options?",
                "my mom lives in {loc} and keeps falling, what kind of doctor should she see",
                "{loc} pt w/ recurrent cellulitis, abx choice",
                "dad just moved to {loc}, how do i transfer his {drug} prescription",
                "got back from {loc} last week and now fever and a rash, should i go in",
                "son in {loc} wants to visit after my surgery is that ok",
                "traveled to {loc} for a wedding, calf swollen since the flight",
                "elderly aunt in {loc} needs home health, who do we call",
            ),
            "shorthand": _t(
                "{age}M from {loc} c/o {symptom} x{days}d.",
                "{age}F, lives alone in {loc}, presents after fall.",
                "Pt recently relocated from {loc}; records requested.",
                "Dtr lives in {loc}, is primary contact.",
                "Visiting from {loc}. No PCP locally. Out of {drug}.",
                "SH: lives w/ spouse in {loc}, retired, no tobacco.",
                "Recent travel to {loc}; no sick contacts.",
                "Pt drives in from {loc} for HD MWF.",
                "{age}M from {loc}; no travel outside {loc_state} this year.",
            ),
            "narrative": _t(
                "The patient moved from {loc} last spring and now lives with her daughter.",
                "He was born and raised in {loc} and went back there after retiring.",
                "Her husband stayed behind in {loc} while she completed treatment.",
                "She reports that the {symptom} began during a family visit to {loc}.",
                "He lives in {loc} with his partner and two dogs.",
                "The patient's son flew in from {loc} to help with discharge planning.",
                "She commutes from {loc} three days a week for infusions.",
                "He lives in {loc} and has not left {loc_state} in the past year.",
            ),
            "field": _t(
                "City: {loc}",
                "Residence: {loc}\nLiving situation: with spouse",
                "Home city | {loc}",
                "Patient: {pt}\nDOB: {date}\nCity of residence: {loc}",
                "Emergency contact: daughter ({loc})",
                "- Lives in: {loc}\n- Meds: {drug}",
                "Travel history: {loc} ({days} days ago)",
                "Contact phone: {phone}\nHome: {loc}",
            ),
        },
        "care_site": {
            "query": _t(
                "seen at {site} last month, still waiting on biopsy results",
                "transferred from {site} to here, how do i get my imaging",
                "my dad had his bypass at {site}, is cardiac rehab covered",
                "{site} sent me home with a drain, when does it come out",
                "pt followed by {site} hepatology, which labs to send?",
                "had a sleep study at {site} and they want a cpap, is that normal",
                "she was in the er at {site} for {symptom} and they did nothing",
            ),
            "shorthand": _t(
                "Transferred from {site} for higher level of care.",
                "s/p lap chole at {site} {year}.",
                "Onc f/u at {site}, last seen {date}.",
                "PCP @ {site}. Meds reconciled; continues {drug}.",
                "Recent adm {site} for CHF exacerbation.",
                "Records from {site} reviewed; CT neg for PE.",
                "{age}F s/p TKA at {site}, here for PT.",
                "Echo study done at {site}, EF 4{n}%.",
            ),
            "narrative": _t(
                "She was seen at {site} in the spring for a second opinion.",
                "He was transferred from {site} after the CT showed a bleed.",
                "Her cardiologist at {site} adjusted the {drug} two weeks ago.",
                "The patient completed radiation at {site} and now follows up here.",
                "They drove to {site} for the sleep study, which showed severe apnea.",
                "He has received all of his diabetes care at {site} since {year}.",
                "She delivered her first child at {site} without complications.",
            ),
            "field": _t(
                "Transferring facility: {site}",
                "Prior care: {site}",
                "Referring site | {site}",
                "Outside records: {site} ({date})",
                "- Oncology: {site}\n- Cardiology: none",
                "MRN: {mrn}\nAdmitted from: {site}",
                "Last colonoscopy: {year}, {site}",
            ),
        },
        "p5": {
            "query": _t(
                "had surgery at {site} in {site_loc}, now the wound is red",
                "should i do chemo at {site} in {site_loc} or closer to home",
                "flew to {site} in {site_loc} for a second opinion, they said wait",
                "my brother is at {site} in {site_loc}, what does stable condition mean",
            ),
            "shorthand": _t(
                "Tx'd at {site}, {site_loc}.",
                "Prior CABG {site} ({site_loc}) {year}.",
                "Transferred from {site} in {site_loc} via air.",
                "{age}M, recent adm {site}, {site_loc}, for {symptom}.",
                "Followed by neuro at {site} in {site_loc}.",
            ),
            "narrative": _t(
                "She underwent the procedure at {site} in {site_loc} and recovered without complications.",
                "He was airlifted to {site} in {site_loc} after the accident on the highway.",
                "She was treated at {site} in {site_loc} for six weeks.",
                "His first transplant evaluation took place at {site}, {site_loc}.",
            ),
            "field": _t(
                "Outside facility: {site}, {site_loc}",
                "Transfer origin | {site} | {site_loc}",
                "Prior hospitalization: {site} ({site_loc})",
                "Surgical history: appendectomy, {site}, {site_loc}",
            ),
        },
        "p6": {
            "query": _t(
                "dr {clin_last} from {site} wants an mri, will insurance cover it",
                "my surgeon at {site}, dr. {clin_last}, retired, who takes over",
                "{clin} at {site} said to stop {drug} before the procedure, for how long",
                "can dr {clin_last} at {site} see my daughter too",
            ),
            "shorthand": _t(
                "D/w Dr. {clin_last} ({site} cards), agrees w/ plan.",
                "Seen by {clin}, MD, {site} neuro.",
                "Per Dr. {clin_last} at {site}: continue {drug}.",
                "Case reviewed w/ Dr. {clin} of {site}; will accept transfer.",
                "Tumor board: Dr. {clin_last} ({site}) recommends resection.",
            ),
            "narrative": _t(
                "Dr. {clin} from {site} reviewed the scan and agreed with surgery.",
                "Her oncologist, Dr. {clin_last} at {site}, recommended a repeat biopsy.",
                "We spoke with Dr. {clin} of {site}, who will accept the transfer.",
                "His rheumatologist, {clin}, practices at {site} and manages the infusions.",
            ),
            "field": _t(
                "Referring provider: Dr. {clin}, {site}",
                "Consultant | {clin} | {site}",
                "PCP: {clin} ({site})",
                "Attending: Dr. {clin_last}\nPrimary site: {site}",
            ),
        },
        "mixed": {
            "query": _t(
                "seen at {site}, {cite} guidelines say start aspirin, should i",
                "had my ablation at {site} and the {cite} website says no driving for a week?",
                "followed at {site} for crohns, does {cite} recommend yearly scopes",
            ),
            "shorthand": _t(
                "Seen at {site}; per {cite} guidelines, start ASA.",
                "F/u {site} onc. Dosing per {cite} published protocol.",
                "Pt treated at {site}. {cite} handout given re: low-FODMAP diet.",
                "{age}F followed at {site}; read {cite} review of {drug}, questions answered.",
            ),
            "narrative": _t(
                "She was treated at {site}, and her husband printed the {cite} patient guide on recovery.",
                "He receives his care at {site}; we discussed a {cite} study on the drug.",
                "After her surgery at {site}, she followed the {cite} recommendations for wound care.",
            ),
            "field": _t(
                "Outside care: {site}\nReference: {cite} guideline",
                "Treating center | {site}\nEducation | {cite} handout",
                "Surgery site: {site}\nDiet plan source: {cite}",
            ),
        },
        "n1_finding": {
            "query": _t(
                "pt w/ {finding}, next steps? already on {drug}",
                "{finding}, does that mean surgery at {age}",
                "{age} y/o with {finding} what now",
                "how worried should i be about {finding}, been having {symptom}",
                "my dad ({age}) was told {finding}, what does that mean",
            ),
            "shorthand": _t(
                "{age}F. Assessment: {finding}.",
                "{age}F, {finding}.",
                "Imp: {finding}; cont {drug}, RTC {n} wk.",
                "{age}M w/ {symptom}; {finding}. Continue {drug}.",
                "HPI: {symptom} x{days}d. Exam/scoring: {finding}.",
            ),
            "narrative": _t(
                "Her workup for {symptom} was notable for {finding}.",
                "He was found to have {finding} and was counseled about {drug}.",
                "Review of the chart shows {finding}, unchanged over {n} years.",
                "She asked what {finding} means for her plans at age {age}.",
                "Over the last {days} days his {symptom} persisted, and testing showed {finding}.",
            ),
            "field": _t(
                "Assessment: {finding}\nPlan: {drug}",
                "Score | {finding} | day {days}",
                "- Result: {finding}\n- Meds: {drug}",
                "Impression: {finding}\nAge: {age}",
                "Chief complaint: {symptom}\nKey finding: {finding}",
            ),
        },
        "n1_device": {
            "query": _t("how long does he need the {device}, its been {days} days", "can i shower with a {device} on, {age} yo with {symptom}"),
            "shorthand": _t("{age}M in {device}, NV intact.", "Remains in {device}; f/u {days}d. Cont {drug}."),
            "narrative": _t("She was fitted for a {device} before discharge and given {drug} for pain.", "He has worn the {device} for {n} weeks without skin breakdown."),
            "field": _t("Device: {device}\nDays in place: {days}", "Orthosis | {device} | since day {n}"),
        },
        "n1_study": {
            "query": _t("does {cohort_study} data apply to women my age ({age})", "im {age} on {drug}, what did the {cohort_study} find"),
            "shorthand": _t("{age}F; risk discussed using {cohort_study} data.", "CV risk ({cohort_study} based) reviewed; cont {drug}."),
            "narrative": _t("We reviewed risk estimates derived from the {cohort_study} with him before starting {drug}.", "At {age}, her estimated risk from the {cohort_study} equations was modest."),
            "field": _t("Risk model source: {cohort_study}\nStatin: {drug}", "Risk basis | {cohort_study} | age {age}"),
        },
        "n2": {
            "query": _t(
                "tested neg for {dz} but still have {symptom}, why",
                "can you get {dz} twice? had it {n} years ago",
                "{dz} symptoms in kids? my son is {n} and has {symptom}",
                "pt {age}F with ?{dz}, which test",
                "does {drug} interact with treatment for {dz}",
            ),
            "shorthand": _t(
                "{dz} serologies neg x{n}.",
                "Hx {dz} {n} yrs ago, treated, no sequelae.",
                "R/o {dz}; f/u titers in {days}d.",
                "{age}M w/ fever, ?{dz}.",
                "{age}F, {symptom} after hiking; {dz} considered.",
            ),
            "narrative": _t(
                "She was treated for {dz} {n} years ago and has had no recurrence.",
                "Testing for {dz} was sent given the fever and {symptom}.",
                "His {symptom} was initially attributed to {dz}.",
                "We discussed how {dz} is usually diagnosed and why {drug} was continued.",
                "At {age}, she had never been tested for {dz} before this visit.",
            ),
            "field": _t(
                "PMH: {dz}, on {drug}",
                "Infectious workup | {dz} - negative | day {days}",
                "- Problem list: {dz}, HTN\n- Age: {age}",
                "Pending labs: {dz} testing (sent {days}d ago)",
                "Presenting symptom: {symptom}\nConsidered: {dz}",
            ),
        },
        "n3": {
            "query": _t(
                "{cite} guidelines on statins after {age}?",
                "what does {cite} recommend for {symptom} after surgery",
                "a {cite} study said {drug} helps memory, true?",
                "read on {cite} site that {symptom} can be from anxiety, is that right",
                "{cite} says take {drug} at night, does it matter",
            ),
            "shorthand": _t(
                "Per {cite} guidelines, start ASA 81. {age}F.",
                "{drug} dosing per {cite} recs (online).",
                "Pt read {cite} handout re: {symptom}, questions answered.",
                "{cite} study cited by pt; d/w risks of {drug}.",
                "{age}M asks re {cite} article on {drug}; reviewed.",
            ),
            "narrative": _t(
                "She had read a {cite} study suggesting {drug} could help her {symptom}.",
                "We reviewed the {cite} recommendations on screening intervals with him at age {age}.",
                "He found the {cite} website helpful for understanding why he has {symptom}.",
                "According to guidance published by {cite}, the {drug} dose was not changed.",
                "Her daughter sent her a {cite} video about {symptom}, which we discussed.",
            ),
            "field": _t(
                "Reference: {cite} guideline\nMedication: {drug}",
                "Education material | {cite} handout | {symptom}",
                "Source: {cite} recommendations ({n} pages)",
                "Diet plan: {cite} low-sodium sheet\nAge: {age}",
                "Patient question: {drug} timing\nAnswered from: {cite} guide",
            ),
        },
        "n3_clinician": {
            "query": _t("dr {clin_last} printed some {cite} info on {drug}, is it current"),
            "shorthand": _t("Dr. {clin_last} reviewed {cite} guidelines w/ pt.", "Per {clin}, MD: follow {cite} protocol for taper."),
            "narrative": _t("Dr. {clin} walked her through the {cite} recommendations for {drug}.", "Her nurse practitioner, {clin}, shared a {cite} article about the condition."),
            "field": _t("Counseling by: Dr. {clin_last}\nMaterials: {cite} handout"),
        },
        "n4_state": {
            "query": _t(
                "pt from {state}, {age} yo, need any extra screening?",
                "moving to {state} next month, how do i refill my {drug}",
                "is valley fever a thing in {state}? have {symptom}",
                "my parents live in {state}, can i manage their {drug} remotely",
                "visiting {state} for {n} weeks, is {symptom} from the altitude",
            ),
            "shorthand": _t(
                "{age}F from {state}, moved here for work.",
                "Recent travel to {state}; no sick contacts. {symptom} x{days}d.",
                "Lives in {state}; here visiting family. Out of {drug}.",
                "{age}F, family in {state}, no local supports.",
                "{age}M, grew up in {state}, nonsmoker.",
            ),
            "narrative": _t(
                "She grew up in {state} and moved here for college; she has had {symptom} since.",
                "He recently returned from {n} weeks in {state}.",
                "Most of her family still lives in {state}, and she takes {drug} as prescribed.",
                "He lives in {state} but spends winters with his sister; he is {age}.",
                "Her {symptom} began on a road trip through {state}.",
            ),
            "field": _t(
                "State of residence: {state}\nAge: {age}",
                "Travel: {state} ({days} days ago)",
                "Licensed pharmacy state | {state} | {drug}",
                "- Insurance: {state} Medicaid\n- Age: {age}",
                "Prior residence: {state} ({n} yrs)",
            ),
        },
        "n4_country": {
            "query": _t("back from {country} {days} days ago, diarrhea since", "pt from {country}, {age} yo, need TB screening?", "visiting family in {country} next month, should i bring extra {drug}"),
            "shorthand": _t("{age}M, emigrated from {country} {n} yrs ago.", "Recent travel to {country}; malaria ppx not taken. {symptom} x{days}d."),
            "narrative": _t("She was born in {country} and came here as a child; she now takes {drug}.", "He spent {n} weeks visiting relatives in {country} and returned with {symptom}."),
            "field": _t("Country of origin: {country}\nInterpreter: requested\nAge: {age}", "Travel history | {country} | {days} days ago"),
        },
        "n4_region": {
            "query": _t("traveling around {region} for {n} weeks this summer, which vaccines", "{age} y/o moving to {region}, will {symptom} get better"),
            "shorthand": _t("{age}F, recent travel to {region}; {symptom} x{days}d.", "Returned from {region} {days}d ago, afebrile."),
            "narrative": _t("He spent most of his childhood in {region} and now takes {drug}.", "Her {symptom} started after a trip through {region}."),
            "field": _t("Recent travel: {region}\nDuration: {n} weeks", "Exposure history | {region} | {days} days ago"),
        },
        "n5_word": {
            "query": _t(
                "mom {word}, should i worry? she is {age}",
                "dad is {age} and {word}, is that normal",
                "patient {word} - anything to check? on {drug}",
                "my wife {word} and now has {symptom}",
            ),
            "shorthand": _t("{age}F, {word}.", "Pt {word}; otherwise stable on {drug}.", "{age}M on {drug}, {word}.", "{age}F, {symptom} x{days}d; {word}."),
            "narrative": _t("Her daughter says she {word} and has had {symptom}.", "He {word}, according to his wife, and remains on {drug}.", "On review today, she {word}; she is {age}.", "For the past {days} days he {word}."),
            "field": _t("Notes: patient {word}\nMeds: {drug}", "- Functional: pt {word}\n- Age: {age}", "Social history | {word} | {n} children", "Nursing note: pt {word}; c/o {symptom}"),
        },
        "n5_person": {
            "query": _t("my kid {person_first} has {symptom} since yesterday", "can {person} get a refill of {drug}"),
            "shorthand": _t("Pt {person}, {age}F, here for med refill.", "Accompanied by sister {person}."),
            "narrative": _t("{person} is a {age}-year-old who presents with {symptom}.", "{person_first} says the {drug} is upsetting their stomach."),
            "field": _t("Patient: {person}", "Emergency contact: {person} (adult child)"),
        },
    },
    "validation": {
        "p1": {
            "query": _t(
                "patient visiting from {loc}, lost her inhaler, can urgent care refill",
                "grandson lives in {loc}, can he be my health proxy",
                "back from {loc} 3 days ago, cough and chills",
            ),
            "shorthand": _t(
                "{age}F, recently moved from {loc}, establishing care.",
                "Sister in {loc} manages meds by phone.",
                "SH: works in {loc}, commutes daily; {loc_state} resident for years.",
            ),
            "narrative": _t(
                "Since moving to {loc}, he has not had a primary care physician.",
                "Her mother, who lives in {loc}, will stay with her for two weeks.",
                "He spent the winter in {loc} and first noticed the swelling there.",
            ),
            "field": _t(
                "Hometown: {loc}",
                "Current location - {loc}",
                "Next of kin: brother, {loc}",
                "Callback - {phone}\nPatient city - {loc}",
            ),
        },
        "care_site": {
            "query": _t(
                "had my scope at {site} and they found polyps, next step?",
                "wife admitted to {site} icu, what is a picc line",
                "can i get my records from {site} sent to a new doctor",
            ),
            "shorthand": _t(
                "Dx'd at {site}, here for second opinion.",
                "Pt receives HD at {site} TTS.",
                "Hx MI {year}, cath at {site}.",
            ),
            "narrative": _t(
                "Most of his care has been at {site}, where the transplant was done.",
                "After the fall, she spent four nights at {site}.",
                "Her pulmonary function study at {site} showed mild obstruction.",
            ),
            "field": _t(
                "Transfer from: {site}",
                "Site of surgery - {site}",
                "Dialysis unit | {site}",
                "Chart no. {mrn} | outside hospital {site}",
            ),
        },
        "p5": {
            "query": _t("they want to move him to {site} in {site_loc}, is the ambulance covered"),
            "shorthand": _t("Outside hosp: {site}, {site_loc}.", "OSH imaging from {site} in {site_loc} reviewed."),
            "narrative": _t("Her records were faxed from {site} in {site_loc} this morning.", "He had his hip replaced at {site} in {site_loc}."),
            "field": _t("Hospital of birth: {site}, {site_loc}", "Prior admission - {site} / {site_loc}"),
        },
        "p6": {
            "query": _t("{clin} at {site} referred me here, need appt"),
            "shorthand": _t("Curbside w/ Dr. {clin_last} @ {site} ID.", "Spoke to {clin}, {site} transplant."),
            "narrative": _t("Dr. {clin_last}, a hepatologist at {site}, called with the results.", "Her surgeon, Dr. {clin}, operates at {site}."),
            "field": _t("Accepting physician - Dr. {clin}, {site}", "Co-managing | Dr. {clin_last} | {site}"),
        },
        "mixed": {
            "query": _t("followed at {site}, does {cite} say i need yearly scans"),
            "shorthand": _t("Cards at {site}. {cite} anticoag guide reviewed."),
            "narrative": _t("She followed up at {site} and brought a {cite} article about the procedure."),
            "field": _t("Prior facility: {site}\nSource: {cite} recommendations"),
        },
        "n1_finding": {
            "query": _t("explain {finding} in plain words, im {age}", "is {finding} why i have {symptom}"),
            "shorthand": _t("A/P: {finding}. {age}M.", "{age}M; {finding}; no change in {drug}."),
            "narrative": _t("The evaluation for {symptom} showed {finding}.", "Last year's assessment documented {finding}; she remains on {drug}."),
            "field": _t("Finding - {finding} (age {age})", "Scale result | {finding} | {drug}"),
        },
        "n1_device": {
            "query": _t("is a {device} supposed to be this tight after {days} days"),
            "shorthand": _t("{device} on; skin checks q shift. {age}F."),
            "narrative": _t("The {device} will stay on for {n} more weeks until the repeat imaging."),
            "field": _t("Brace/collar - {device} (day {days})"),
        },
        "n1_study": {
            "query": _t("heard about the {cohort_study}, does it apply to me at {age}"),
            "shorthand": _t("Counseled on CV risk ({cohort_study} estimates); {drug} offered."),
            "narrative": _t("Her risk at age {age} was estimated with equations from the {cohort_study}."),
            "field": _t("Cohort reference - {cohort_study}; plan {drug}"),
        },
        "n2": {
            "query": _t("is {dz} contagious? my kid is {n}", "what are the long term effects of {dz}, still have {symptom}"),
            "shorthand": _t("Concern for {dz}; ID consulted. {age}F.", "Neg w/u for {dz} x{n}."),
            "narrative": _t("The {symptom} was not felt to be consistent with {dz}.", "He asked whether {dz} could explain the {symptom}."),
            "field": _t("Differential - {dz}; age {age}", "Ruled out | {dz} | {days}d"),
        },
        "n3": {
            "query": _t("{cite} article says turmeric helps {symptom}?", "according to {cite}, how much water a day at {age}"),
            "shorthand": _t("Low-FODMAP info from {cite} website provided; {age}F.", "Counseled on {drug} using {cite} patient guide."),
            "narrative": _t("His wife printed a {cite} fact sheet on {drug}.", "The plan follows a {cite} review of anticoagulation in adults over {age}."),
            "field": _t("Patient education - {cite} website ({symptom})", "Guideline cited | {cite} | {drug}"),
        },
        "n3_clinician": {
            "query": _t("dr. {clin_last} sent a {cite} link about {drug}, can i trust it"),
            "shorthand": _t("{clin}, NP, reviewed {cite} recs w/ family."),
            "narrative": _t("Dr. {clin_last} cited a {cite} trial when explaining the options."),
            "field": _t("Educator - {clin}\nResource - {cite} guide"),
        },
        "n4_state": {
            "query": _t("we live in {state}, are there ticks here? kid has {symptom}", "relocating to {state}, need a new endo for my {drug}"),
            "shorthand": _t("{age}M from {state}; no recent travel.", "Family in {state}, visits q{n} mo."),
            "narrative": _t("Her family still lives in {state}; she is {age}.", "He plans to travel to {state} for {n} weeks in the summer."),
            "field": _t("Birthplace - {state}; age {age}", "Residence state | {state} | {drug}"),
        },
        "n4_country": {
            "query": _t("going to {country} for {n} weeks, need shots? on {drug}"),
            "shorthand": _t("{age}F born in {country}, in US {n} yrs."),
            "narrative": _t("She returned from {country} {days} days ago with {symptom}."),
            "field": _t("Country of birth - {country}; age {age}"),
        },
        "n4_region": {
            "query": _t("moving to {region} at {age}, is the air better for asthma"),
            "shorthand": _t("Traveled through {region} {days}d ago; now {symptom}."),
            "narrative": _t("Her parents emigrated from {region} before she was born; she is {age} now."),
            "field": _t("Travel - {region} ({n} wk)"),
        },
        "n5_word": {
            "query": _t("grandpa {word}, normal at {age}?", "my husband {word}, worth mentioning? also {symptom}"),
            "shorthand": _t("Pt {word}. No acute complaints. {age}M.", "{age}F, {word}; continue {drug}."),
            "narrative": _t("His son mentioned that he {word} and has {symptom}.", "At the visit she {word}; she is {age}."),
            "field": _t("Comment - pt {word}; {drug}", "Function | {word} | age {age}"),
        },
        "n5_person": {
            "query": _t("my grandchild {person_first} has had {symptom} for {days} days"),
            "shorthand": _t("{person}, {age}M, f/u for {symptom}."),
            "narrative": _t("{person} returned for review of the {drug} dose."),
            "field": _t("Name: {person}\nVisit type: follow-up"),
        },
    },
}

# Which negative families can match each positive phenomenon (rotated in order).
MATCHED_NEGATIVES = {
    "p1": ("n4", "n1", "n2", "n5"),
    "p2": ("n3", "n1", "n4", "n5"),
    "p3": ("n3", "n2", "n5", "n4"),
    "p4": ("n3", "n5", "n1", "n2"),
    "p5": ("n4", "n3", "n1", "n2"),
    "p6": ("n3", "n2", "n5", "n4"),
    "mixed": ("n3", "n1", "n4", "n2"),
}
