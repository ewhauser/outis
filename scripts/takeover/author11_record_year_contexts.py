"""Compose a training candidate from authored clinical/year contrasts only.

No inference, data discovery, network, or external corpus reads. Outputs are
exclusive-create and frozen after review; revisions must use new version paths.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import unicodedata

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
from tokenizers import Tokenizer
from clinical_deid.data.annotation_policy import annotation_errors, canonical_identity_group, identity_split
from clinical_deid.types import ENTITY_TYPES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/tk11_train_record_year_contexts.jsonl"
REPORT = ROOT / "reports/takeover/mix3_001/author11_record_year_contexts"
HF = ROOT / "models/takeover-mbbase-mix3-001/step-003081/hf"

# Each family has separately authored clinical observations, functional details,
# reconciliation questions and actions. These are invented, not clinical advice.
# Fields: key | title | historical event | future milestone | presentation |
# examination/reasoning | function/self-management | record question | plan.
FAMILY_TEXT = r"""
asthma|Asthma follow-up|first admission for an asthma flare|review of the written breathing plan|The patient describes wheeze with cold air but no symptoms while sitting. Rescue use has increased during gardening. There is no fever, pleuritic pain or recent exposure to smoke inside the home.|Air entry is symmetric and conversation is unhurried. Technique with the demonstration inhaler is inconsistent until the spacer is fitted. A poor seal could explain some of the apparent treatment failure.|The patient wants to finish a short walk without stopping at every corner. The spacer is kept in a drawer away from the inhaler, making it easy to forget. Both can fit in the same pouch.|The older summary describes oxygen treatment without distinguishing continuous use from observation during the acute illness. Current home equipment history does not support ongoing oxygen use.|Arrange technique teaching and a symptom diary. Bring actual inhalers to the next visit before changing the maintenance regimen. Seek urgent assessment for marked breathlessness or inability to speak comfortably.
copd|Respiratory rehabilitation review|completion of the first pulmonary rehabilitation course|review of the long-term exercise plan|Breathlessness limits carrying groceries, while washing and dressing remain manageable. Morning sputum is unchanged. The patient reports no new chest discomfort, blood in sputum or abrupt decline in walking distance.|Breathing settles after rest. The patient can demonstrate pursed-lip breathing but tends to rush when standing from a chair. Today's limitations appear partly related to pacing and leg fatigue.|Shopping is divided into smaller trips, with heavy bags left for a companion. The patient would like to resume the walking group but worries about being unable to keep up with its faster members.|A discharge summary calls rehabilitation complete, whereas the therapy record describes several missed sessions. The patient remembers learning the exercises but cannot confirm the number of supervised visits.|Request the final therapy assessment and set a tolerable walking target. Review recovery time after activity and reinforce the existing rescue plan. A new change in sputum or resting symptoms should prompt a call.
heart_failure|Volume-status review|admission for fluid overload|review of the home weight-monitoring plan|Shoes fit normally this week and sleeping remains comfortable with the usual pillow. The patient is uncertain which increase in weight should trigger a call. There has been no fainting or new chest pressure.|Leg swelling is minimal and the patient does not become breathless while describing symptoms. The medication list contains two loop-diuretic entries that may represent a historical substitution rather than concurrent prescriptions.|The scale is placed on carpet and gives different readings when moved. The patient can use a notebook but prefers a simple table with the same morning routine, clothing and scale location.|The prior discharge list includes a temporary higher dose. Its stop instruction was separated from the drug list during scanning, so the copied medication history may overstate the intended maintenance dose.|Reconcile the actual tablets with the prescribing team before altering treatment. Demonstrate consistent weighing on a firm surface and confirm the individualized call parameters in writing with teach-back.
afib|Rhythm clinic reconciliation|first documented atrial fibrillation episode|review of the rhythm-management preferences|Brief palpitations occur with hurried activity, but the patient has not had syncope. The main concern is whether an old rhythm diagnosis still belongs on the active list after a later tracing appeared normal.|Pulse is regular during this assessment. A normal pulse today does not establish the absence of intermittent arrhythmia. The history of symptoms, prior monitoring and medication tolerance needs to be considered together.|The patient takes tablets from a weekly organizer and can explain which one controls the pulse. Bruising is occasional, with no reported black stools, persistent bleeding or head injury.|A copied discharge paragraph mixes the episode diagnosis with a later normal tracing. The original rhythm strip would clarify what was observed and whether the diagnosis was confirmed or provisional.|Request the original tracing and monitor report. Review pulse and symptom recording, medication adherence and bleeding precautions. Escalate sustained symptoms, fainting or new neurologic complaints through the usual urgent pathway.
hypertension|Blood-pressure follow-up|start of home blood-pressure monitoring|review of the home blood-pressure routine|The patient brings a handwritten list with variable readings and asks whether a single high value means treatment has failed. Several measurements were taken immediately after climbing stairs or drinking strong coffee.|The cuff appears small for the upper arm. Repeating a measurement after quiet sitting is more useful than comparing readings obtained under different conditions. No acute headache, weakness or chest pain is reported.|The patient can sit with feet supported at the kitchen table and is willing to measure at consistent times. A family member sometimes records the numbers but does not choose the medication dose.|An older note says home monitoring was established but does not record the cuff size or technique. Its copied instruction to check frequently may have encouraged anxious repeated measurements.|Check cuff fit and compare the home device with the office device. Agree on a limited recording schedule and review the average before making a routine medication change.
diabetes|Diabetes self-management review|completion of initial diabetes education|review of the long-term self-management goals|The patient describes irregular meals when appointments run late and occasional shakiness before supper. Symptoms improve after eating, but no meter reading was obtained during those episodes. Foot skin remains intact.|The medication history must be checked for agents that can cause low glucose. A single symptom report cannot establish the cause. The patient demonstrates meter use correctly after replacing an expired strip container.|Lunch is often delayed while helping a partner with errands. Keeping a planned snack in a bag may be more workable than an elaborate meal schedule. The patient wants to continue managing the meter without assistance.|A scanned education sheet lists a former treatment regimen. Copying its meal instructions into the current plan could be confusing because the present tablets and daily schedule differ.|Review symptomatic readings and the current medication list with the treating clinician. Reinforce the individualized low-glucose plan, regular foot checks and a manageable meal routine using teach-back.
ckd|Kidney clinic record review|first nephrology consultation|review of the kidney-protection plan|The patient asks whether an old kidney label means dialysis is imminent. Appetite and urine pattern are unchanged, and there is no new swelling. The concern arose while reading a copied problem list.|Several laboratory summaries use different units and equations, so isolated numbers should not be compared without their original reports. Today's discussion emphasizes the trend and reversible contributors rather than a single stage label.|The patient uses an over-the-counter pain tablet intermittently but cannot recall its ingredients. The bottle can be brought to the next medication review. Fluid advice has been interpreted as a fixed rule despite changing symptoms.|The earliest referral letter described a question about reduced filtration. Later summaries repeat that phrase as a confirmed stage without attaching the supporting laboratory report.|Obtain the original laboratory trend and reconcile nonprescription medicines. Provide an individualized written plan after review of current results, including whom to contact for reduced urine, new swelling or persistent vomiting.
knee_arthroplasty|Knee replacement follow-up|left knee replacement|review of the long-term joint-replacement record|The patient reports stiffness on first standing that eases with gentle movement. There is no wound drainage, fever or sudden inability to bear weight. The chief goal is climbing the porch steps more comfortably.|The scar is healed and the patient rises slowly from a chair. The available copied operative summary does not identify the implant details. Function and new symptoms should guide the need for further evaluation.|A rail is available on one side of the steps, and the patient uses it consistently. Exercises were stopped after a busy period because the printed sheet was misplaced, rather than because of a new injury.|The referral and the copied problem list disagree about whether the surgery was a partial or total replacement. The patient remembers which knee was treated but not the implant type.|Request the operative report and restore an achievable exercise routine with therapy input. New warmth, rapidly increasing swelling or loss of weight-bearing ability should prompt timely reassessment.
hip_fracture|Hip-fracture recovery review|repair of a hip fracture|review of the home mobility plan|The patient is walking indoors with an aid and remains cautious at the doorway threshold. Pain is mild at rest but fatigue increases with longer trips. There has been no new fall or sudden change in leg function.|The aid is set too low and encourages a stooped posture. Demonstration with corrected height improves the step sequence. The patient needs additional practice before carrying objects while walking.|Loose rugs have been removed, but a pet bowl still narrows the kitchen route. The patient prefers a tray attachment to asking for help with every drink. Night lighting is adequate along the main route.|The rehabilitation summary contains weight-bearing advice from an early phase of recovery. It should not be treated as a permanent restriction without checking the later orthopedic instructions.|Confirm current restrictions with the orthopedic team and arrange an equipment check. Review safer transfers, home obstacles and the plan for obtaining help after a fall without attempting an unsafe independent lift.
osteoporosis|Bone-health planning visit|first low-trauma wrist fracture|review of the bone-health treatment course|The patient asks whether a healed wrist means bone treatment is no longer needed. There is no new focal bone pain. The medication history includes a treatment break, but the reason is unclear.|The copied fracture note establishes an injury history rather than the current state of bone strength. Medication decisions require the treatment record, current risk assessment and tolerance of prior therapy.|The patient remembers difficulty swallowing a large tablet and stopped it without informing the clinic. Calcium-containing foods are acceptable, and short walks are already part of the daily routine.|The old drug list may reflect an intended prescription that was never filled. Pharmacy dispensing history and the patient's account can distinguish attempted treatment from sustained use.|Clarify prior treatment exposure and swallowing symptoms before selecting the next step. Discuss safe activity, fall prevention and a follow-up plan that the patient can realistically complete.
lumbar_pain|Back-pain history reconciliation|lumbar decompression procedure|review of the back-care goals|The patient reports an aching back after prolonged standing, relieved by changing position. There is no new bladder dysfunction, saddle numbness or progressive leg weakness. The patient wants help interpreting an old surgical summary.|Movement is guarded but the patient can change position without assistance. The present symptom pattern may not reflect the same process that led to surgery, so the old diagnosis should not substitute for a current assessment.|Cooking is interrupted by sitting breaks, while brief walks are tolerated. The patient is interested in changing work height and pacing chores before considering another invasive treatment.|The copied narrative alternates between decompression and fusion. The patient recalls a hospital stay but is unsure whether hardware was placed; the operative note is needed for accurate reconciliation.|Request the original operative report and assess current neurologic findings. Develop a paced activity plan and explain the symptoms that require urgent reassessment while avoiding assumptions based on the copied procedure name.
carpal_tunnel|Hand-function review|carpal tunnel release|review of the hand-function plan|Intermittent finger tingling returns with prolonged gripping, and the patient wonders if the prior operation failed. There is no recent hand injury. Dropping a cup has increased concern about safely preparing hot drinks.|Symptoms need to be mapped by digit and activity rather than assigned automatically to the prior diagnosis. Grip discomfort and sensory symptoms may have more than one contributor.|The patient can use larger handles and take breaks during repetitive tasks. A wrist splint is available but has been worn too tightly, leaving temporary pressure marks that resolve after removal.|The older note describes numbness before surgery; later copy-forward entries make it appear continuously present. The patient recalls a prolonged interval of relief that is absent from those summaries.|Document the current distribution and function, review splint fit and obtain the operative follow-up if available. Escalate new persistent weakness or progressive sensory loss for further assessment.
stroke|Stroke follow-up record review|stroke requiring inpatient rehabilitation|review of the communication and mobility supports|The patient reports stable word-finding difficulty and uses extra time to answer. There is no new focal weakness or sudden change in speech. The visit focuses on reconciling an old rehabilitation account with present abilities.|A copied discharge paragraph understates the patient's current independence. The patient follows the discussion and can demonstrate the usual transfer with the familiar aid, although fatigue increases after repeated attempts.|Written choices help when a word is difficult to retrieve. The patient prefers to speak first before a companion adds details and wants this preference included in future referral material.|The old rehabilitation score was measured during acute recovery. Carrying it into current notes without context could imply a persistent level of dependence that the patient no longer has.|Update the functional description after direct assessment, confirm the prevention regimen with the treating team and preserve the patient's communication preferences. Review urgent action for any sudden new neurologic symptom.
seizure|Seizure-history reconciliation|first witnessed seizure|review of the seizure-safety plan|The patient asks whether a copied seizure entry explains every brief lapse in concentration. No recent convulsion is reported. Sleep has been irregular, and the patient sometimes forgets whether a dose was already taken.|The available narrative lacks a clear description of the original event. Episodes of inattention should be characterized separately rather than assumed to be recurrent seizures solely because the diagnosis appears in the header.|A marked pill organizer may reduce uncertainty about missed or duplicate doses. The patient is willing to keep a brief event description, including what a witness observed and how recovery occurred.|An old emergency note describes a suspected event, while a later specialist letter uses more definite language. The original specialist assessment is needed before simplifying the diagnostic history.|Request the relevant specialist report, reconcile actual medication use and review an individualized safety plan. Prolonged or repeated events without recovery require emergency help; routine questions can go through the clinic.
migraine|Headache follow-up|first specialist evaluation for recurrent migraine|review of the headache-management plan|Headaches recur after poor sleep and prolonged screen use. The patient describes a familiar pattern without a sudden maximal onset or new focal deficit. The concern is whether the old scan must be repeated automatically.|Frequency, disability and use of acute medicines should be recorded before judging the current plan. The historical scan report and present symptoms answer different questions and should not be conflated.|The patient can shorten screen sessions but cannot avoid all computer use. A simple diary recording headache days, associated symptoms and medicine use seems feasible, whereas a lengthy daily questionnaire does not.|The copied consultation says imaging was reassuring at the time but does not provide the original wording. A summary alone cannot establish which imaging was performed or why.|Clarify the old investigation if it affects current decisions and review the current headache pattern. Discuss the established rescue instructions and explain when a changed or abrupt severe headache warrants urgent assessment.
parkinsonism|Movement-disorder care review|first assessment for parkinsonism|review of the daily support plan|The patient reports slowness when dressing and hesitation at narrow doorways. Tremor is less bothersome than difficulty initiating steps. There has been no new fainting or abrupt change in alertness.|The patient benefits from pausing before turning and from a clear verbal cue. The reported timing of symptoms in relation to medication needs a focused diary before attributing them to progression.|Buttons take extra time, and the patient prefers clothing changes that preserve independence. A companion can help record difficult periods but should allow the patient to complete familiar tasks at a comfortable pace.|An early referral used a provisional syndrome label. Later notes copied it as a fixed diagnosis without including the specialist's differential or subsequent examination findings.|Request the specialist summary and review medication timing, falls and daily function. Consider therapy input for cueing and dressing strategies, with reassessment if swallowing, alertness or mobility changes substantially.
memory|Memory-clinic follow-up|completion of the first memory assessment|review of the household support arrangements|The patient misplaces papers but continues familiar household routines. The main difficulty is following appointments across several letters. There is no abrupt confusion, fever or recent change in consciousness.|The screening score is only one part of the assessment. Sensory difficulty, mood, sleep and day-to-day function need review before the copied score is interpreted as a complete description of ability.|The patient wants to retain control of ordinary purchases and agrees to keep appointments on one visible calendar. A companion can check the calendar without taking over all decisions.|The earlier assessment used a different screening form and noted hearing difficulty. Comparing totals without those details may create a false impression of rapid decline.|Obtain the full prior assessment and assess current function with the patient's permission. Simplify appointment information, review reversible contributors and agree on a support plan that respects retained abilities.
hearing|Hearing-aid review|first hearing-aid fitting|review of the hearing-support plan|The patient hears one-to-one conversation better than speech around a crowded table. The hearing aid whistles when a hat presses against it. There is no ear pain, drainage or sudden hearing loss.|Device fit and maintenance should be checked before concluding that amplification has failed. The patient can identify the controls but has difficulty opening the battery compartment.|The patient prefers written backup for complex instructions and sits facing the speaker. Repeatedly increasing volume has not improved clarity in noisy settings, which has led to avoiding some social visits.|The fitting report copied into the referral describes a previous device. It does not establish the settings or maintenance needs of the device currently worn.|Arrange an audiology device check and use communication strategies during visits. A sudden hearing change should be assessed promptly; routine fitting difficulties can be addressed through the hearing service.
cataract|Vision follow-up|cataract extraction from the right eye|review of the eye-care follow-up plan|The patient notices glare when reading a bright tablet screen and asks whether the previous cataract has returned. There is no sudden curtain-like loss of vision, severe eye pain or new shower of floaters.|The copied surgical history does not describe the current lens capsule or ocular surface. Current symptoms need an eye examination instead of interpretation from the old operation label alone.|Larger type and reduced glare help with reading medication instructions. The patient uses several drops but is unsure which bottle is current and which was left over from postoperative care.|The older postoperative sheet includes temporary drops that remain on the general medication list. Its taper instructions were not included in the copied list.|Arrange a current eye assessment and reconcile the drop bottles with the eye team. Explain how to seek urgent help for sudden visual loss or severe pain, and provide readable instructions.
glaucoma|Glaucoma record reconciliation|start of pressure-lowering eye drops|review of the long-term eye-monitoring plan|The patient uses drops most evenings but sometimes misses the eye and repeats the dose. Vision feels unchanged. The patient wonders whether a copied normal pressure reading means follow-up is unnecessary.|A single old pressure value does not capture the full monitoring history. Technique, adherence, visual fields and the treating eye team's assessment all matter when interpreting the current plan.|The patient can rest the bottle hand against the forehead but squeezes hard when anxious. A demonstration device and larger print may make administration more reliable.|The prior letter lists a target pressure without describing the later monitoring results. The general problem list has retained that target while omitting changes made by the eye service.|Confirm the active drops and monitoring schedule with the eye team. Practice administration and discuss practical reminders, while encouraging prompt assessment of acute eye pain or a sudden visual change.
breast_survivorship|Breast-cancer survivorship review|completion of primary breast-cancer treatment|review of the survivorship care plan|The patient reports stable scar tightness and wants a concise account of prior treatment for a new clinician. There is no new breast lump or unexplained progressive symptom reported during this visit.|The copied summary combines surgery and additional treatment into a single phrase. A survivorship plan should distinguish completed treatment, ongoing medicines and the current surveillance recommendations.|The patient can manage routine activities but avoids an overhead exercise because of pulling at the scar. A gradual movement plan and clear contact route would be more useful than repeating the entire treatment narrative.|The referral omits whether the endocrine medicine was stopped or intended to continue. The oncology treatment summary and current prescribing record are needed to resolve that question.|Request the definitive treatment summary, reconcile ongoing therapy and document present concerns. Confirm the surveillance plan with oncology and arrange assessment of new focal or persistent symptoms.
prostate_surveillance|Prostate surveillance review|initial prostate biopsy|review of the surveillance preferences|The patient asks whether surveillance means that nothing is being done. Urinary symptoms are stable, and there is no new bone pain. Anxiety increased after reading a copied pathology phrase without its explanation.|The pathology report, subsequent tests and specialist discussion are needed to understand the chosen approach. The old biopsy wording alone should not be used to infer the current risk or next step.|The patient prefers a written explanation of which tests are routine and which changes require an earlier call. Travel to appointments is manageable if several visits can be coordinated.|The copied record alternates between watchful waiting and active surveillance. Those terms may reflect different intentions, so the actual specialist plan should be obtained rather than guessed.|Clarify the surveillance strategy with the specialist and review the patient's preferences. Provide a concise schedule after confirmation, with a clear route for questions or a change in symptoms.
colon_surveillance|Colorectal surveillance planning|removal of a colon polyp|review of the colorectal surveillance plan|The patient has received conflicting reminders about a repeat colonoscopy. Bowel pattern is unchanged, with no reported bleeding or progressive weight loss. The patient wants to avoid both unnecessary repetition and a missed examination.|The original pathology and quality of the prior examination influence the interval. A copied word such as polyp is insufficient to identify the appropriate follow-up without those details.|Preparation was difficult because written instructions arrived late and conflicted with the medicine list. The patient can complete a preparation if the plan is clear and help is arranged in advance.|The procedure summary lists removal of a lesion but omits size, histology and preparation quality. A later reminder appears to have used a default interval.|Obtain the full procedure and pathology reports before finalizing surveillance timing. Reconcile preparation instructions with the treating team and encourage assessment of new bleeding or other persistent bowel changes.
reflux|Upper gastrointestinal review|endoscopy for persistent reflux|review of the reflux-management plan|Burning discomfort follows late meals and improves when the patient remains upright. There is no progressive difficulty swallowing or vomiting blood. The patient is unsure whether a copied endoscopy label requires lifelong treatment.|The original endoscopy and biopsy reports are more informative than the problem-list abbreviation. Current symptom burden and medicine use should be reviewed separately from historical findings.|The patient often eats just before lying down because meals are shared after a partner returns home. Smaller portions and adjusting the resting position may be more feasible than a rigid meal schedule.|The old discharge sheet includes a short treatment course, while the copied medication list gives no stop date. It is unclear whether later clinicians intentionally continued the medicine.|Request the original report if it changes management and reconcile current treatment with the prescriber. Review practical symptom measures and explain which new swallowing or bleeding symptoms require prompt assessment.
ibd|Inflammatory bowel disease review|first admission for inflammatory bowel disease|review of the bowel-disease care plan|The patient describes a stable bowel pattern but worries that an old admission note means a flare is continuing. There is no new rectal bleeding, fever or marked abdominal pain reported today.|The historical admission documents severe symptoms during an earlier episode. It should not substitute for an assessment of current disease activity, nutrition, medicines and the specialist's follow-up plan.|The patient keeps spare supplies when away from home and wants to return to longer outings. Uncertainty about toilets, rather than current urgency alone, limits confidence.|A copied inpatient list includes medicines used briefly during the acute episode. The current specialist list is shorter, and the patient is concerned about accidentally restarting an old medicine.|Confirm the active regimen with the bowel-disease team, review current symptoms and clarify which copied medicines are historical. Provide a contact plan for worsening pain, bleeding or dehydration.
ostomy|Stoma care follow-up|creation of the stoma|review of the stoma-care supplies plan|The patient reports intermittent leakage when bending and mild skin soreness under the edge of the appliance. Output is otherwise usual, with no vomiting or sudden cessation. The main concern is confidence when leaving home.|The appliance opening may no longer match the current stoma shape. A fitting assessment is preferable to simply adding more adhesive, which has irritated the skin during removal.|The patient can change the appliance but needs extra time to see the lower edge. A mirror improves visibility, and supplies are stored within reach without requiring deep bending.|The surgical discharge instructions describe an early postoperative fitting. The copied measurements may no longer represent the present appliance needs after recovery.|Arrange review with a stoma nurse, assess the skin and update the supply list after fitting. Discuss when obstruction symptoms, severe pain or substantial output changes require urgent contact.
sleep_apnea|Sleep-device follow-up|first positive-airway-pressure titration|review of the sleep-device plan|The patient removes the mask after waking with a dry mouth and feels less rested after those nights. There is no new chest pain or acute breathlessness. The machine is used inconsistently because of discomfort.|Mask leak, humidification and the recorded use pattern should be reviewed before assuming that treatment is ineffective. The old titration pressure does not establish current comfort or fit.|The patient can replace the water chamber but is uncertain about cleaning the mask seal. A simple sequence demonstrated with the actual equipment is preferred to a long written manual.|The copied sleep report lists a prior mask model. Equipment changes made by the supplier are missing from the clinic summary, making the instructions difficult to follow.|Arrange an equipment and adherence review with the sleep service. Discuss mask comfort and safe alertness practices, and confirm the active settings through the treating team before any adjustment.
depression|Mood follow-up|first course of counseling for depression|review of the relapse-prevention plan|The patient reports reduced enjoyment and difficulty initiating chores but continues eating and attending visits. There are no current thoughts of self-harm reported. The patient wants the record to reflect improvement as well as previous difficulty.|The copied symptom score was obtained during a more difficult period. Current mood, sleep, function and safety need direct review instead of assuming the older severity rating remains unchanged.|Brief contact with a trusted friend helps the patient begin a walk. Large lists of activities feel overwhelming, so one achievable daily task is preferred as a starting point.|The earlier counseling summary describes goals that were partly met. The copied problem list contains the initial symptom description but none of the progress or strategies the patient found useful.|Review safety and the current treatment plan, agree on one manageable activity and arrange appropriate follow-up. Provide a clear route for urgent help if self-harm thoughts emerge or safety deteriorates.
anxiety|Anxiety-management review|initial therapy assessment for panic symptoms|review of the coping and support plan|Episodes of intense worry occur before crowded appointments and settle after the patient leaves the waiting area. The patient remains concerned about the meaning of an old panic label and avoids some routine care.|The account should distinguish anticipated anxiety from unexplained acute physical symptoms. The old diagnosis does not remove the need to assess a new or substantially changed symptom pattern.|Waiting near a quieter area and knowing what will happen during the visit improves attendance. The patient uses a brief grounding exercise but finds repeated reassurance calls less helpful over time.|The copied therapy note includes an early exposure exercise without its graded steps. The patient interpreted that fragment as a demand to tolerate a crowded setting immediately.|Obtain the fuller therapy plan if available and agree on manageable preparation for visits. Review current symptoms and supports, with appropriate urgent assessment for new concerning physical symptoms or a safety crisis.
alcohol_recovery|Recovery support review|completion of residential alcohol treatment|review of the recovery-support plan|The patient reports continued engagement with recovery supports and worries that copied admission language obscures later progress. Sleep is variable, but there is no current intoxication or withdrawal symptom reported.|Current substance use, cravings, mood and safety should be discussed directly and without assuming that the acute admission description remains current. The patient identifies predictable situations that increase cravings.|A regular meal and evening telephone check-in help maintain routine. The patient prefers small meetings and wants practical help planning for social events where alcohol may be offered.|The copied discharge narrative contains a temporary sedating medicine used during acute care. It should be reconciled before it is mistakenly treated as ongoing treatment.|Confirm the active support and medication plan, discuss early warning signs and arrange a reachable contact route. Acute withdrawal symptoms or immediate safety concerns require prompt medical assessment.
smoking|Tobacco cessation follow-up|first structured tobacco-cessation attempt|review of the tobacco-cessation goals|The patient has reduced cigarette use and identifies the first break after meals as the hardest time. A previous quit attempt is described as unsuccessful in the copied note, which the patient finds discouraging.|The history contains useful information about triggers and strategies even though abstinence was not sustained. Current use and preferences should guide discussion without treating the earlier outcome as a prediction.|Replacing one smoking break with a short walk has been manageable. The patient wants to choose the next change and is willing to discuss medicine options after prior side effects are clarified.|The earlier record lists a cessation medicine but does not state whether it was actually taken or why it was stopped. The patient recalls nausea but not the product name.|Clarify prior medicine exposure, recognize the achieved reduction and agree on a specific next step. Offer follow-up support and discuss available treatments in the context of the patient's preferences and tolerance.
gout|Gout-history reconciliation|first joint aspiration confirming gout|review of the gout-prevention plan|The patient has no acutely swollen joint today but asks whether a copied flare prescription should be taken every day. The medicine box contains both a prevention tablet and a short-course pain medicine.|The original aspirate report would clarify the basis of the diagnosis. Current prevention and flare instructions should be separated so that a copied acute prescription does not become a routine daily medicine.|The patient can read the labels but finds the similar containers confusing. Keeping an updated medicine card beside the organizer may help distinguish scheduled treatment from instructions used only during a flare.|The old emergency list was copied without its limited duration. The patient has occasionally continued the short-course medicine after symptoms settled because the summary did not include a stop instruction.|Reconcile the actual tablets with the prescriber and provide distinct routine and flare instructions. A new hot swollen joint, especially with fever or inability to bear weight, needs timely assessment.
rheumatoid|Inflammatory arthritis review|start of disease-modifying treatment for rheumatoid arthritis|review of the arthritis-monitoring plan|The patient reports morning hand stiffness that improves with movement and difficulty opening jars. There is no fever or rapidly swollen single joint. The copied record lists several past medicines beside the current one.|Medication exposure, current inflammation and function need separate review. A list of previously tried agents can be mistaken for concurrent treatment if discontinued medicines are not clearly identified.|Larger grips and a jar opener reduce discomfort, and the patient wants to continue preparing simple meals. The weekly medicine schedule is easier to follow when paired with a familiar household routine.|The earlier specialist letter describes a planned treatment change but does not prove the new medicine was started. Subsequent prescribing records and the patient's account are needed.|Confirm the active regimen and required monitoring with rheumatology. Review function, adherence and infection precautions appropriate to the confirmed treatment, and update the shared medication list after reconciliation.
psoriasis|Skin clinic follow-up|first specialist treatment for psoriasis|review of the skin-treatment plan|The patient describes plaques that improve with treatment and recur when application becomes difficult. Itching interrupts rest, but there is no widespread tenderness or fever. The main difficulty is reaching the back.|The copied prescription lists several strengths without specifying the intended body sites. Current skin findings and actual use should be reviewed before concluding that the medicine has failed.|The patient uses a long-handled applicator and can manage most areas without help. Greasy residue makes morning application inconvenient, so a workable schedule is more likely to be followed.|The old treatment letter includes a brief stronger course for a limited area. That instruction lost its location and duration when copied into the general list.|Clarify the current site-specific plan with the skin service and demonstrate the intended amount. Review symptoms of infection or a substantial change in the rash and arrange follow-up appropriate to current severity.
eczema|Dermatitis care review|first dermatology assessment for chronic hand eczema|review of the skin-protection routine|The patient reports cracked hands after frequent washing and discomfort when using household cleaners. There is no rapidly spreading redness or fever. Different creams have been tried without a consistent routine.|The pattern of exposure and actual product use may explain persistent irritation. The copied allergy list contains a nonspecific cream reaction that needs clarification before useful products are unnecessarily excluded.|Cotton-lined gloves are tolerable for short cleaning tasks, but the patient removes them when they become damp. A plain moisturizer near the sink is easier to remember than one stored in the bedroom.|The earlier note refers to burning on broken skin rather than a confirmed allergy. Later copy-forward entries turned this into a broad intolerance to all topical treatment.|Clarify the reported reaction, review skin protection and establish a manageable treatment routine with the clinician. Increasing pain, discharge or spreading redness should prompt reassessment.
venous_ulcer|Leg-wound follow-up|first treatment course for a venous leg ulcer|review of the wound-prevention plan|The patient reports a closed wound with fragile surrounding skin and worries about recurrence after prolonged standing. There is no new drainage or fever. The compression garment is difficult to apply without assistance.|The copied wound dimensions describe an earlier open lesion rather than the current appearance. Current skin, swelling and circulation assessment should guide the ongoing care plan.|The patient can elevate the leg while reading and uses a donning aid when it is within reach. Avoiding all walking has increased stiffness, so advice needs to distinguish activity from prolonged immobile standing.|The old dressing order remained active after closure. Continuing to copy it could create confusion about whether a wound is still open and which supplies are actually required.|Update the wound status after examination and confirm the current compression plan with the treating team. Review skin care, feasible movement and contact instructions for recurrent breakdown, pain or spreading redness.
urinary_incontinence|Bladder symptom review|first pelvic-floor therapy assessment|review of the bladder-management plan|The patient reports leakage with coughing and occasional urgency when reaching the door. There is no pain on urination, visible blood or fever. The patient is embarrassed by a copied description that overstates pad use.|The symptom pattern and current impact should be documented in the patient's own terms. An old bladder diary is useful history but does not establish current frequency or fluid intake.|The patient can practice exercises during a regular seated activity but is unsure of the technique. Restricting all fluids before leaving home has caused thirst and has not reliably prevented leakage.|The earlier therapy summary says instruction was provided but does not establish that the exercise technique was checked. Later notes assumed the full program had been completed.|Arrange technique review if desired, obtain a brief current diary and discuss practical toileting strategies. New pain, visible blood, retention or systemic symptoms need assessment rather than attribution to the chronic pattern.
bph|Lower urinary tract symptom review|first urology assessment for prostate enlargement|review of the urinary-symptom care plan|The patient describes a slow stream and getting up during the night but no inability to pass urine. The symptoms are bothersome mainly because of interrupted sleep. There is no fever or visible blood.|The copied symptom score was obtained before a medicine trial and should not be treated as current. Present symptom burden, side effects and the patient's priorities require direct review.|The patient takes an evening medicine but sometimes omits it after feeling lightheaded on standing. The route to the bathroom is clear, and a night light is already used.|The older letter mentions considering a procedure without documenting a decision. Subsequent copies make the procedure appear scheduled even though the patient recalls choosing continued observation.|Confirm the current medicine and prior decision with urology, review standing symptoms and document present preferences. Inability to pass urine, severe pain or systemic illness warrants urgent assessment.
dysphagia|Swallowing follow-up|first instrumental swallowing assessment|review of the swallowing-support plan|The patient reports occasional coughing with hurried drinks but tolerates the usual meals when taking time. There has been no new inability to swallow or marked weight loss. The patient asks about a copied texture restriction.|The historical assessment described function during an acute illness. Its recommendations should be checked against subsequent reassessment rather than continued indefinitely through copying.|The patient can sit upright and take smaller sips but is often rushed during shared meals. Clear instructions for the usual carers would help maintain a consistent pace without unnecessary restriction.|The scanned report lacks the final page that contained the planned reassessment. A later therapy note may explain whether the initial texture advice was changed.|Obtain the complete assessment and arrange current swallowing review when indicated. Confirm practical pacing and positioning advice, with prompt assessment for persistent choking, inability to swallow or respiratory deterioration.
dental|Oral health care review|first fitting of the upper denture|review of the oral-care plan|The patient reports rubbing beneath a denture after meals and has begun avoiding firm foods. There is no facial swelling or fever. The denture is stable during conversation but becomes uncomfortable with prolonged wear.|The copied fitting note describes the initial appliance, while the patient recalls a later adjustment. Current fit and mucosal examination are needed before deciding whether another adhesive will help.|The patient can remove and clean the denture but has difficulty seeing a rough area on its underside. Softer foods remain varied, and the patient would like to return to ordinary family meals.|The original instructions mention an early adjustment visit that is absent from the copied record. A documented fitting date alone does not establish whether subsequent changes were made.|Arrange dental assessment of the fit and sore area, review cleaning and discuss maintaining nutrition while awaiting review. Spreading swelling, fever or difficulty swallowing requires prompt assessment.
"""

FAMILIES = [dict(zip(("key", "title", "event", "milestone", "present", "exam", "function", "record", "plan"), line.split("|")))
            for line in FAMILY_TEXT.strip().splitlines()]
assert len(FAMILIES) == 40 and all(len(f) == 9 for f in FAMILIES)

LAYOUTS = ("narrative", "sectioned_followup", "patient_message", "referral_reply", "home_review",
           "problem_oriented", "handoff", "record_reconciliation")
HEADINGS = (
    ("Reason for contact", "Current account", "Background reviewed", "Observed function", "Medicines and routine", "Record clarification", "Interpretation", "Reference material", "Agreed plan", "For the next clinician"),
    ("Subjective", "History of present concern", "Relevant history", "Function", "Medication review", "Outside records", "Assessment", "Evidence reviewed", "Plan", "Continuity note"),
    ("Message received", "Patient's concern", "Chart context", "Daily activities", "Medicine questions", "Response after record review", "Clinical interpretation", "Reading discussed", "Next steps", "Reply summary"),
    ("Reason for referral", "Symptoms reported", "Prior care", "Functional account", "Treatment reconciliation", "Documents available", "Impression", "Supporting material", "Requested action", "Transfer summary"),
    ("Visit purpose", "Home account", "Care history", "Home routine", "Support and medicines", "Folder review", "Assessment", "Education material", "Shared plan", "Home-care handoff"),
    ("Main problem", "Current symptoms", "Historical component", "Effect on daily life", "Treatment use", "Record discrepancies", "Working assessment", "Background reading", "Actions", "Continuing-care summary"),
    ("Handoff purpose", "Current status", "Established history", "Usual function", "Reconciliation tasks", "Source note review", "Clinical priorities", "Shared reading", "Follow-up actions", "Receiving-team note"),
    ("Reconciliation request", "Patient account", "Source chronology", "Current abilities", "Actual treatment", "Copy-forward review", "Meaning for care", "Reference notes", "Updated plan", "Carry-forward summary"),
)

FIRST = ("Maren", "Tavi", "Ellora", "Nolan", "Sefina", "Ilan", "Corin", "Vela", "Rohan", "Naima", "Lucien", "Amina", "Soraya", "Emil", "Zara", "Lio", "Anika", "Tomas", "Mira", "Devon")
LAST = ("Fennelridge", "Orendale", "Vossmere", "Caldwick", "Ternbrook", "Solmere", "Iversonne", "Brackenfell", "Larkstone", "Dunelowe", "Mirevale", "Pellwick", "Asterley", "Wrenhurst", "Bellquay", "Alderfen", "Rookvale", "Sableford", "Lindenmere", "Quillbrook")
MID = ("Ari", "Bevan", "Celia", "Dara", "Enid", "Faris", "Galen", "Hana", "Ivo", "Jori")

def personas():
    result = []
    for a in FIRST:
        for b in LAST:
            for c in MID:
                name = f"{a} {c} {b}"
                group = canonical_identity_group(name)
                if identity_split(group) == "train":
                    result.append((name, group))
    random.Random(1100426).shuffle(result)
    return result[:250]

def mark(kind, value):
    return f"[[{kind}|{value}]]"

def parse_marked(marked):
    pieces, spans, ledger, previous, offset = [], [], [], 0, 0
    for match in re.finditer(r"\[\[([A-Z_]+)\|([^\[\]]+)\]\]", marked):
        literal, value, tag = marked[previous:match.start()], match[2], match[1]
        pieces.extend((literal, value))
        offset += len(literal)
        item = {"start": offset, "end": offset + len(value), "text": value}
        ledger.append(dict(item, semantic_role=tag))
        if tag.startswith("DATE_"):
            spans.append(dict(item, label="DATE"))
        offset += len(value)
        previous = match.end()
    pieces.append(marked[previous:])
    text = "".join(pieces)
    assert "[[" not in text
    return text, spans, ledger

def public_paragraph(year, variant, verbose):
    ref = year if year <= 2023 else 2014 + variant * 3
    options = (
        f"A public clinical review published in {mark('PUBLIC_YEAR', ref)} was discussed alongside the patient's questions about its publication year. The printed handout carries revision {mark('VERSION', str(year) + '.2')}.",
        f"Background reading included a {mark('PUBLIC_YEAR', ref)} journal review; its publication year was checked before discussing the patient's question. The education packet uses document-version {mark('VERSION', str(year) + '.4')} in its footer.",
        f"The patient brought a general educational chapter published in {mark('PUBLIC_YEAR', ref)} and asked how the advice might have changed since that year. The clinic's copy is labeled content release {mark('VERSION', str(year) + '.1')}.",
        f"The reference list includes a public practice overview from {mark('PUBLIC_YEAR', ref)}; the clinician reviewed its publication year and scope. A publisher's revision code, {mark('VERSION', str(year) + '.3')}, appears below the title.",
    )
    result = options[variant % 4]
    if verbose:
        result += " The discussion focused on the patient's question and how the material related to the present concern. General reading cannot resolve a discrepancy in an individual's treatment history; the original source record is still needed when that discrepancy changes care."
    return result

def temporal_parts(f, year, variant, control, index):
    mode = ("historical_event", "copied_document", "future_plan")[variant]
    d = lambda value, kind: str(value) if control else mark("DATE_" + kind, value)
    if variant == 0:
        first = f"The patient dates the {f['event']} to {d(year, 'HISTORICAL')} and recalls why care was sought."
        second = f"In the patient's earlier history, {d(year, 'HISTORICAL')} is the year of the {f['event']}; the present symptoms are being assessed separately."
        third = f"The longitudinal summary will retain {d(year, 'HISTORICAL')} for the {f['event']} after the source account is reconciled."
        fourth = f"At handoff, the {f['event']} in {d(year, 'HISTORICAL')} remains relevant background, with current function described above."
        extra = f"A separate follow-up assessment occurred in {d(min(year + 3, 2025), 'DISTINCT_EVENT')}; it concerned progress after that earlier care."
        if control:
            first = f"The patient dates the {f['event']} to earlier care and recalls why care was sought."
            second = f"The patient's earlier history includes the {f['event']}; the present symptoms are being assessed separately."
            third = f"The longitudinal summary will retain the {f['event']} after the source account is reconciled."
            fourth = f"At handoff, the {f['event']} remains relevant background, with current function described above."
            extra = "A separate follow-up assessment occurred after that earlier care; it concerned subsequent progress."
    elif variant == 1:
        source_date = dt.date(min(year + 1, 2025), 3 + index % 6, 4 + index % 20)
        date_text = source_date.strftime("%A, %B ") + str(source_date.day) + source_date.strftime(", %Y")
        first = f"A copied letter dated {d(date_text, 'DOCUMENT_DATE')} records the {f['event']} in {d(year, 'HISTORICAL')}."
        second = f"The source-history paragraph places the {f['event']} in {d(year, 'HISTORICAL')}, consistent with the patient's account."
        third = f"The reconciliation note keeps {d(year, 'HISTORICAL')} attached to the {f['event']}, rather than to the present review."
        fourth = f"For continuity, the care summary describes the {f['event']} during {d(year, 'HISTORICAL')}; the copied letter is retained with its own date."
        extra = "The electronic scan completed at 14:10; that time reflects the scanning workflow."
        if control:
            first = f"A copied letter dated during earlier care records the {f['event']}."
            second = f"The source-history paragraph places the {f['event']} in earlier care, consistent with the patient's account."
            third = f"The reconciliation note keeps the {f['event']} attached to the earlier episode, rather than to the present review."
            fourth = f"For continuity, the care summary describes the previous {f['event']}; the copied letter is retained in the folder."
    else:
        first = f"The patient and clinician propose {d(year, 'FUTURE')} for a future {f['milestone']}, with earlier symptom-driven care continuing as needed."
        second = f"The planning discussion identifies {d(year, 'FUTURE')} as the year for the {f['milestone']}; it does not delay the actions for today's concern."
        third = f"The care-preference summary carries a proposed {f['milestone']} in {d(year, 'FUTURE')}, subject to changes in need and preference."
        fourth = f"At handoff, the proposed {f['milestone']} remains penciled in for {d(year, 'FUTURE')}; the receiving team can revisit that choice with the patient."
        extra = f"The longer view includes reconsidering those preferences during {d(str(year) + '–' + str(year + 1), 'PLAN_RANGE')} if the support arrangements change."
        if control:
            first = f"The patient and clinician propose a future {f['milestone']}, with earlier symptom-driven care continuing as needed."
            second = f"The planning discussion includes the {f['milestone']}; it does not delay the actions for today's concern."
            third = f"The care-preference summary carries a proposed {f['milestone']}, subject to changes in need and preference."
            fourth = f"At handoff, the proposed {f['milestone']} remains penciled in for a future visit; the receiving team can revisit that choice with the patient."
            extra = "The longer view includes reconsidering those preferences if the support arrangements change."
    return mode, first, second, third, fourth, extra

def render_sections(title, bodies, layout, headings, case):
    heading = (lambda s: s.upper()) if case == "uppercase_headings" else (lambda s: s.lower()) if case == "lowercase_headings" else (lambda s: s)
    if layout == "narrative":
        return heading(title) + "\n\n" + "\n\n".join(bodies)
    if layout == "patient_message":
        lead = "Patient message and clinical response: " + heading(title)
        return lead + "\n\n" + "\n\n".join(f"{heading(h)}\n{b}" for h, b in zip(headings, bodies))
    if layout == "referral_reply":
        return "To the receiving clinical team\n" + heading(title) + "\n\n" + "\n\n".join(f"{heading(h)} — {b}" for h, b in zip(headings, bodies)) + "\n\nPlease return the requested source material with the next update."
    if layout == "home_review":
        return heading(title) + "\nHome-care review\n\n" + "\n\n".join(f"{heading(h)}: {b}" for h, b in zip(headings, bodies))
    if layout == "problem_oriented":
        return heading(title) + "\n" + "\n\n".join(f"{i + 1}. {heading(h)}\n{b}" for i, (h, b) in enumerate(zip(headings, bodies)))
    if layout == "handoff":
        return heading(title) + "\nContinuity handoff\n\n" + "\n\n".join(f"[{heading(h)}]\n{b}" for h, b in zip(headings, bodies))
    if layout == "record_reconciliation":
        return heading(title) + "\nRecord reconciliation worksheet\n\n" + "\n\n".join(f"{heading(h)}\n  {b}" for h, b in zip(headings, bodies))
    return heading(title) + "\n\n" + "\n\n".join(f"{heading(h)}: {b}" for h, b in zip(headings, bodies))

def compose(f, index, variant, control, long, layout_no, pub_first):
    rng = random.Random(81700 + index * 101 + variant * 19)
    year = (1988 + (index * 7) % 36) if variant < 2 else 2027 + index % 4
    age = 65 + index % 24
    mode, first, second, third, fourth, extra = temporal_parts(f, year, variant, control, index)
    reading = public_paragraph(year, index % 4, long)
    first_body = f"The patient is {age} years old and requests review of the current concern together with the earlier record. " + (reading if pub_first else first)
    history = first if pub_first else reading
    duration = ("Questions have become more pressing over the last 3 weeks.",
                "The patient has kept notes on the concern for 8 days.",
                "A short daily record covering 10 days was brought for review.",
                "Symptoms and practical questions were tracked over 2 weeks.")[index % 4]
    current = f["present"] + f" The concern is rated {2 + index % 6}/10 for its effect on ordinary activities. " + duration
    routine = f["function"]
    observation = f["exam"]
    record = f["record"]
    plan = f["plan"]
    if long:
        shared = rng.choice((
            "The patient brought the containers currently in use and separated those kept only for reference. Several entries in the printed list could not be confirmed from the containers. The unresolved items are being checked with the prescribing team, and the patient was asked to avoid using an old list as a new instruction.",
            "The patient describes actual use before the printed medication list is reviewed. This sequence reveals a difference between an old instruction and the present routine. The clinician records the uncertainty and asks for the original prescribing advice before changing the shared list. Nonprescription products will be included at the next review.",
            "The medicine review includes missed doses, practical barriers and products obtained without a prescription. The patient prefers one current list kept beside the organizer. Items carried forward from the earlier episode are being verified, because their presence in a summary does not establish that they are still taken.",
            "The patient can explain the purpose of most current treatments but has questions about two older instructions. These are recorded for clarification with the relevant prescriber. The plan will be written in one place after reconciliation, with an opportunity for the patient to explain it back in ordinary language.",
        ))
        interpretation = "The present assessment separates the original episode from the concern described today. Stability in one part of the history does not answer all of the patient's current questions. The source information requested below will be used where it changes a decision; routine care and the symptom-based contact plan can continue while it is obtained."
        handoff = "The patient reviewed the proposed wording and identified the main question to carry forward. The receiving clinician should confirm unresolved details before acting on a copied instruction. The patient knows the usual clinic contact route and agrees to bring the updated list to the next relevant visit."
        bodies = [first_body, current, history + " " + observation,
                  routine + " The patient prefers a small achievable change over several simultaneous tasks. The plan takes account of the usual household routine and the help that is actually available.",
                  shared, second + " " + record,
                  interpretation + " " + extra,
                  "During teaching, the patient restated the main distinction between the earlier episode and the current next step. Questions about the reading were addressed in plain language. " + third,
                  plan + " The timing of immediate follow-up depends on symptoms and the treating team's assessment, with earlier contact if the described warning signs develop.",
                  fourth + " " + handoff]
    else:
        # Short and medium notes retain meaningful local ownership of every year.
        if index % 3 == 0:
            bodies = [first_body, current, history + " " + record, plan]
        elif index % 3 == 1:
            bodies = [first_body, current, history + " " + observation, routine, second + " " + record, plan + " " + extra]
        else:
            bodies = [first_body, current, history + " " + record, observation, plan + " " + extra]
    case = ("title_case_headings", "uppercase_headings", "lowercase_headings")[index % 3]
    layout = LAYOUTS[layout_no]
    headings = HEADINGS[layout_no]
    # Short notes have fewer sections but use section roles matching their bodies.
    if not long:
        selection = {4: (0, 1, 5, 8), 5: (0, 1, 5, 6, 8), 6: (0, 1, 2, 3, 5, 8)}[len(bodies)]
        headings = tuple(headings[x] for x in selection)
    marked = render_sections(f["title"], bodies, layout, headings, case)
    text, spans, ledger = parse_marked(marked)
    return text, spans, ledger, dict(primary_year=year, temporal_class=mode, layout=layout, heading_case=case,
                                     public_first=pub_first, requested_long=long, ordinary_age=age)

def load_window_function():
    tree = ast.parse((ROOT / "src/clinical_deid/detection/hf_encoder.py").read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "window_jobs"]
    scope = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "hf_encoder_window_ast", "exec"), scope)
    return scope["window_jobs"]

def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def sha(data):
    return hashlib.sha256(data).hexdigest()

def balanced_schedule(seed):
    """Match controls to their positive and balance each stratum exactly."""
    paired = [(i, i % 3) for i in range(250)]
    others = [(i,v) for i in range(250) for v in range(3) if v != i % 3]
    rng = random.Random(seed)
    rng.shuffle(paired)
    rng.shuffle(others)
    return set(paired[:125] + others[:250])

def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    assert not OUT.exists() and not OUT.with_suffix(".jsonl.manifest.json").exists(), "Preserve existing versions; do not overwrite."
    tokenizer = Tokenizer.from_file(str(HF / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    probe = tokenizer.encode("clinical note", add_special_tokens=False).ids
    wrapped = tokenizer.encode("clinical note", add_special_tokens=True).ids
    assert len(wrapped) - len(probe) == 2
    outer = json.loads((ROOT / "models/takeover-mbbase-mix3-001/config.json").read_text())
    assert outer["max_length"] == 512
    window_jobs = load_window_function()
    conv = module_at("author11_conventions", ROOT / "scripts/takeover/normalize_conventions.py")
    intervals = module_at("author11_intervals", ROOT / "scripts/takeover/normalize_date_intervals.py")
    rows, coverage, role_ledger = [], [], []
    identity_names = []
    long_schedule, public_schedule = balanced_schedule(110029), balanced_schedule(111317)
    for index, (name, group) in enumerate(personas()):
        f = FAMILIES[index % 40]
        paired_variant = index % 3
        for slot in range(4):
            control = slot == 3
            variant = paired_variant if control else slot
            # Every matched control has its positive's length class and order.
            long = (index, variant) in long_schedule
            public_first = (index, variant) in public_schedule
            # Same layout within each four-note group; every family cycles layouts.
            layout_no = (index // 40 + index) % len(LAYOUTS)
            text, spans, ledger, info = compose(f, index, variant, control, long, layout_no, public_first)
            pair = f"tk11-pair-{index:04d}" if control or slot == paired_variant else f"tk11-single-{index:04d}-{slot}"
            row = dict(id=f"tk11-record-year-{index:04d}-{slot}", text=text, spans=spans, split="train",
                       annotation_policy="clinical-phi-v2", identity_group=group, identity_groups=[group],
                       pair_id=pair, template_family="tk11_" + f["key"],
                       phenomena=["patient_calendar_year", "public_reference_contrast", "report_version_numerals",
                                  "ordinary_age", "duration", "score_like_number", info["temporal_class"],
                                  "matched_phi_free_control" if control else "personal_year_context"] +
                                  (["repeated_personal_year", "multi_window_context"] if long and not control else []),
                       cohort="long_document" if long else "realistic", negative_only=control,
                       authoring_layout=info["layout"])
            assert not annotation_errors(text, spans)
            assert identity_split(group) == "train" and group == canonical_identity_group(name)
            assert bool(spans) != control
            assert all(text[s["start"]:s["end"]] == s["text"] for s in spans)
            # Every four-digit numeric occurrence has an authored semantic role.
            for m in re.finditer(r"(?<!\d)\d{4}(?!\d)", text):
                roles = [x for x in ledger if x["start"] <= m.start() and x["end"] >= m.end()]
                assert len(roles) == 1, (row["id"], m.group(), text[max(0,m.start()-40):m.end()+40])
            expected = {(x["start"], x["end"], x["text"]) for x in ledger if x["semantic_role"].startswith("DATE_")}
            assert expected == {(s["start"],s["end"],s["text"]) for s in spans}
            counts = Counter()
            normalized = conv.normalize(row, counts)
            interval_normalized, edits, exclusions = intervals.normalize_row(row)
            assert normalized == row and not counts, (row["id"], "convention normalization changed row")
            assert interval_normalized == row and not edits, (row["id"], "interval normalization changed row")
            encoding = tokenizer.encode(text, add_special_tokens=False)
            offsets = encoding.offsets
            jobs = list(window_jobs(len(encoding.ids), 510))
            ownership = [0] * len(offsets)
            for a,b,left,right in jobs:
                assert b-a <= 510
                for t in range(left, right):
                    ownership[t] += 1
            assert all(n == 1 for n in ownership)
            span_windows = []
            crossings = []
            for s in spans:
                token_positions = [i for i,(a,b) in enumerate(offsets) if a < s["end"] and b > s["start"]]
                assert token_positions
                owners = sorted({j for i in token_positions for j,(_,_,left,right) in enumerate(jobs) if left <= i < right})
                # The entire containing sentence is the local ownership context.
                start = max(text.rfind(". ",0,s["start"])+2,text.rfind("\n",0,s["start"])+1,0)
                end_candidates = [x for x in (text.find(". ",s["end"]),text.find("\n",s["end"])) if x>=0]
                end = min(end_candidates) if end_candidates else len(text)
                context_tokens = [i for i,(a,b) in enumerate(offsets) if a < end and b > start]
                local_all = all(jobs[o][0] <= min(context_tokens) and max(context_tokens) < jobs[o][1] for o in owners)
                if not local_all:
                    crossings.append(dict(span=s, owners=owners, sentence=text[start:end]))
                span_windows.append(dict(text=s["text"],start=s["start"],end=s["end"],owners=owners,
                                         token_start=min(token_positions), token_end=max(token_positions)+1,
                                         local_context_in_every_owner=local_all))
            repeated = [s for s in span_windows if s["text"] == str(info["primary_year"])]
            distinct_repeated_owners = sorted({o for s in repeated for o in s["owners"]})
            if long:
                assert len(jobs) >= 2, (row["id"],len(encoding.ids))
                if not control:
                    assert len(repeated) >= 3
                    assert len(distinct_repeated_owners) >= 2, (row["id"],distinct_repeated_owners)
            rows.append(row)
            role_ledger.append(dict(id=row["id"],roles=ledger))
            coverage.append(dict(id=row["id"],negative_only=control,family=f["key"],**info,
                                 content_tokens=len(encoding.ids), windows=jobs, span_windows=span_windows,
                                 repeated_primary_year_occurrences=len(repeated),
                                 repeated_year_owner_windows=distinct_repeated_owners,
                                 context_crossings=crossings))
        identity_names.append(dict(canonical_identity=name,identity_group=group))
    assert len(rows) == 1000 and sum(r["negative_only"] for r in rows) == 250
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["text"] for r in rows}) == len(rows)
    label_support = Counter(s["label"] for r in rows for s in r["spans"])
    stats = dict(rows=len(rows),positive_rows=750,phi_free_controls=250,families=len(FAMILIES),
                 layouts=dict(Counter(c["layout"] for c in coverage)),
                 temporal_classes=dict(Counter(c["temporal_class"] for c in coverage)),
                 multi_window_rows=sum(len(c["windows"])>1 for c in coverage),
                 requested_long_rows=sum(c["requested_long"] for c in coverage),
                 phi_bearing_long_rows=sum(c["requested_long"] and not c["negative_only"] for c in coverage),
                 long_positive_repeated_at_least_three=sum(c["requested_long"] and not c["negative_only"] and c["repeated_primary_year_occurrences"]>=3 for c in coverage),
                 long_positive_repeated_across_three_owners=sum(c["requested_long"] and not c["negative_only"] and len(c["repeated_year_owner_windows"])>=3 for c in coverage),
                 context_crossings=sum(len(c["context_crossings"]) for c in coverage),
                 token_length_min=min(c["content_tokens"] for c in coverage),
                 token_length_max=max(c["content_tokens"] for c in coverage),
                 token_length_median=sorted(c["content_tokens"] for c in coverage)[500],
                 semantic_roles=dict(Counter(x["semantic_role"] for e in role_ledger for x in e["roles"])),
                 label_support={label:label_support[label] for label in ENTITY_TYPES},
                 heading_cases=dict(Counter(c["heading_case"] for c in coverage)),
                 ordering=dict(Counter("public_first" if c["public_first"] else "personal_first" for c in coverage)),
                 whole_row_normalization_changes=0,
                 exact_offset_errors=0,duplicate_ids=0,duplicate_texts=0,partition_errors=0)
    cue_patterns = {"patient":r"\bpatient\b", "copied":r"\bcopied\b", "history":r"\bhistory\b",
                    "record":r"\brecord\b", "plan":r"\bplan\b", "year":r"\byear\b",
                    "current":r"\bcurrent\b", "proposed":r"\bpropos(?:e|ed)\b", "reference":r"\breference\b",
                    **{word:rf"\b{word}\b" for word in ("dated","dates","places","explains","previous","later","occurred","future","identifies")}}
    stats["cue_presence"] = {key:{str(neg):sum(bool(re.search(p,r["text"],re.I)) for r in rows if r["negative_only"]==neg)/sum(r["negative_only"]==neg for r in rows) for neg in (False,True)} for key,p in cue_patterns.items()}
    stats["layout_by_control"] = {str(neg):dict(Counter(c["layout"] for c in coverage if c["negative_only"]==neg)) for neg in (False,True)}
    stats["length_by_control"] = {str(neg):{"long":sum(c["requested_long"] for c in coverage if c["negative_only"]==neg),"rows":sum(c["negative_only"]==neg for c in coverage)} for neg in (False,True)}
    stats["ordering_by_length_and_control"] = {f"long={long},control={neg}":dict(Counter("public_first" if c["public_first"] else "personal_first" for c in coverage if c["requested_long"]==long and c["negative_only"]==neg)) for long in (False,True) for neg in (False,True)}
    payload = "".join(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n" for r in rows).encode()
    corpus_hash = sha(payload)
    manifest = dict(schema_version=1,role="train",split="train",status="training_candidate_authored",
                    annotation_policy="clinical-phi-v2",corpus=str(OUT),sha256=corpus_hash,rows=len(rows),
                    label_support=stats["label_support"],provenance={
                        "author":"author11_record_year_contexts", "method":"Separately composed synthetic clinical family narratives and literal occurrence markup; no source patient records.",
                        "source_script":str(Path(__file__)),"source_script_sha256":sha(Path(__file__).read_bytes()),
                        "exposure":"Policy, normalizer source, window-function AST, local tokenizer/config only. No other corpora, predictions, reports, network or model weights read.",
                        "identity_scheme":"Invented three-part personas hashed after NFC normalization and casefold; each digest satisfies int(first8,16)%10<8. Names are metadata only.",
                        "reserved_identifier_schemes":"No contact, network, government or medical identifiers are emitted. Version codes identify generic documents, never patients.",
                        "date_conventions":"Patient-specific calendar years and document dates are DATE; whole calendar intervals are one DATE; weekdays included; clock times outside; public years and document-version codes excluded."},
                    tokenizer={"path":str(HF/"tokenizer.json"),"sha256":sha((HF/"tokenizer.json").read_bytes()),"max_length":512,"content_window":510,"ownership":"AST extraction of hf_encoder.window_jobs; overlap127, stride256"},
                    support=stats,
                    limitations=["Synthetic templated training data; no clinical truth, representativeness, performance or qualification claim.",
                                 "DATE-only scope; no all-label coverage claim. Sparse patient identities are metadata only.",
                                 "Year-dense long records intentionally repeat meaningful historical or planning anchors across reconciliation sections.",
                                 "Any reported local-context crossing is a deliberate record-context stress case; original gold is preserved.",
                                 "Future dates describe revisable long-term care-plan reviews, with near-term symptomatic care continuing.",
                                 "Public publication references are invented generic educational material, not assertions of a specific real guideline."])
    # Candidate generation is exclusive; the final freeze is a separate review step.
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open("xb") as stream: stream.write(payload)
    with OUT.with_suffix(".jsonl.manifest.json").open("x") as stream: json.dump(manifest,stream,ensure_ascii=False,indent=2)
    for filename,value in (("self_check.json",stats),("window_coverage.json",coverage),("semantic_ledger.json",role_ledger),("synthetic_identity_groups.json",identity_names)):
        with (REPORT/filename).open("x") as stream: json.dump(value,stream,ensure_ascii=False,indent=2)
    print(json.dumps({"corpus":str(OUT),"sha256":corpus_hash,**stats},ensure_ascii=False))

if __name__ == "__main__":
    main()
