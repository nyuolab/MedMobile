prompt_get_part = f'''

## Question
{{question}} \n

Here is a medical question. Select one of the following topics that this question would best be classified as:

Topic List = [
    "Part 1: The Profession of Medicine",
    "Part 2: Cardinal Manifestations and Presentation of Diseases",
    "Part 3: Pharmacology",
    "Part 4: Oncology and Hematology",
    "Part 5: Infectious Diseases",
    "Part 6: Disorders of the Cardiovascular System",
    "Part 7: Disorders of the Respiratory System",
    "Part 8: Critical Care Medicine",
    "Part 9: Disorders of the Kidney and Urinary Tract",
    "Part 10: Disorders of the Gastrointestinal System",
    "Part 11: Immune-Mediated, Inflammatory, and Rheumatologic Disorders",
    "Part 12: Endocrinology and Metabolism",
    "Part 13: Neurologic Disorders",
    "Part 14: Poisoning, Drug Overdose, and Envenomation",
    "Part 15: Disorders Associated with Environmental Exposures",
    "Part 16: Genes, the Environment, and Disease",
    "Part 17: Global Medicine",
    "Part 18: Aging",
    "Part 19: Consultative Medicine",
    "Part 20: Frontiers"
]

Respond with a number, 1-20 representing the part that the question most likely belongs in. Only respond with this number.
'''

prompt_eval = f'''
## Question
{{question}} \n

## Choices
{{choices}} \n

Here is a medical question, some answer choices. Think through the problem step by step and provide your selection like this:
model generated chain of thought explanation. Therefore, the answer is [final model answer (e.g. A,B,C,D)] for instance [A].
\n
'''

prompt_eval_bare = f'''
## Question
{{question}} \n

## Choices
{{choices}} \n

\n
'''

prompt_eval_bare_fully = f'''
{{question}} \n
{{choices}}
'''

prompt_eval_bare_fully_with_examples = f'''
{{examples}} \n
{{question}} \n
{{choices}}
'''

prompt_eval_with_examples = f'''
## Examples
{{examples}}

Above are examples for medical Q&A.

## Question
{{question}} \n

## Choices
{{choices}} \n

Here is a medical question, some answer choices. Think through the problem step by step and provide your selection like this:
model generated chain of thought explanation. Therefore, the answer is [final model answer (e.g. A,B,C,D)] for instance [A].
\n
'''

prompt_eval_context_bare = f'''
{{context}} \n
{{question}} \n
{{choices}}
'''
prompt_eval_with_context = f'''
## Context
{{context}} \n

## Question
{{question}} \n

## Choices
{{choices}} \n

Here is some context from a textbook, a medical question, some answer choices. Think through the problem step by step and provide your selection like this:
model generated chain of thought explanation. Therefore, the answer is [final model answer (e.g. A,B,C,D)] for instance [A].
\n '''

prompt_eval_with_context_and_examples = f'''
## Examples
{{examples}}

Above are examples of medical Q&A.

## Context
{{context}} \n

## Question
{{question}} \n

## Choices
{{choices}} \n

Here is some context from a textbook, a medical question, some answer choices. Think through the problem step by step and provide your selection like this:
model generated chain of thought explanation. Therefore, the answer is [final model answer (e.g. A,B,C,D)] for instance [A].
\n '''

prompt_example = f'''
## Question
{{question}} \n

## Choices
{{choices}} \n

## Answer
{{answer}} \n
'''

gpt_chain_of_thoughts_MEDPROMPT = {
    "prompt_name": "gpt_chain_of_thoughts",
    "response_type": "MC",
    "examples": [
        {
            "question": """A 6-year-old girl is brought to the emergency department because of right elbow swelling and pain 30 minutes after falling onto her outstretched right arm. She has been unable to move her right elbow since the fall. Physical examination shows bruising, swelling, and tenderness of the right elbow; range of motion is limited by pain. An x-ray of the right arm shows a supracondylar fracture of the humerus with anterior displacement of the proximal fragment. Further evaluation is most likely to show which of the following findings?

A. Absent distal radial pulse
B. Radial deviation of the wrist
C. Inability to abduct shoulder
D. Inability to flex the elbow
""",
            "answer": """A supracondylar fracture of the humerus, especially with anterior displacement of the proximal fragment, is a common and serious injury in children. It can cause injury to the brachial artery, leading to compromised blood flow to the forearm and hand. Therefore, the most likely additional finding in this case would be an absent distal radial pulse due to this potential vascular compromise. This is a serious complication and requires immediate orthopedic consultation and likely surgical intervention to prevent long-term damage.
So, the answer is [A].""",
        },
        {
            "question": """Two weeks after undergoing low anterior resection for rectal cancer, a 52-year-old man comes to the physician because of swelling in both feet. He has not had any fever, chills, or shortness of breath. His temperature is 36°C (96.8°F) and pulse is 88/min. Physical examination shows a normal thyroid and no jugular venous distention. Examination of the lower extremities shows bilateral non-pitting edema that extends from the feet to the lower thigh, with deep flexion creases. His skin is warm and dry, and there is no erythema or rash. Microscopic examination of the interstitial space in this patient's lower extremities would be most likely to show the presence of which of the following?

A. Acellular, protein-poor fluid
B. Lymphocytic, hemosiderin-rich fluid
C. Lipid-rich, protein-rich fluid
D. Protein-rich, glycosaminoglycan-rich fluid
""",
            "answer": """This patient's bilateral, symmetric lower extremity swelling extending to the thighs, nonpitting quality, and absence of systemic symptoms (eg, shortness of breath, orthopnea) are suggestive of lymphedema. Lymphedema develops due to a blockage in the lymphatic system that prevents lymph fluid from draining adequately. In this case, the patient's recent surgery (low anterior resection for rectal cancer) probably involved extensive lymph node dissection, leading to secondary lymphedema.

Microscopically, lymph is a protein-rich fluid that also contains lipids, lymphocytes, and other substances. It does not typically contain significant amounts of glycosaminoglycans or hemosiderin.

Therefore, the answer is [C].""",
        },
        {
            "question": """A 34-year-old gravida 2, para 1 woman at 37+6 weeks of gestation presents for elective cesarean delivery. She says she has been having increased fatigue over the past few weeks. Past medical history includes gestational hypertension for which she has been taking an antihypertensive drug twice daily since week 24. Her vital signs include: temperature 36.7°C (98.0°F), blood pressure 120/75 mm Hg, pulse 127/min. Physical examination reveals generalized pallor. Her laboratory results reveal microcytic, hypochromic anemia with anisocytosis, hemoglobin of 9 g/dL, a differential with 14% lymphocytes, an ESR of 22 mm/hr, and a reticulocyte production index of 3.1. A direct antiglobulin test is positive. LFTs, creatinine, ferritin level, vitamin B12 level, coagulation studies, and urinalysis are normal. Which of the following is the most likely diagnosis in this patient?

A. Preeclampsia
B. Hereditary spherocytosis
C. HELLP syndrome
D. Drug-induced immune hemolytic reaction
""",
            "answer": """The patient's microcytic, hypochromic anemia and a high reticulocyte count are indicative of increased erythropoiesis, which, in this clinical scenario, is likely due to hemolysis. The positive direct antiglobulin test (Coomb's test) confirms that the hemolysis is immune-mediated. Given that other causes of immune hemolysis such as autoimmune disorders and infections are unlikely in this patient, the most probable cause of hemolysis is a drug-induced immune hemolytic reaction. Drugs can cause immune hemolysis by acting as haptens that bind to the red blood cell membrane, causing the formation of antibodies against the drug-cell complex (immune complex type) or by inducing the formation of autoantibodies against red blood cells (autoimmune type). While any drug can potentially cause immune hemolysis, common culprits include penicillins, cephalosporins, quinidine, and antihypertensives like methyldopa. 

Hence, the answer is [D].""",
        },
        {
            "question": """You are counseling a pregnant woman who plans to breast-feed exclusively regarding her newborn's nutritional requirements. The child was born at home and the mother only plans for her newborn to receive vaccinations but no other routine medical care. Which vitamins should be given to the newborn?

A. Folic acid
B. Vitamin K
C. Vitamin D
D. Vitamin K and Vitamin D
""",
            "answer": """The American Academy of Pediatrics (AAP) recommends that all newborns receive a single dose of intramuscular vitamin K at birth. This is because newborns have low stores of vitamin K at birth, leading to a risk of vitamin K deficiency bleeding, also known as hemorrhagic disease of the newborn. This is a potentially life-threatening condition that can cause bleeding into the brain. Because this baby was born at home and the mother plans to avoid routine medical care, this baby may not have received this essential vitamin.

Additionally, the AAP recommends that all infants and children, including those who are breastfed, have a minimum intake of 400 IU/day of vitamin D beginning soon after birth. While breast milk is the best source of most nutrients, it is usually deficient in vitamin D.

The other option, folic acid, is not routinely supplemented in newborns as they usually receive adequate amounts from breast milk or formula.

Therefore, the answer is [D].""",
        },
        {
            "question": """An investigator is studying nutritional deficiencies in humans. A group of healthy volunteers are started on a diet deficient in pantothenic acid. After 4 weeks, several of the volunteers develop irritability, abdominal cramps, and burning paresthesias of their feet. These symptoms are fully reversed after reintroduction of pantothenic acid to their diet. The function of which of the following enzymes was most likely impaired in the volunteers during the study?

A. Methionine synthase
B. Alpha-ketoglutarate dehydrogenase
C. Glutathione reductase
D. Dopamine beta-hydroxylase
""",
            "answer": """Pantothenic acid is also known as vitamin B5, which is a component of coenzyme A (CoA) and phosphopantetheine. CoA is required for the synthesis and oxidation of fatty acids, and the metabolism of carbohydrates and proteins. One of the enzymes that requires CoA for its function is alpha-ketoglutarate dehydrogenase, a key enzyme in the citric acid cycle (also known as Krebs cycle or TCA cycle).

This enzyme catalyzes the conversion of alpha-ketoglutarate to succinyl CoA, with the concomitant reduction of NAD+ to NADH. Deficiencies in pantothenic acid would impair the function of this enzyme, leading to the observed symptoms.

So, the answer is [B].""",
        },
    ],
}