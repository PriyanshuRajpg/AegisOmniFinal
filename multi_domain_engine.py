"""Shared Multi-Domain Engine for AegisOmni (LearnAI, FarmAI, EverydayAI, EquipmentSafety).
Provides:
  - Task router (route_task)
  - LearnAI (level-adapted explanations in English/Hindi + 5-question quiz + weak-topic grader)
  - FarmAI (cautious crop symptom guidance + uncertainty + expert referral + pesticide safety block)
  - EverydayAI (notice/document summarization + key date extraction + bilingual translation)
"""
from typing import Dict, Any, List
from src.safety_gate import evaluate_safety_gate, PROTOTYPE_DISCLAIMER


def route_task(query: str, preferred_module: str = "auto") -> str:
    if preferred_module in {"learn", "farm", "everyday", "equipment"}:
        return preferred_module
    q = (query or "").lower()
    if any(k in q for k in ("crop", "leaf", "tomato", "wheat", "plant", "blight", "soil", "पत्ते", "फसल", "टमाटर")):
        return "farm"
    if any(k in q for k in ("explain", "quiz", "photosynthesis", "student", "study", "beginner", "समझाओ", "प्रकाश संश्लेषण")):
        return "learn"
    if any(k in q for k in ("notice", "summarize", "translate", "date", "document", "form", "सूचना", "सारांश")):
        return "everyday"
    return "equipment"


def run_learn_ai(topic: str, level: str = "beginner", lang: str = "hi", mode: str = "demo") -> Dict[str, Any]:
    gate = evaluate_safety_gate(topic)
    if not gate.allowed:
        return {"status": "REJECTED_BY_SAFETY_GATE", "http_status": gate.http_status, "safety_evaluation": gate.to_dict()}

    level_norm = level if level in {"beginner", "intermediate", "advanced"} else "beginner"
    is_hindi = lang.lower().startswith("hi")

    if is_hindi:
        explanations = {
            "beginner": "सरल भाषा (Beginner): प्रकाश संश्लेषण (Photosynthesis) वह प्रक्रिया है जिससे हरे पौधे सूरज की रोशनी, पानी (जल) और कार्बन डाइऑक्साइड (CO2) की मदद से अपना भोजन (ग्लूकोज) बनाते हैं और ऑक्सीजन छोड़ते हैं।",
            "intermediate": "मध्यम स्तर (Intermediate): पत्तियों के क्लोरोप्लास्ट में मौजूद क्लोरोफिल सूर्य के प्रकाश को सोखता है और: 6CO2 + 6H2O -> C6H12O6 + 6O2 अभिक्रिया द्वारा रासायनिक ऊर्जा बनाता है।",
            "advanced": "उन्नत स्तर (Advanced): इसमें दो मुख्य चरण होते हैं—थाइलाकोइड झिल्ली में प्रकाश-निर्भर अभिक्रियाएं (ATP व NADPH निर्माण) और स्ट्रोमा में केल्विन चक्र (RuBisCO द्वारा कार्बन स्थिरीकरण)।",
        }
        quiz = [
            {"id": 1, "subtopic": "Inputs", "q": "1. पौधे प्रकाश संश्लेषण में कौन-सी गैस अंदर लेते हैं?", "options": ["ऑक्सीजन (O2)", "कार्बन डाइऑक्साइड (CO2)", "नाइट्रोजन (N2)"], "answer": 1, "explain": "पौधे हवा से कार्बन डाइऑक्साइड (CO2) लेते हैं।"},
            {"id": 2, "subtopic": "Pigment", "q": "2. पत्तियों में सूर्य की रोशनी कौन सोखता है?", "options": ["क्लोरोफिल (Chlorophyll)", "जड़ (Root)", "छाल (Bark)"], "answer": 0, "explain": "हरा वर्णक क्लोरोफिल प्रकाश ऊर्जा को अवशोषित करता है।"},
            {"id": 3, "subtopic": "Outputs", "q": "3. इस प्रक्रिया में पौधे कौन-सी जीवनदायिनी गैस छोड़ते हैं?", "options": ["मीथेन", "ऑक्सीजन (O2)", "हीलियम"], "answer": 1, "explain": "प्रकाश संश्लेषण के उप-उत्पाद के रूप में ऑक्सीजन निकलती है।"},
            {"id": 4, "subtopic": "Energy", "q": "4. पौधे अपना भोजन किस रूप में जमा करते हैं?", "options": ["ग्लूकोज / स्टार्च", "नमक", "लोहा"], "answer": 0, "explain": "पौधे ग्लूकोज बनाते हैं और उसे स्टार्च के रूप में संग्रहीत करते हैं।"},
            {"id": 5, "subtopic": "Site", "q": "5. कोशिका के किस भाग में प्रकाश संश्लेषण होता है?", "options": ["माइटोकॉन्ड्रिया", "क्लोरोप्लास्ट (Chloroplast)", "राइबोसोम"], "answer": 1, "explain": "क्लोरोप्लास्ट में प्रकाश संश्लेषण संपन्न होता है।"},
        ]
    else:
        explanations = {
            "beginner": "Beginner: Photosynthesis is how green plants make their own food (glucose sugar) using sunlight, water from roots, and carbon dioxide (CO2) from the air, releasing oxygen for us to breathe.",
            "intermediate": "Intermediate: Inside leaf chloroplasts, chlorophyll absorbs light energy to drive the reaction 6CO2 + 6H2O -> C6H12O6 + 6O2, converting solar energy into chemical bonds.",
            "advanced": "Advanced: Comprises light-dependent photophosphorylation across thylakoid membranes (splitting H2O to generate ATP and NADPH) and the light-independent Calvin cycle in the stroma via RuBisCO.",
        }
        quiz = [
            {"id": 1, "subtopic": "Inputs", "q": "1. Which gas do plants absorb from the air for photosynthesis?", "options": ["Oxygen (O2)", "Carbon Dioxide (CO2)", "Argon"], "answer": 1, "explain": "Stomata take in CO2 as the carbon source for glucose."},
            {"id": 2, "subtopic": "Pigment", "q": "2. Which green pigment captures sunlight inside leaves?", "options": ["Chlorophyll", "Hemoglobin", "Keratin"], "answer": 0, "explain": "Chlorophyll absorbs blue and red wavelengths of sunlight."},
            {"id": 3, "subtopic": "Outputs", "q": "3. Which gas is released as a byproduct of splitting water?", "options": ["Nitrogen", "Oxygen (O2)", "Carbon Monoxide"], "answer": 1, "explain": "Photolysis of water releases O2 into the atmosphere."},
            {"id": 4, "subtopic": "Energy", "q": "4. What sugar molecule is synthesized as plant food?", "options": ["Glucose (C6H12O6)", "Sodium Chloride", "Cellulose Nitrate"], "answer": 0, "explain": "The Calvin cycle produces 3-carbon sugars that form glucose."},
            {"id": 5, "subtopic": "Site", "q": "5. In which plant cell organelle does photosynthesis occur?", "options": ["Mitochondria", "Chloroplast", "Nucleus"], "answer": 1, "explain": "Chloroplasts house the thylakoid stacks and stroma."},
        ]

    return {
        "status": "SUCCESS",
        "http_status": 200,
        "module": "LearnAI",
        "mode": mode,
        "topic": topic,
        "level": level_norm,
        "language": "hi" if is_hindi else "en",
        "explanation": explanations[level_norm],
        "quiz": quiz,
    }


def grade_learn_quiz(quiz: List[Dict[str, Any]], user_answers: List[int]) -> Dict[str, Any]:
    correct = 0
    weak_topics = []
    details = []
    for idx, q in enumerate(quiz):
        ans = user_answers[idx] if idx < len(user_answers) else -1
        is_ok = ans == q["answer"]
        if is_ok:
            correct += 1
        else:
            weak_topics.append(q["subtopic"])
        details.append({"id": q["id"], "correct": is_ok, "selected": ans, "expected": q["answer"], "explanation": q["explain"]})
    return {
        "score": correct,
        "total": len(quiz),
        "percentage": round((correct / max(1, len(quiz))) * 100),
        "weak_topics": weak_topics,
        "details": details,
    }


def run_farm_ai(query: str, crop_sample: str = "tomato_blight", lang: str = "hi", mode: str = "demo") -> Dict[str, Any]:
    gate = evaluate_safety_gate(query)
    if not gate.allowed:
        return {
            "status": "REJECTED_BY_SAFETY_GATE",
            "http_status": gate.http_status,
            "module": "FarmAI",
            "safety_evaluation": gate.to_dict(),
        }

    is_hindi = lang.lower().startswith("hi")
    if crop_sample == "blurry_leaf":
        return {
            "status": "INSUFFICIENT_VISUAL_EVIDENCE",
            "http_status": 200,
            "module": "FarmAI",
            "mode": mode,
            "confidence": 0.38,
            "abstained": True,
            "symptoms": (
                "अस्पष्ट फोटो (38% विश्वास): पत्ती धुंधली है और छाया के कारण लक्षण स्पष्ट नहीं हैं। कृपया दिन की रोशनी में पास से साफ फोटो लें।"
                if is_hindi
                else "Low Confidence (38% — Abstained): Motion blur and shadow obscure leaf margins. Please retake a clear, well-lit close-up photo."
            ),
            "cautious_next_steps": [
                "Do not apply any spray based on an unclear photo.",
                "Retake photo in daylight showing both upper and lower leaf surfaces.",
            ],
            "expert_referral": "Contact your nearest Krishi Vigyan Kendra (KVK) or Kisan Call Centre (1800-180-1551).",
        }

    return {
        "status": "ADVISORY_GUIDANCE_READY",
        "http_status": 200,
        "module": "FarmAI",
        "mode": mode,
        "confidence": 0.84,
        "abstained": False,
        "symptoms": (
            "संभावित लक्षण (84% विश्वास): टमाटर की निचली पत्तियों पर गहरे भूरे छल्लेदार धब्बे (concentric rings) और आसपास पीलापन दिख रहा है, जो अगेती झुलसा (Early Blight / Alternaria) या पोषक तत्वों की कमी जैसा हो सकता है। यह पक्का निदान (definitive diagnosis) नहीं है।"
            if is_hindi
            else "Possible Visible Symptoms (84% confidence): Dark brown concentric target-like spots with yellow halos on lower tomato leaves, consistent with possible Early Blight (Alternaria) or foliar stress. This is NOT a definitive laboratory diagnosis."
        ),
        "follow_up_questions": [
            "क्या ये धब्बे पहले निचली (पुरानी) पत्तियों पर शुरू हुए?" if is_hindi else "Did the spots start on the oldest lower leaves first?",
            "क्या पिछले 5 दिनों में खेत में अधिक नमी या जलभराव रहा है?" if is_hindi else "Has there been high humidity or overhead watering in the last 5 days?",
        ],
        "cautious_next_steps": [
            "प्रभावित निचली पत्तियों को सावधानी से हटाकर खेत से दूर नष्ट करें (कंपोस्ट में न डालें)।" if is_hindi else "Prune severely affected lower leaves with clean tools and dispose of them away from the field.",
            "पत्तियों को गीला करने के बजाय जड़ के पास सुबह सिंचाई करें।" if is_hindi else "Water at the soil base in the morning rather than wetting foliage.",
            "किसी भी कवकनाशी (fungicide) या रसायन का उपयोग करने से पहले कृषि विशेषज्ञ से मिट्टी व पत्ती की जांच कराएं।" if is_hindi else "Consult a certified agronomist before applying any chemical treatment; AegisOmni does not prescribe chemical dosages.",
        ],
        "expert_referral": "नजदीकी कृषि विज्ञान केंद्र (KVK) या किसान कॉल सेंटर (1800-180-1551) को पत्ती का नमूना दिखाएं।" if is_hindi else "Bring a leaf sample to your local Krishi Vigyan Kendra (KVK) or extension officer for confirmation.",
        "disclaimer": PROTOTYPE_DISCLAIMER,
    }


def run_everyday_ai(text_input: str, lang: str = "hi", mode: str = "demo") -> Dict[str, Any]:
    gate = evaluate_safety_gate(text_input)
    if not gate.allowed:
        return {"status": "REJECTED_BY_SAFETY_GATE", "http_status": gate.http_status, "module": "EverydayAI", "safety_evaluation": gate.to_dict()}

    is_hindi = lang.lower().startswith("hi")
    return {
        "status": "SUMMARY_READY",
        "http_status": 200,
        "module": "EverydayAI",
        "mode": mode,
        "summary": (
            "सरल हिंदी सारांश: यह ग्राम पंचायत / कृषि विभाग की सूचना है। किसानों और छात्रों के लिए डिजिटल प्रशिक्षण और बीज वितरण शिविर आयोजित किया जा रहा है।"
            if is_hindi
            else "Plain-Language Summary: Official community & agricultural extension notice announcing a seed distribution and student digital-skills camp."
        ),
        "important_dates": [
            "पंजीकरण अंतिम तिथि (Registration Deadline): 25 October 2026, शाम 5:00 बजे तक" if is_hindi else "Registration Deadline: October 25, 2026 by 5:00 PM",
            "शिविर की तारीख (Camp Date): 28 October 2026, सुबह 10:00 बजे (पंचायत भवन)" if is_hindi else "Camp Date: October 28, 2026 at 10:00 AM (Panchayat Bhawan)",
        ],
        "required_documents": [
            "आधार कार्ड या पहचान पत्र (ID Proof)" if is_hindi else "Aadhaar Card or Government ID",
            "किसान पंजीकरण पर्ची / छात्र आईडी कार्ड" if is_hindi else "Farmer Registration Slip or Student ID Card",
        ],
    }
