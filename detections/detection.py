import re
from transformers import pipeline
from nltk.corpus import words
from textblob import TextBlob

emotion_pipeline = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
valid_words=set(words.words())

def is_gibberish(text):
    if len(set(text))<=2:
        return True
    
    if re.match(r"^[^a-zA-Z\s]+$", text):  
        return True
    
    vowels = "aeiouAEIOU"
    vowel_count = sum(1 for char in text if char in vowels)
    if vowel_count < max(1, len(text) * 0.2):  
        return True  

    return False 

def is_meaningful(text):
    blob=TextBlob(text)
    corrected_text=str(blob.correct())

    if corrected_text.lower() in valid_words:
        return False
    
    if is_gibberish(text):
        return False
    return True


def detect_text_emotion(text):

    if not text.strip():
        return {"error": "Please enter a statement."}, 400  
    
    if not is_meaningful(text):
        return {"error":"No Emotion Detected. Please enter a valid statement."},400
    
    try:
        result = emotion_pipeline(text)  

        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
            result = result[0]  

        if isinstance(result, list) and all(isinstance(item, dict) for item in result):
            sorted_emotions = sorted(result, key=lambda x: x["score"], reverse=True)

            top_emotion = sorted_emotions[0]
            if top_emotion["label"] == "neutral" and top_emotion["score"] > 0.95:
                return {"error": "No Emotion Detected."}, 400

            emotion_analysis = [
                {"label": em["label"], "score": round(em["score"], 4)}
                for em in sorted_emotions[:3]  # Showing top 5 emotions
            ]

            response = {
                "Dominant_emotion": {
                    "label": top_emotion["label"],  
                    "score": round(top_emotion["score"], 4)  
                },
                "Emotion Analysis": emotion_analysis  
            }
            return response, 200
        
        else:
            return {"error": "Unexpected model output format."}, 500

    except Exception as e:
        return {"error": f"Model error: {str(e)}"}, 500
