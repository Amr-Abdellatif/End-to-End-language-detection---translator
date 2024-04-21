from fastapi import FastAPI
from pydantic import BaseModel
from language_detection import *
import time
from translation_utils_eng_ara import *
from translation_utils_ara_eng import *
from language_checker import detect_language
import uvicorn

app = FastAPI()


# Define response body for translation endpoint
class TranslationResponse(BaseModel):
    original_sentence: str 
    translated_text: str
    processing_time: float

# Define response body
class PredictionResponse(BaseModel):
    predicted_language: str
    processing_time: float  # Add processing time field


model, cv, le = load_model_and_vectorizer(language_detection_path) # loading language detection model


@app.get("/")
async def home():
    return {'Home': 'Hello, World!'}


# Define endpoint to predict language
@app.post("/predict_language/", response_model=PredictionResponse)
async def predict_language(Sentence: str):
    start_time = time.time()  # Record start time
    # Predict the language
    x = cv.transform([Sentence]).toarray()
    predicted_language_id = model.predict(x)[0]
    predicted_language = le.inverse_transform([predicted_language_id])[0] # return actual value
    end_time = time.time()  # Record end time
    processing_time = end_time - start_time  # Calculate processing time
    print(f"Request took {processing_time} seconds.")
    return {"predicted_language": predicted_language, "processing_time": processing_time}  # Include processing time in the response body


@app.post("/translate/", response_model=TranslationResponse)
async def translate_text(Sentence: str):
    start_time = time.time()
    
    language_detector = detect_language(Sentence)

    if language_detector == "Arabic":
        translated_text = evaluateSpecificSentence_ara_eng(encoder, decoder, Sentence, input_lang, output_lang)
    elif language_detector == "English":
        translated_text = evaluate_specific_sentence_eng_ara(encoder1, decoder1, Sentence, input_lang1, output_lang1)
    else:
        translated_text = "Unknown language"

    end_time = time.time()
    processing_time = end_time - start_time
    return {"original_sentence": Sentence, "translated_text": translated_text, "processing_time": processing_time}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)