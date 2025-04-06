import os
import json
import base64
import numpy as np
import pickle
from gtts import gTTS
from PIL import Image
from django.shortcuts import render
from django.http import JsonResponse, FileResponse
from tensorflow.keras.models import load_model
from .training import predict_alzheimer, brain_predict, predict_disease
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.files.storage import default_storage

# Get app directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Load symptoms list
with open(os.path.join(APP_DIR, "symptoms.json"), "r") as f:
    symptoms_list = json.load(f)

# Diabetes models
with open(os.path.join(APP_DIR, 'models', 'diabetes-model.pkl'), 'rb') as model_file:
    diabetes_model = pickle.load(model_file)
with open(os.path.join(APP_DIR, 'models', 'scaler.pkl'), 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Alzheimer model
alzheimer_model = load_model(os.path.join(APP_DIR, 'models', 'alzheimer-model.h5'))

# Heart model
with open(os.path.join(APP_DIR, 'models', 'heart-model.pkl'), 'rb') as model_file:
    heart_model = pickle.load(model_file)

# Kidney model
with open(os.path.join(APP_DIR, 'models', 'kidney-model.pkl'), 'rb') as model_file:
    kidney_model = pickle.load(model_file)

# Uploads folder
UPLOAD_FOLDER = os.path.join(settings.MEDIA_ROOT, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


from django.shortcuts import render

def all_cards_view(request):
    cards = [
        {"title": "Alzheimer's Disease", "desc": "Detect Alzheimer's symptoms from MRI scans.", "url": "alzheimer"},
        {"title": "Brain Tumor", "desc": "Identify presence of brain tumors using MRI.", "url": "brain"},
        {"title": "Diabetes", "desc": "Predict the likelihood of diabetes with health data.", "url": "diabetes"},
        {"title": "General Disease", "desc": "Analyze common symptoms to detect diseases.", "url": "general"},
        {"title": "Heart Disease", "desc": "Check your heart health with quick metrics.", "url": "heart"},
        {"title": "Kidney Disease", "desc": "Evaluate kidney function and detect issues.", "url": "kidney"},
    ]
    return render(request, 'ML/all_cards.html', {"cards": cards})

@csrf_exempt
def alzheimer_view(request):
    if request.method == 'POST' and request.FILES.get("image"):
        file = request.FILES['image']
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        with default_storage.open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        label = predict_alzheimer(file_path)
        with open(file_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode()
        return JsonResponse({
            "label": label,
            "photo": f"data:image/png;base64,{image_base64}"
        })
    return render(request, 'ML/alzheimer.html')


@csrf_exempt
def diabetes_view(request):
    if request.method == 'POST':
        try:
            fields = [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            ]
            features = [float(request.POST.get(field)) for field in fields]
            features = scaler.transform(np.array(features).reshape(1, -1))
            prediction = diabetes_model.predict(features)[0]
            result = "Diabetic" if prediction == 1 else "Not Diabetic"
            return render(request, 'ML/diabetes.html', {'result': result})
        except Exception:
            return render(request, 'ML/diabetes.html', {'error': "Invalid input! Please enter valid numbers."})
    return render(request, 'ML/diabetes.html')


@csrf_exempt
def heart_view(request):
    if request.method == 'POST':
        try:
            features = [float(request.POST.get(f)) for f in [
                'age', 'gender', 'chest_pain', 'resting_bp', 'cholesterol',
                'fasting_bs', 'resting_ecg', 'max_hr', 'exercise_angina',
                'oldpeak', 'st_slope', 'ca', 'thal'
            ]]
            prediction = heart_model.predict(np.array(features).reshape(1, -1))
            result = "You have heart disease." if prediction[0] == 1 else "You do not have heart disease."
            return render(request, 'ML/heart.html', {'prediction_result': result})
        except Exception as e:
            return render(request, 'ML/heart.html', {'error': f"Error during prediction: {str(e)}"})
    return render(request, 'ML/heart.html')


@csrf_exempt
def brain_view(request):
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        with default_storage.open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        prediction = brain_predict(file.name)
        return JsonResponse({
            "image_name": file.name,
            "prediction": prediction
        })
    return render(request, 'ML/brain.html')


@csrf_exempt
def kidney_view(request):
    if request.method == 'POST':
        try:
            features = [float(request.POST.get(f)) for f in [
                'sg', 'htn', 'hemo', 'dm', 'al',
                'appet', 'rc', 'pc'
            ]]
            prediction = kidney_model.predict(np.array(features).reshape(1, -1))
            result = "You have kidney disease." if prediction[0] == 1 else "You do not have kidney disease."
            return render(request, 'ML/kidney.html', {'prediction_result': result})
        except Exception as e:
            return render(request, 'ML/kidney.html', {'error': f"Error during prediction: {str(e)}"})
    return render(request, 'ML/kidney.html')


@csrf_exempt
def general_view(request):
    if request.method == "POST":
        user_symptoms = request.POST.getlist("symptoms[]")  # Get symptoms as a list
        days = int(request.POST.get("days", 5))
        if not user_symptoms:
            return JsonResponse({"error": "No symptoms entered"}), 400
        advice, predictions = predict_disease(user_symptoms, days)
        response = {
            "advice": advice,
            "predictions": []
        }
        for disease, details in predictions.items():
            response["predictions"].append({
                "disease": disease,
                "description": details["desc"],
                "precautions": details["prec"],
                "medications": details["drugs"]["Medications"],
                "diet": details["drugs"]["Diet"]
            })
            print(response)

        #**Generate Speech**
        # speech_text = "The predicted disease is: "
        # speech_text += f" disease_name. Description: {details['desc']}. "
        # speech_text += f"Precautions to take: {', '.join(details['prec'])}. "
        # speech_text += f"Recommended medications: {', '.join(details['drugs']['Medications'])}. "
        # speech_text += f"Suggested diet: {', '.join(details['drugs']['Diet'])}. "
        # tts = gTTS(text=speech_text, lang='en')
        # audio_file_path = os.path.join(settings.BASE_DIR, 'static', 'ML', 'speech.mp3')
        # tts.save(audio_file_path)

        return JsonResponse(response)
    return render(request, 'ML/general.html', {'symptoms': symptoms_list})


def speak(request):
    audio_path = os.path.join(settings.BASE_DIR, 'static', 'ML', 'speech.mp3')
    return FileResponse(open(audio_path, 'rb'), content_type='audio/mpeg')
