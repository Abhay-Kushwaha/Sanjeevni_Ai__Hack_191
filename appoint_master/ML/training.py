import pandas as pd
import pickle
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler,MultiLabelBinarizer)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from werkzeug.utils import secure_filename
from PIL import Image
import joblib
import base64

from django.conf import settings  # needed for BASE_DIR
# Load the drug dataset only once when module is imported
# df = pd.read_csv(os.path.join(settings.BASE_DIR, 'ML/dataset/disease_drug.csv'))


def diabetes():
    dataset_path = "dataset/diabetes.csv"
    if not os.path.exists(dataset_path):
        print("Dataset not found!")
        return
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Diabetes Model Accuracy: {accuracy * 100:.2f}%")

    # Save Model & Scaler
    os.makedirs("models", exist_ok=True)
    with open("models/diabetes-model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    with open("models/scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    # print("Diabetes model and scaler saved successfully!")


def predict_alzheimer(img_path):
    try:
        model1 = load_model(os.path.join(settings.BASE_DIR, 'ML', 'models', 'brain.h5'), compile=False)
        class2label = {0: 'Mild Demented', 1: 'Moderate Demented', 2: 'Non Demented', 3: 'Very Mild Demented'}
        img = Image.open(img_path).convert("RGB").resize((224, 224))

        # Convert to numpy array and normalize
        img_array = np.asarray(img, dtype=np.float32) / 255.0
        img_array = img_array.reshape((1, *img_array.shape))
        class2label = {0: 'Mild Demented', 1: 'Moderate Demented', 2: 'Non Demented', 3: 'Very Mild Demented'}

        # Make prediction
        predictions = model1.predict(img_array)
        pred_class = np.argmax(predictions[0])  # Get the class with highest probability
        pred_label = class2label.get(pred_class, "Undefined")
        # print(f"🩺 Predicted Class: {pred_class}, Label: {pred_label}") 
        return pred_label
    except Exception as e:
        # print("Error in prediction:", e)
        return "Undefined"
    

def heart():
    data = pd.read_csv('dataset/heart.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    with open('models/heart-model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    # print("Heart disease prediction model trained and saved successfully.")

def brain_predict(input_img):
    saved_model = load_model(os.path.join(settings.BASE_DIR, 'ML', 'models', 'brain.h5'), compile=False)
    saved_model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    print("Processing image : " + input_img)

    # Load and preprocess the image
    img_path = os.path.join(settings.MEDIA_ROOT, 'uploads', input_img)  # Ensure correct file path
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        return "Error: Image not found"
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    output = saved_model.predict(img)
    print("Model Output:", output)

    # Convert output to boolean (assuming binary classification)
    status = "Tumor Detected" if output[0][0] == 1 else "No Tumor"
    return status


def kidney():
    dataset = pd.read_csv('dataset/kidney.csv')
    # Dropping unneccsary feature :
    dataset = dataset.drop('id', axis=1)

    # Replacing Categorical Values with Numericals
    dataset['rbc'] = dataset['rbc'].replace(to_replace = {'normal' : 0, 'abnormal' : 1})
    dataset['pc'] = dataset['pc'].replace(to_replace = {'normal' : 0, 'abnormal' : 1})
    dataset['pcc'] = dataset['pcc'].replace(to_replace = {'notpresent':0,'present':1})
    dataset['ba'] = dataset['ba'].replace(to_replace = {'notpresent':0,'present':1})
    dataset['htn'] = dataset['htn'].replace(to_replace = {'yes' : 1, 'no' : 0})
    dataset['dm'] = dataset['dm'].replace(to_replace = {'\tyes':'yes', ' yes':'yes', '\tno':'no'})
    dataset['dm'] = dataset['dm'].replace(to_replace = {'yes' : 1, 'no' : 0})
    dataset['cad'] = dataset['cad'].replace(to_replace = {'\tno':'no'})
    dataset['cad'] = dataset['cad'].replace(to_replace = {'yes' : 1, 'no' : 0})
    dataset['appet'] = dataset['appet'].replace(to_replace={'good':1,'poor':0,'no':np.nan})
    dataset['pe'] = dataset['pe'].replace(to_replace = {'yes' : 1, 'no' : 0})
    dataset['ane'] = dataset['ane'].replace(to_replace = {'yes' : 1, 'no' : 0})
    dataset['classification'] = dataset['classification'].replace(to_replace={'ckd\t':'ckd'})
    dataset["classification"] = [1 if i == "ckd" else 0 for i in dataset["classification"]]

    # Coverting Objective into Numericals:
    dataset['pcv'] = pd.to_numeric(dataset['pcv'], errors='coerce')
    dataset['wc'] = pd.to_numeric(dataset['wc'], errors='coerce')
    dataset['rc'] = pd.to_numeric(dataset['rc'], errors='coerce')

    # Handling Missing Values:
    features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    for feature in features:
        dataset[feature] = dataset[feature].fillna(dataset[feature].median())

    # Dropping feature (Multicollinearity):
    dataset.drop('pcv', axis=1, inplace=True)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    X = dataset[['sg', 'htn', 'hemo', 'dm', 'al', 'appet', 'rc', 'pc']]
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=33)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    with open('models/kidney-model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    # print("Kidney disease prediction model trained and saved successfully.")

def general_disease():
    df = pd.read_csv("dataset/symptoms_df.csv")
    df['Symptoms'] = df[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']].values.tolist()
    df['Symptoms'] = df['Symptoms'].apply(lambda x: list(set(s.strip().lower() for s in x if pd.notnull(s) and s.strip())))
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df['Symptoms'])
    y = df['Disease']
    model = RandomForestClassifier()
    model.fit(X, y)
    # Save model and symptom encoder
    with open('general_disease.pkl', 'wb') as f:
        pickle.dump((model, mlb), f)
    # print("Model saved as general_disease.pkl")

def predict_disease(user_symptoms, days):
    with open('ML/models/general_disease.pkl', 'rb') as f:
        model, mlb = pickle.load(f)
    user_symptoms = [s.strip().lower() for s in user_symptoms if s.strip()]
    known_symptoms = set(mlb.classes_)
    filtered_symptoms = [s for s in user_symptoms if s in known_symptoms]
    if not filtered_symptoms:  # If no valid symptoms are found
        return "No valid symptoms detected. Please enter known symptoms.", {}
    input_vector = mlb.transform([filtered_symptoms])
    predicted_diseases = model.predict(input_vector)

    # Get the top 2 most common diseases
    unique, counts = np.unique(predicted_diseases, return_counts=True)
    sorted_diseases = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:2]
    print("Predicted Diseases:", sorted_diseases)

    # Load descriptions & precautions
    description_dict, precautions_dict = {}, {}
    with open("ML/dataset/symptom_description.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)  # Skip header
        for row in csv_reader:
            description_dict[row[0].strip().lower()] = row[1].strip()
    with open("ML/dataset/symptom_precaution.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)  # Skip header
        for row in csv_reader:
            disease_name = row[0].strip().lower()
            precautions_dict[disease_name] = [p.strip().capitalize() for p in row[1:] if p.strip()]

    # Build response
    output = {}
    for disease, _ in sorted_diseases:
        disease_lower = disease.strip().lower()
        output[disease] = {
            "desc": description_dict.get(disease_lower, "No description available"),
            "prec": precautions_dict.get(disease_lower, ["No precautions available"]),
            "drugs": get_drugs_for_disease(disease_lower)  # Pass lowercase disease name
        }
        print(f"Disease: {disease}, Desc: {output[disease]['desc']}, Precautions: {output[disease]['prec']}, Drugs: {output[disease]['drugs']}")

    return "If symptoms persist, consult a doctor.", output

def get_drugs_for_disease(disease):
    df = pd.read_csv("ML/dataset/medicine.csv")
    df["Disease"] = df["Disease"].str.lower()
    disease_lower = disease.strip().lower()
    disease_data = df[df["Disease"].str.contains(disease_lower, na=False, case=False)]
    if disease_data.empty:
        print(f"No medicine found for {disease_lower}")
        return {"Medications": ["No drug found"], "Diet": ["No dietary recommendations"]}
    medications = eval(disease_data.iloc[0]["Medication"])
    diet = eval(disease_data.iloc[0]["Diet"])

    return {"Medications": medications, "Diet": diet}
# Call
# if __name__ == "__main__":
    # diabetes()
    # heart()
    # kidney()
    # general_disease()
