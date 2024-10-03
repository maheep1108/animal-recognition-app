from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from flask_cors import CORS

app = Flask(__name__)  # Initialize the Flask app
CORS(app)  # Enable CORS after the app is initialized

@app.route('/')
def index():
    return "Flask API for Animal Recognition is running"

# Load the fine-tuned MobileNetV2 model
model = load_model("animal_recognition_model.h5")

# Class names for the Stanford Dogs dataset (replace with the actual class names of your breeds)
class_names = [
    'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekingese', 'Shih-Tzu',
    'Blenheim_spaniel', 'papillon', 'toy_terrier', 'Rhodesian_ridgeback', 'Afghan_hound',
    'basset', 'beagle', 'bloodhound', 'bluetick', 'black_and_tan_coonhound',
    'Walker_hound', 'English_foxhound', 'redbone', 'borzoi', 'Irish_wolfhound',
    'Italian_greyhound', 'whippet', 'Ibizan_hound', 'Norwegian_elk_hound', 'otterhound',
    'Saluki', 'Scottish_deerhound', 'Weimaraner', 'Staffordshire_bullterrier',
    'American_Staffordshire_terrier', 'Bedlington_terrier', 'Border_terrier', 
    'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier', 'Norwich_terrier',
    'Yorkshire_terrier', 'wire_haired_fox_terrier', 'Lakeland_terrier', 'Sealyham_terrier',
    'Airedale', 'cairn', 'Australian_terrier', 'Dandie_Dinmont', 'Boston_bull', 
    'miniature_schnauzer', 'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier',
    'Tibetan_terrier', 'silky_terrier', 'soft_coated_wheaten_terrier', 'West_Highland_white_terrier',
    'Lhasa', 'flat-coated_retriever', 'curly-coated_retriever', 'golden_retriever', 'Labrador_retriever',
    'Chesapeake_Bay_retriever', 'German_short-haired_pointer', 'vizsla', 'English_setter',
    'Irish_setter', 'Gordon_setter', 'Brittany_spaniel', 'clumber', 'English_springer',
    'Welsh_springer_spaniel', 'cocker_spaniel', 'Sussex_spaniel', 'Irish_water_spaniel', 
    'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor',
    'Old_English_sheepdog', 'Shetland_sheepdog', 'collie', 'Border_collie', 
    'Bouvier_des_Flandres', 'Rottweiler', 'German_shepherd', 'Doberman', 
    'miniature_pinscher', 'Greater_Swiss_Mountain_dog', 'Bernese_mountain_dog', 
    'Appenzeller', 'EntleBucher', 'boxer', 'bull_mastiff',
    'Tibetan_mastiff', 'French_bulldog', 'Great_Dane', 'Saint_Bernard', 'Eskimo_dog',
    'malamute', 'Siberian_husky', 'affenpinscher', 'basenji', 'pug', 'Leonberg', 
    'Newfoundland', 'Great_Pyrenees', 'Samoyed', 'Pomeranian', 'chow', 'keeshond', 
    'Brabancon_griffon', 'Pembroke', 'Cardigan', 'toy_poodle', 'miniature_poodle',
    'standard_poodle', 'Mexican_hairless', 'dingo', 'dhole', 'African_hunting_dog'
]


def prepare_image(image):
    """Preprocess the uploaded image for the model."""
    img = image.resize((224, 224))  # Resize to match the input size of the model
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (required by the model)
    return img_array

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image = request.files['image']
        img = Image.open(image)

        # Preprocess the image for the model
        img_array = prepare_image(img)

        # Perform prediction
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        result = class_names[class_index]  # Get the predicted class name

        return jsonify({"classification": result})

    except Exception as e:
        # Return the full error message for debugging
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
