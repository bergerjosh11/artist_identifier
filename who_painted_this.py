import argparse
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO
import numpy as np

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Function to predict the artist given an image URL
def predict_artist(image_url):
    response = requests.get(image_url)
    img = image.load_img(BytesIO(response.content), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions)[0]

    # Get the top predicted class and its probability
    top_prediction = decoded_predictions[0]

    return top_prediction[1]

def main():
    parser = argparse.ArgumentParser(description='Predict the artist of a painting using AI.')
    parser.add_argument('image_url', type=str, help='URL of the painting image')
    args = parser.parse_args()

    predicted_artist = predict_artist(args.image_url)
    print(f"The predicted artist of the painting is: {predicted_artist}")

if __name__ == '__main__':
    main()
