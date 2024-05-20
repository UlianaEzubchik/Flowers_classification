import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="Flowers Classifier", page_icon="🌷")

# Load the pre-trained model
@st.cache_resource
def load_model_cached():
    return load_model('Flowers_classifier.keras')

model = load_model_cached()

# Load the category to name mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Define image dimensions
img_width = 224
img_height = 224

# Sidebar for uploading the image
st.sidebar.header('Upload an Image')
image = st.sidebar.file_uploader('Choose an image file', type=['jpg', 'jpeg', 'png'])

# Main header
st.title('Flowers Classification')
st.write('Upload an image of a flower and the model will classify it.')

# Function to preprocess and predict the uploaded image
@st.cache_data
def predict_image(image):
    # Load and preprocess the image
    image_load = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_arr = np.expand_dims(img_arr, axis=0) / 255.0  # Ensure the image is normalized
    
    # Make a prediction
    predictions = model.predict(img_arr)
    score = tf.nn.softmax(predictions[0])
    
    # Get the highest confidence category
    predicted_class = np.argmax(score)
    predicted_label = cat_to_name[str(predicted_class)]  # Use the mapping to get the flower name
    confidence = np.max(score) * 100
    
    return predicted_label, confidence

# Display the image and prediction result
if image is not None:
    st.image(image, caption='Uploaded Image', width=350)
    predicted_label, confidence = predict_image(image)
    
    # Use columns for layout
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Prediction')
        st.write(f'**{predicted_label}**')
    with col2:
        st.subheader('Confidence')
        st.write(f'**{confidence:.2f}%**')

    st.success('Classification Complete!')
else:
    st.info('Please upload an image file to classify.')

# Additional information in the sidebar
st.sidebar.markdown('''
This application uses a Convolutional Neural Network (CNN) MobileNet to classify images of flowers. 
Upload an image to see the classification result.
''')

# Footer or additional information
st.sidebar.markdown('''
---
**Note:** The model is trained to classify images into one of the predefined categories.
''')
