import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
from PIL import Image
import numpy as np

loaded_model = load_model('fashion_mnist_model.h5')

def process_image(image):
    # with PIL
    img = Image.open(image)
    img = img.resize((28, 28))
    img = img.convert('L') # grayscale
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array


# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a page:', 
                           ['Prediction', 'Code', 'About'])

if options == 'Prediction': # Prediction page
    st.title('Fashion MNIST Image Classification using CNN')

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # User inputs: image
    image = st.file_uploader('Upload an image:', type=['jpg', 'jpeg', 'png'])
    if image is not None:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button('Classify Image'):
        with st.spinner('Model working....'):
            img_array = process_image(image)
            prediction = loaded_model.predict(img_array).argmax()
            st.success(f'Prediction: {class_names[prediction]}')
            
elif options == 'Code':
    st.header('Code')
    # Add a button to download the Jupyter notebook (.ipynb) file
    notebook_path = 'MNIST_Fashion_Model.ipynb'
    with open(notebook_path, "rb") as file:
        btn = st.download_button(
            label="Download Jupyter Notebook",
            data=file,
            file_name="MNIST_Fashion_Model.ipynb",
            mime="application/x-ipynb+json"
        )
    st.write('You can download the Jupyter notebook to view the code and the model building process.')
    st.write('--'*50)

    st.header('GitHub Repository')
    st.write('You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Fashion-MNIST-Image-Classification)')
    st.write('--'*50)
    
elif options == 'About':
    st.title('About')
    st.write('This web app is created using Streamlit, a Python library used for creating web apps.')
    st.write('The web app uses a Convolutional Neural Network (CNN) model trained on the Fashion MNIST dataset to classify images of clothing.')
    st.write('The model is trained to classify images into 10 different classes of clothing:')
    st.write('1. T-shirt/top')
    st.write('2. Trouser')
    st.write('3. Pullover')
    st.write('4. Dress')
    st.write('5. Coat')
    st.write('6. Sandal')
    st.write('7. Shirt')
    st.write('8. Sneaker')
    st.write('9. Bag')
    st.write('10. Ankle boot')
    st.write('--'*50)
    st.write('The web app is open-source. You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Fashion-MNIST-Image-Classification)')
    st.write('--'*50)

    st.header('Contact')
    st.write('You can contact me for any queries or feedback:')
    st.write('Email: gokulnpc@gmail.com')
    st.write('LinkedIn: [Gokuleshwaran Narayanan](https://www.linkedin.com/in/gokulnpc/)')
    st.write('--'*50)
