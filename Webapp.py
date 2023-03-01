import streamlit as st
import numpy as np
import tensorflow as tf
import h5py
from tensorflow.keras.models import load_model
# import cv2 as cv
from PIL import Image, ImageOps


vgg16 = load_model('vgg_16.h5')

st.set_page_config(page_title='Bike or Car', layout='wide')
st.header("Bike VS Car Classification")
# st.footer("Project by Abhishek Biswas")
st.write("This is a Vgg 16 model which is fine tuned to classify images of bikes and cars")

file = st.file_uploader("Kindly upload an image here", type = ['jpg', 'png'])

def pred(img):
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.ANTIALIAS)
    img = np.asarray(img)
    img_reshape = img[np.newaxis, ...]
    p = vgg16.predict(img_reshape)

    return p

if file is not None:
    img = Image.open(file)
    st.image(img, width = 300 )
    p = pred(img)
    class_names = ['Bike', 'Car']
    s = "This Image is Most Likely a : "+class_names[np.argmax(p)]
    st.success(s)
