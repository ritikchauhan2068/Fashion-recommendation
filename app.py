import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import pickle 
import cv2
import os 
from PIL import Image
from annoy import AnnoyIndex


file_extract = np.array(pickle.load(open('file_extract.pkl', 'rb')))
file_names = np.array(pickle.load(open('file_names.pkl', 'rb')))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def build_annoy_index(feature_list, index_path="annoy_index.ann", n_trees=10):
    f = feature_list.shape[1]   # feature dimension
    annoy_index = AnnoyIndex(f, 'euclidean')

    if os.path.exists(index_path):  # If index file already exists, load it
        annoy_index.load(index_path)
        return annoy_index

    # Otherwise build from scratch
    for i, vector in enumerate(feature_list):
        annoy_index.add_item(i, vector)

    annoy_index.build(n_trees)
    annoy_index.save(index_path)  # Save index to disk
    return annoy_index

# Build Annoy index once
annoy_index = build_annoy_index(file_extract)

def recommend(features, annoy_index, filenames, n_neighbors=6):
    indices = annoy_index.get_nns_by_vector(features, n_neighbors, include_distances=False)
    return [filenames[i] for i in indices]

# ---------------- Streamlit UI ---------------- #
st.title('Fashion Recommender System')

uploaded_file = st.file_uploader("Choose an image")

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('saved', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display uploaded image
        display = Image.open(uploaded_file)
        st.image(display, caption="Uploaded Image", use_container_width=True)

        # feature extraction
        features = feature_extraction(os.path.join("saved", uploaded_file.name), model)

        # recommendation
        indices = recommend(features, annoy_index, file_names)

        cols = st.columns(5)
        for i, col in enumerate(cols):
            if i < len(indices):   # safety check
                with col:
                    st.image(indices[i], use_container_width=True)

    else:
        st.header("Some error while saving file")
