import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
import time

# Load ResNet50 model without top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])
start_time = time.time()
# Function to extract features for a batch of images
def extract_features_batch(img_paths, model):
    batch_images = []
    for img_path in img_paths:
        img = image.load_img(img_path, target_size=(224,224))
        img_array = image.img_to_array(img)
        batch_images.append(img_array)
    
    batch_images = np.array(batch_images)
    batch_images = preprocess_input(batch_images)  # preprocess for ResNet50
    
    features = model.predict(batch_images, verbose=0)
    # Normalize each feature vector
    normalized_features = np.array([f / norm(f) for f in features])
    return normalized_features

# Get all image file paths
file_names = [os.path.join('images', f) for f in os.listdir('images')]

# Process images in batches
batch_size = 16  # you can increase/decrease based on your GPU/CPU
file_extract = []

for i in tqdm(range(0, len(file_names), batch_size)):
    batch_files = file_names[i:i+batch_size]
    batch_features = extract_features_batch(batch_files, model)
    file_extract.extend(batch_features)

# Save features and filenames
with open('file_extract.pkl', 'wb') as f:
    pickle.dump(file_extract, f)

with open('file_names.pkl', 'wb') as f:
    pickle.dump(file_names, f)
end_time=time.time()
print(f"Feature extraction completed in {end_time - start_time:.2f} seconds")

