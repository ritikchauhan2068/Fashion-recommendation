import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import pickle 
import cv2
from annoy import AnnoyIndex
feature_list = np.array(pickle.load(open('file_extract.pkl','rb')))
filenames = pickle.load(open('file_names.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
img = image.load_img('1163.jpg',target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)


f = feature_list.shape[1]  # feature dimension
annoy_index = AnnoyIndex(f, 'euclidean')  # use euclidean metric

for i, vector in enumerate(feature_list):
    annoy_index.add_item(i, vector)

n_trees = 10  # More trees → higher accuracy
annoy_index.build(n_trees)
n_neighbors = 6
indices = annoy_index.get_nns_by_vector(normalized_result, n_neighbors, include_distances=False)
print(indices)


for file in indices[1:6]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    cv2.waitKey(0)
