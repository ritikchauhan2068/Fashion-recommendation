import pickle

# Load features
with open('file_extract.pkl', 'rb') as f:
    features = pickle.load(f)

# Load filenames
with open('file_names.pkl', 'rb') as f:
    filenames = pickle.load(f)

# Check data
print(f"Number of feature vectors: {len(features)}")
print(f"Number of filenames: {len(filenames)}")
print(f"Shape of first feature vector: {features[0].shape}")
print(f"First filename: {filenames[0]}")
