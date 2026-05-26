from keras.models import load_model

# Load existing keras model
model = load_model("deepfake_model.keras")

# Save as h5
model.save("deepfake_model.h5")

print("Converted Successfully!")