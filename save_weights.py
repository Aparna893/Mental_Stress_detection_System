from tensorflow.keras.models import load_model

# Load the already trained centralized model
model = load_model("models/centralized_model.keras")

# Save only the trained weights
model.save_weights("models/centralized_weights.weights.h5")

print("Centralized weights saved successfully!")
print("Location: models/centralized_weights.weights.h5")