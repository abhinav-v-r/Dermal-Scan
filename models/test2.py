import numpy as np
from tensorflow.keras.preprocessing import image

def predict_skin_condition(img_path, model, class_labels, threshold=0.3):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))  # adjust size if needed
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)[0]
    results = dict(zip(class_labels, preds))

    # Filter based on threshold
    filtered_results = {cls: round(prob * 100, 2) for cls, prob in results.items() if prob >= threshold}

    # Get top prediction
    pred_class = max(results, key=results.get)

    return pred_class, filtered_results

# Example usage
test_img = "/content/drive/MyDrive/Dherma_Scan_AI/ratnam.jpg"
pred_class, probs = predict_skin_condition(test_img, model, class_labels, threshold=0.1)

print("Predicted main class:", pred_class)
print("Probabilities (above threshold):", probs)