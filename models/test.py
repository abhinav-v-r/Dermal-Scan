from tensorflow.keras.models import load_model

model_path = "/Dermal_Scan/models/dherma_ai_scan_v1.keras"
model = load_model(model_path)
def predict_skin_condition(img_path, model, class_labels, age):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

    # Prediction
    predictions = model.predict(img_array)[0]
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display
    plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%) Age: {age}") # Include age in title
    plt.show()

    return predicted_class, predictions
test_img = "/Dermal_Scan/sample_images/kid2.jpg"
pred_class, probs = predict_skin_condition(test_img, model, class_labels)
print("Class probabilities:", dict(zip(class_labels, probs)))