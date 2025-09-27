# label.py
import cv2

def draw_labels_on_image(image_np, age, features, face_cascade):
    """
    Draw bounding box and predicted labels on the image.

    Args:
        image_np (np.ndarray): Original input image (BGR).
        age (float): Predicted age.
        features (dict): Feature predictions, {feature_name: probability}.
        face_cascade (cv2.CascadeClassifier): Preloaded Haar cascade.

    Returns:
        np.ndarray: Annotated image with bounding box, age, and features.
    """
    # Add padding so labels don’t overlap image content
    top_pad, bottom_pad, left_pad, right_pad = 70, 150, 50, 50
    output_image = cv2.copyMakeBorder(
        image_np,
        top_pad, bottom_pad, left_pad, right_pad,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # Detect face on padded image
    gray_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("DEBUG: No face detected for labeling")
        return output_image

    # Take first detected face
    x, y, w, h = faces[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.6, 2

    # Colors
    box_color = (0, 255, 0)     #green  
    age_color = (255, 165, 0)  #orange
    feature_color = (255,0,0)   

    # Draw bounding box
    cv2.rectangle(
        output_image,
        (x, y),
        (x + w, y + h),
        box_color,
        thickness
    )

    # Draw age label
    age_text = f"Age: {age:.1f} years"
    cv2.putText(output_image, age_text, (x, y - 10), font, font_scale, age_color, thickness)

    # Draw feature labels
    start_y_features = y + h + 25
    for i, (feature_name, probability) in enumerate(features.items()):
        feature_text = f"- {feature_name}: {probability*100:.1f}%"
        current_y = start_y_features + (i * 25)
        cv2.putText(output_image, feature_text, (x, current_y), font, font_scale, feature_color, 1)

    return output_image
