# label.py
import cv2

def get_age_range(age):
    """
    Determine the age range category based on predicted age.
    
    Args:
        age (float): Predicted age.
        
    Returns:
        str: Age range category.
    """
    if age < 3:
        return "0-2"
    elif age < 9:
        return "3-8"
    elif age < 13:
        return "9-12"
    elif age < 18:
        return "13-17"
    elif age < 30:
        return "18-29"
    elif age < 45:
        return "30-44"
    elif age < 60:
        return "45-59"
    else:
        return "60+"

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
        print("DEBUG: No face detected")
        return output_image

    # Take first detected face
    x, y, w, h = faces[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.6, 2

    # Colors
    box_color = (0, 255, 0)     #green  
    age_color = (255, 255, 255)  #orange
    feature_color = (255, 255, 0)

    # Draw bounding box
    cv2.rectangle(
        output_image,
        (x, y),
        (x + w, y + h),
        box_color,
        thickness
    )

    # Draw age label
    age_range = get_age_range(age)
    age_text = f"Age: {age:.1f} years ({age_range})"
    cv2.putText(output_image, age_text, (x, y - 10), font, font_scale, age_color, thickness)

    # Draw feature labels
    start_y_features = y + h + 25
    feature_count = 0
    for i, (feature_name, probability) in enumerate(features.items()):
        # Only show features with probability >= 5%
        if probability * 100 >= 5.0:
            feature_text = f"- {feature_name}: {probability*100:.1f}%"
            current_y = start_y_features + (feature_count * 25)
            cv2.putText(output_image, feature_text, (x, current_y), font, font_scale, feature_color, 1)
            feature_count += 1

    return output_image

def get_home_remedies(condition):
    """
    Provide home remedies for common skin conditions.
    
    Args:
        condition (str): Skin condition name.
        
    Returns:
        str: Home remedies for the specified condition.
    """
    remedies = {
        "wrinkles": "**Home Remedies for Wrinkles:**\n- Apply aloe vera gel daily\n- Use coconut oil as a moisturizer\n- Apply vitamin E oil before bed\n- Stay hydrated and use sunscreen\n- Try facial exercises",
        
        "dark_spots": "**Home Remedies for Dark Spots:**\n- Apply lemon juice (diluted) as a natural bleach\n- Use aloe vera gel twice daily\n- Apply apple cider vinegar with equal parts water\n- Try potato slices or juice on affected areas\n- Use turmeric and honey mask",
        
        "puffy_eyes": "**Home Remedies for Puffy Eyes:**\n- Apply cold cucumber slices or tea bags\n- Use cold spoons on eyes for 5-10 minutes\n- Apply potato slices to reduce swelling\n- Get adequate sleep and reduce salt intake\n- Stay hydrated and elevate head while sleeping",
        
        "clear_face": "**Tips to Maintain Clear Skin:**\n- Follow a consistent cleansing routine\n- Stay hydrated with at least 8 glasses of water daily\n- Use non-comedogenic products\n- Exfoliate 1-2 times weekly\n- Protect skin with SPF 30+ sunscreen"
    }
    
    return remedies.get(condition, "No specific remedies available for this condition.")
