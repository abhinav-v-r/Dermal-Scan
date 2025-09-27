import cv2
import os

def load_cascade(cascade_filename="haarcascade_frontalface_default.xml"):
    """
    Loads a Haar Cascade classifier from a file.

    Args:
        cascade_filename (str): Name of the cascade XML file, assumed
                                to be in the same directory as this script.

    Returns:
        cv2.CascadeClassifier or None: The loaded classifier object if successful,
                                       otherwise None.
    """
    # Build absolute path to the cascade file
    cascade_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cascade_filename)

    # Check if the file exists
    if not os.path.exists(cascade_path):
        print(f"ERROR: Cascade file not found at '{cascade_path}'")
        return None

    # Load the classifier
    cascade_classifier = cv2.CascadeClassifier(cascade_path)

    # Verify that the classifier loaded correctly
    if cascade_classifier.empty():
        print(f"ERROR: Could not load cascade classifier from '{cascade_path}'. The file may be corrupt.")
        return None

    print(f"Haar Cascade classifier '{cascade_filename}' loaded successfully.")
    return cascade_classifier


# --- Example usage for testing ---
if __name__ == '__main__':
    print("Testing the cascade loader...")

    face_detector = load_cascade()

    if face_detector is not None:
        print("Cascade loaded successfully and is ready for use.")
    else:
        print("Cascade loading failed. Please review the error messages above.")