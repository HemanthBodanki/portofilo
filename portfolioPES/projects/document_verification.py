import cv2
import pytesseract
import numpy as np
import face_recognition
import logging
from difflib import SequenceMatcher
from datetime import datetime

# -----------------------------
# CONFIGURATION
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

SIMILARITY_THRESHOLD = 0.7
FACE_MATCH_THRESHOLD = 0.6

# -----------------------------
# LOGGING SETUP
# -----------------------------
logging.basicConfig(
    filename="verification_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise removal
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return thresh


# -----------------------------
# OCR TEXT EXTRACTION
# -----------------------------
def extract_text(image_path):
    processed_img = preprocess_image(image_path)

    text = pytesseract.image_to_string(processed_img)

    return text.strip()


# -----------------------------
# TEXT SIMILARITY CHECK
# -----------------------------
def compare_texts(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()


# -----------------------------
# KEYWORD VALIDATION
# -----------------------------
def validate_keywords(text, required_fields):
    results = {}
    text_lower = text.lower()

    for field in required_fields:
        if field.lower() in text_lower:
            results[field] = True
        else:
            results[field] = False

    return results


# -----------------------------
# FACE MATCHING
# -----------------------------
def extract_face_encodings(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return None

    return encodings[0]


def compare_faces(img1_path, img2_path):
    enc1 = extract_face_encodings(img1_path)
    enc2 = extract_face_encodings(img2_path)

    if enc1 is None or enc2 is None:
        return False, 0.0

    distance = face_recognition.face_distance([enc1], enc2)[0]
    match = distance < FACE_MATCH_THRESHOLD

    confidence = 1 - distance
    return match, confidence


# -----------------------------
# FINAL VERIFICATION SYSTEM
# -----------------------------
def verify_document_system(doc1, doc2, face1=None, face2=None):
    print("🔍 Extracting text...")
    text1 = extract_text(doc1)
    text2 = extract_text(doc2)

    print("\n📄 Document 1 Text:\n", text1[:300])
    print("\n📄 Document 2 Text:\n", text2[:300])

    # Text similarity
    similarity = compare_texts(text1, text2)
    print(f"\n📊 Text Similarity Score: {similarity:.2f}")

    # Keyword validation
    required_fields = ["name", "date", "id", "dob"]
    validation_result = validate_keywords(text1, required_fields)

    print("\n🔑 Keyword Validation:")
    for k, v in validation_result.items():
        print(f"{k}: {'✅ Found' if v else '❌ Missing'}")

    # Face matching (optional)
    face_match = False
    face_confidence = 0

    if face1 and face2:
        print("\n🧠 Performing Face Matching...")
        face_match, face_confidence = compare_faces(face1, face2)
        print(f"Face Match: {'✅ Yes' if face_match else '❌ No'}")
        print(f"Confidence: {face_confidence:.2f}")

    # Final decision
    verified = (
        similarity >= SIMILARITY_THRESHOLD and
        all(validation_result.values()) and
        (face_match if face1 and face2 else True)
    )

    print("\n==============================")
    if verified:
        print("✅ DOCUMENT VERIFIED SUCCESSFULLY")
        status = "VERIFIED"
    else:
        print("❌ DOCUMENT VERIFICATION FAILED")
        status = "FAILED"
    print("==============================")

    # Logging
    logging.info(f"""
    STATUS: {status}
    Similarity: {similarity}
    Face Confidence: {face_confidence}
    Keywords: {validation_result}
    """)

    return verified


# -----------------------------
# MAIN FUNCTION
# -----------------------------
if __name__ == "__main__":
    doc1_path = "doc1.jpg"
    doc2_path = "doc2.jpg"

    # Optional face images
    face1_path = "face1.jpg"
    face2_path = "face2.jpg"

    try:
        verify_document_system(
            doc1_path,
            doc2_path,
            face1=face1_path,
            face2=face2_path
        )
    except Exception as e:
        print("⚠️ Error:", str(e))
        logging.error(str(e))
