import cv2
import os
import pytesseract
from ultralytics import YOLO
import re


# Function to extract frames from a video at a given interval
def extract_frames(video_path, output_folder, frame_interval=1):
    """Extract frames from a video at a given interval."""
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()
        if success and frame_count % (fps * frame_interval) == 0:
            frame_name = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_name, frame)
        frame_count += 1

    cap.release()
    print(f"Frames saved in: {output_folder}")


# Function to detect laptop screens and extract website content from a frame
def detect_screens_and_websites(frame_path, log_file, annotated_folder):
    """Detect screens (like TVs) and extract website text from the frame, zooming and enhancing detected regions."""
    # Load YOLO model (pretrained or custom-trained for screen detection)
    model = YOLO("yolov8n.pt")  # Use a lightweight YOLO model

    # Read the frame
    frame = cv2.imread(frame_path)
    
    # Detect objects (e.g., laptop screens, TVs, keyboards)
    results = model.predict(source=frame, save=False, conf=0.5)
    detections = results[0].boxes  # Bounding boxes and associated data
    detected_websites = []

    if detections:
        for detection in detections:
            bbox = detection.xyxy[0]  # Bounding box coordinates
            conf = detection.conf[0].item()  # Confidence score
            class_id = int(detection.cls[0].item())  # Object class ID

            # Debug: Print the detected class and confidence score
            print(f"Detected Class ID: {class_id}, Confidence: {conf:.2f}")

            # If the detected class is a TV (assuming class_id = 0 for TV, adjust based on your model)
            if class_id == 0:  # Modify this condition based on your class mappings
                x1, y1, x2, y2 = map(int, bbox[:4])  # Extract bounding box coordinates
                # Crop the detected area (TV region)
                cropped_region = frame[y1:y2, x1:x2]

                # Zoom in on the TV region
                zoom_factor = 2  # Increase the size by a factor of 2
                cropped_region = cv2.resize(cropped_region, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

                # Enhance the image without grayscaling:
                enhanced = cv2.convertScaleAbs(cropped_region, alpha=1.5, beta=50)  # Improve brightness and contrast
                blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)  # Reduce noise

                # Apply OCR to the preprocessed image
                detected_text = pytesseract.image_to_string(blurred)
                websites = find_websites_in_text(detected_text)
                detected_websites.extend(websites)

                # Annotate the frame with the detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, detected_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # Log the detection and extracted text
                with open(log_file, "a") as log:
                    log.write(f"Frame: {os.path.basename(frame_path)}, Detection: TV, Conf: {conf:.2f}, Text: {detected_text}\n")

                # Save the zoomed and preprocessed image for debugging
                zoomed_folder = "zoomed_frames"
                os.makedirs(zoomed_folder, exist_ok=True)
                zoomed_path = os.path.join(zoomed_folder, f"zoomed_{os.path.basename(frame_path)}")
                cv2.imwrite(zoomed_path, blurred)

    # Save the annotated frame
    os.makedirs(annotated_folder, exist_ok=True)
    annotated_path = os.path.join(annotated_folder, os.path.basename(frame_path))
    cv2.imwrite(annotated_path, frame)

    return detected_websites


# Function to find website links in text using regex
def find_websites_in_text(text):
    """Find website links in text using regex."""
    # Clean text: Remove symbols, line breaks, and excessive spaces
    cleaned_text = re.sub(r'[^\w\s\.-]', '', text).replace('\n', ' ').strip()
    
    # Regex to match domains like ".com", ".app", ".in", etc.
    website_pattern = r"(?:[a-zA-Z0-9_-]+\.)+[a-zA-Z]{2,}"
    
    return re.findall(website_pattern, cleaned_text)


# Main function to process the video
def process_video_for_websites(video_path):
    """Process a video to extract websites displayed on screens."""
    frame_folder = "temp_frames"
    annotated_folder = "annotated_frames"
    log_file = "detection_log.txt"

    # Clear previous log file
    if os.path.exists(log_file):
        os.remove(log_file)

    extract_frames(video_path, frame_folder, frame_interval=1)

    detected_websites = []
    for frame_file in os.listdir(frame_folder):
        frame_path = os.path.join(frame_folder, frame_file)
        websites = detect_screens_and_websites(frame_path, log_file, annotated_folder)
        detected_websites.extend(websites)

    # Remove duplicates and return results
    detected_websites = list(set(detected_websites))
    print(f"Detected Websites: {detected_websites}")
    print(f"Log saved to: {log_file}")
    print(f"Annotated frames saved to: {annotated_folder}")
    return detected_websites


# Run the script on a sample video
if __name__ == "__main__":
    video_file = "sample4.mp4"  # Replace with your video file
    detected_sites = process_video_for_websites(video_file)
    print("Final Detected Websites:", detected_sites)
