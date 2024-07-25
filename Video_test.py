!pip install yt-dlp
!pip install dlib
!pip install tensorflow
!pip install opencv-python
!pip install numpy

import cv2
import dlib
import numpy as np
import tensorflow as tf
from yt_dlp import YoutubeDL
from google.colab.patches import cv2_imshow

# Load the pre-trained sunglasses detection model
model_path = '/content/drive/MyDrive/sunglasses_detector_model.h5'
print("Loading model...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Initialize dlib's face detector
print("Initializing face detector...")
detector = dlib.get_frontal_face_detector()
print("Face detector initialized.")

# Define the function to preprocess the frames
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))  # Resize to the input shape of the model
    frame = frame / 255.0  # Normalize the frame
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Function to annotate the frame
def annotate_frame(frame):
    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Extract the face region
        face_img = frame[y:y+h, x:x+w]
        
        # Preprocess the face region
        preprocessed_face = preprocess_frame(face_img)
        
        # Predict using the model
        prediction = model.predict(preprocessed_face)[0][0]
        
        # Determine if sunglasses are detected
        label = 'Sunglasses' if prediction > 0.01 else 'No Sunglasses'
        
        # Draw bounding box and label
        color = (0, 255, 0) if label == 'Sunglasses' else (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return frame

# Function to process video from YouTube link using yt-dlp
def process_video_from_youtube(youtube_url):
    print("Fetching video information from YouTube...")
    # Define yt-dlp options
    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'noplaylist': True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        # Fetch video information
        info = ydl.extract_info(youtube_url, download=False)
        # Get the best video URL
        video_url = info['url']
    print("Video information fetched successfully.")

    print("Opening video stream...")
    # Open the video stream
    cap = cv2.VideoCapture(video_url)
    
    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object to save the video
    out = cv2.VideoWriter('/content/annotated_video.mp4', fourcc, fps, (width, height))
    print("Video stream opened successfully.")

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}...")
        
        # Annotate the frame
        annotated_frame = annotate_frame(frame)
        
        # Write the annotated frame to the output video
        out.write(annotated_frame)
        
        # Show the frame (optional)
        cv2_imshow(annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing interrupted by user.")
            break
    
    # Release the capture and close the window
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing completed. Annotated video saved.")

# URL of the YouTube video
youtube_url = 'https://youtu.be/QWCw4MN1ZjQ'

# Process the YouTube video
print("Starting video processing...")
process_video_from_youtube(youtube_url)
print("Video processing finished.")
