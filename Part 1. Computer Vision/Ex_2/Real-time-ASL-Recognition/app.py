import cv2
import mediapipe as mp
import numpy as np
import time
import os
import torch
from lstm.lstm_architecture import LSTMModel
from mlp.mlp_architecture import MLPModel
from seq2seq.encoder_architecture import Encoder
from seq2seq.decoder_architecture import Decoder
from seq2seq.predict import evaluate
from PIL import Image, ImageDraw, ImageFont

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LSTM_MODEL = LSTMModel()
LSTM_MODEL.load_state_dict(torch.load('lstm/lstm.pth', map_location=torch.device('cpu'), weights_only=False))
LSTM_MODEL.eval()  # Set to evaluation mode

MLP_MODEL = MLPModel()
MLP_MODEL.load_state_dict(torch.load('mlp/mlp.pth', map_location=torch.device('cpu'), weights_only=False))
MLP_MODEL.eval()  # Set to evaluation mode

ENCODER = Encoder()
ENCODER.load_state_dict(torch.load('seq2seq/encoder_final.pth', map_location=torch.device('cpu'), weights_only=False))

DECODER = Decoder()
DECODER.load_state_dict(torch.load('seq2seq/decoder_final.pth', map_location=torch.device('cpu'), weights_only=False))

# Define sign classes
WORDS_LABEL = ["bạn","tên", "tôi","có","ngồi", "xin chào","gặp", "nice", "làm ơn"] 

ALPHABETS_LABEL = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y']

MODE = 1  # 1 for LSTM, 0 for MLP

def put_vietnamese_text(img, text, position, font_scale=1, color=(255, 255, 255), thickness=2):
    """
    Function to properly display Vietnamese text using PIL and convert back to OpenCV
    """
    try:
        # Convert OpenCV image to PIL
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Try to load a font that supports Vietnamese characters
        try:
            # You may need to adjust the font path based on your system
            # Common Vietnamese-supporting fonts:
            font_paths = [
                "/System/Library/Fonts/Arial.ttf",  # macOS
                "C:/Windows/Fonts/arial.ttf",        # Windows
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                "/usr/share/fonts/TTF/arial.ttf"     # Some Linux distributions
            ]
            
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, size=int(30 * font_scale))
                    break
            
            if font is None:
                font = ImageFont.load_default()
                
        except:
            font = ImageFont.load_default()
        
        # Draw text
        draw.text(position, text, font=font, fill=color)
        
        # Convert back to OpenCV format
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_cv
        
    except Exception as e:
        print(f"Error in put_vietnamese_text: {e}")
        # Fallback to regular cv2.putText if PIL method fails
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        return img

def extract_landmarks(results):
    # Extract pose landmarks
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Extract hand landmarks
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Concatenate all landmarks into a feature vector
    if(MODE==1): return np.concatenate([pose, lh, rh]) #LSTM
    return np.concatenate([lh, rh]) #MLP

def draw_styled_landmarks(image, results):
    # # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

# For frame collection and sequence processing
SEQUENCE_LENGTH = 80  #FRAME_LENGTH
SEQUENCE = []
THRESHOLD = 0.2  # Confidence threshold

cap = cv2.VideoCapture(0)

# For FPS calculation
PREV_TIME = 0
CURR_TIME = 0

# Initialize variables for alphabet tracking
ALPHABET_TOKEN_DICT = {}  # Dictionary to track predicted alphabets
WORD_TOKEN_DICT = {}  # Dictionary to track predicted words
SENTENCE = ""    # Sentence to store results
LAST_DETECTED_TIME = time.time()  # Time when hands were last detected
HAND_PRESENT = False  # Flag to track if hands are present
SENTENCE_RESET_TIME = 0
SENTENCE_PROCESSED = False 

try:
    while cap.isOpened():
        #changing mode by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            MODE = 1 if MODE == 0 else 0
            print(f"Mode changed to: {'LSTM' if MODE == 1 else 'MLP'}")
            ALPHABET_TOKEN_DICT = {}
            WORD_TOKEN_DICT = {}
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        # Calculate FPS
        CURR_TIME = time.time()
        fps = 1 / (CURR_TIME - PREV_TIME) if PREV_TIME > 0 else 0
        PREV_TIME = CURR_TIME
        
        # Process the frame
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # Display FPS
        cv2.putText(image, f'FPS: {int(fps)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display mode
        cv2.putText(image, f'Mode: {"LSTM" if MODE == 1 else "MLP"}', (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Extract landmarks for prediction
        if results.left_hand_landmarks or results.right_hand_landmarks:
            frame_features = extract_landmarks(results)
            HAND_PRESENT = True
            LAST_DETECTED_TIME = time.time()
            word=""
            SENTENCE_PROCESSED = False 

            
            if MODE == 1:
                
                SEQUENCE.append(frame_features)
                
                # Keep only the most recent frames
                if len(SEQUENCE) > SEQUENCE_LENGTH:
                    SEQUENCE = SEQUENCE[-SEQUENCE_LENGTH:]
                    
                # If we have enough frames, make a prediction
                if len(SEQUENCE) == SEQUENCE_LENGTH:
                    # Prepare the input tensor - explicitly on CPU
                    input_tensor = torch.tensor(np.array([SEQUENCE]), dtype=torch.float32)
                    
                    # Get the model prediction
                    with torch.no_grad():
                        output = LSTM_MODEL(input_tensor)
                        output = torch.softmax(output, dim=1)  # Apply softmax to get probabilities
                        print(output)
                        
                        
                    # Get the predicted class and confidence
                    confidence, predicted_class_idx = torch.max(output, dim=1)
                    predicted_class = WORDS_LABEL[predicted_class_idx.item()]
                    
                    # Display prediction if confidence is above threshold
                    if confidence.item() > THRESHOLD:
                        #add the predicted class and count to the token dictionary
                        if predicted_class in WORD_TOKEN_DICT:
                            WORD_TOKEN_DICT[predicted_class] += 1
                        else:
                            WORD_TOKEN_DICT[predicted_class] = 1
                        
                        # Use Vietnamese text function for sign display
                        image = put_vietnamese_text(image, f'Sign: {predicted_class}', (10, 70), 
                                                  font_scale=1, color=(0, 255, 0))
                        cv2.putText(image, f'Confidence: {confidence.item():.2f}', (10, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        # Display a message
                        cv2.putText(image, "No sign detected", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
            else:
                token = {}
                input_tensor = torch.tensor(np.array([frame_features]), dtype=torch.float32)
                with torch.no_grad():
                    output = MLP_MODEL(input_tensor) 
                    output = torch.softmax(output, dim=1)  # Apply softmax to get probabilities
                    print(output)
                # Get the predicted class and confidence
                confidence, predicted_class_idx = torch.max(output, dim=1)
                predicted_class = ALPHABETS_LABEL[predicted_class_idx.item()]
                # Display prediction if confidence is above threshold
                if confidence.item() > THRESHOLD:
                    # Add the predicted class to the token dictionary
                    if predicted_class in ALPHABET_TOKEN_DICT:
                        ALPHABET_TOKEN_DICT[predicted_class] += 1
                    else:
                        ALPHABET_TOKEN_DICT[predicted_class] = 1
                    cv2.putText(image, f'Sign: {predicted_class}', (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f'Confidence: {confidence.item():.2f}', (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    # Display a message
                    cv2.putText(image, "No sign detected", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
        else:
            SEQUENCE=[]
            if HAND_PRESENT:  # If hands were previously detected but not anymore
                HAND_PRESENT = False
                # In MLP mode, add most frequent alphabet to sentence when hands disappear
                if MODE == 0 and ALPHABET_TOKEN_DICT:  # Only if we have collected some predictions
                    most_frequent_alphabet = max(ALPHABET_TOKEN_DICT, key=ALPHABET_TOKEN_DICT.get) if ALPHABET_TOKEN_DICT else ""
                    if most_frequent_alphabet:  # Only if we have collected some predictions
                        SENTENCE += most_frequent_alphabet + " "
                    ALPHABET_TOKEN_DICT = {}  # Reset the token dictionary
                else: 
                    most_frequent_word = max(WORD_TOKEN_DICT, key=WORD_TOKEN_DICT.get) if WORD_TOKEN_DICT else ""
                    if most_frequent_word:  # Only if we have collected some predictions
                        SENTENCE += most_frequent_word + " "
                    WORD_TOKEN_DICT = {}  # Reset the token dictionary
            
            # Check if no hand sign is detected for 5 seconds
            if time.time() - LAST_DETECTED_TIME > 6.5 and SENTENCE and not SENTENCE_PROCESSED:
                #SENTENCE = evaluate(ENCODER, DECODER, SENTENCE)
                SENTENCE_PROCESSED = True
                print(f"Final sentence: {SENTENCE}")
                SENTENCE_RESET_TIME = time.time() + 3.5

            if SENTENCE_RESET_TIME > 0 and time.time() >= SENTENCE_RESET_TIME:
                SENTENCE = ""
                SENTENCE_RESET_TIME = 0    
                
            
            # Display a message
            cv2.putText(image, "No hands detected", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Always display the current sentence using Vietnamese text function
        image = put_vietnamese_text(image, f'Sentence: {SENTENCE}', (20, 450), 
                                  font_scale=1, color=(255, 255, 0))
        
        cv2.imshow('Sign Language Recognition', image)
        # Exit on ESC key
        if cv2.waitKey(5) & 0xFF == 27:
            break
            
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    # Make sure to release resources
    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Destroy all windows
    holistic.close()  # Close the holistic model
    print("Application closed successfully")