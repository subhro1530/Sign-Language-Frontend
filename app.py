import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Load the trained model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Label mapping (Add more characters based on your model)
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
               19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Streamlit UI setup
st.title("Sign Language Detection Web App")
st.write("Use your camera to detect sign language gestures in real-time.")

# Option to use camera
use_camera = st.checkbox("Use Camera", value=True)

# Variables to hold the last prediction
prediction_frames = 0
last_prediction = ''

if use_camera:
    # Use OpenCV to capture from webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not access the camera.")
    else:
        frame_placeholder = st.empty()  # Placeholder for displaying video frames

        while True:
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = cap.read()

            if not ret:
                st.error("Error: Could not read frame from camera.")
                break

            H, W, _ = frame.shape

            # Convert the frame to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks on the frame
                    mp_drawing.draw_landmarks(
                        frame,  # image to draw on
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # Collect x and y coordinates of hand landmarks
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    # Normalize landmarks relative to the bounding box of the hand
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))  # Normalized x-coordinate
                        data_aux.append(y - min(y_))  # Normalized y-coordinate

                # Create a bounding box around the hand
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                # Ensure the correct number of features (84 in this case)
                if len(data_aux) == 84:
                    prediction = model.predict([np.asarray(data_aux)])  # Predict the class
                    predicted_character = labels_dict[int(prediction[0])]

                    # Store the prediction and reset the frame count
                    last_prediction = predicted_character
                    prediction_frames = 20  # Keep the prediction for the next 20 frames

            # Draw the prediction box with the predicted character
            if prediction_frames > 0:
                # Decrease the frame count
                prediction_frames -= 1

                # Draw the bounding box and display the last prediction
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
                cv2.rectangle(frame, (x1, y1 - 40), (x2, y1), (0, 255, 0), cv2.FILLED)  # Filled box for the text
                cv2.putText(frame, f'Predicted: {last_prediction}', (x1 + 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # White text

            # Display the frame in Streamlit
            frame_placeholder.image(frame, channels="BGR")

            # Break the loop when 'q' is pressed (only useful in local environments)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()  # Release the camera after loop ends
        cv2.destroyAllWindows()
