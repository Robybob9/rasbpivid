import time
from flask import Flask, Response, render_template
import cv2
import face_recognition
import numpy as np
import requests
import os
import pandas as pd

app = Flask(__name__)

# Initialize the camera
camera = cv2.VideoCapture(0)  # Adjust index if needed (e.g., for USB camera)

def get_known_encodings():
    try:
        df = pd.read_parquet("known_persons.parquet")
        return df
    except:
        return None
    
def save_known_encodings():
    if os.path.exists("known_persons.parquet"):
        df = pd.read_parquet("known_persons.parquet")
    else:
        df = pd.DataFrame(columns=["128_encoding", "filename", "person_name"])
    folder_path = "templates\known"
    for im_known in os.listdir(folder_path):
        if im_known not in df["filename"].values.tolist():
            file_path = os.path.join(folder_path, im_known)
            known_image = cv2.imread(file_path)
            try:
                known_encodings = face_recognition.face_encodings(known_image, num_jitters=5, model="large")[0]
            except:
                continue
            df.loc[len(df.index)] = (known_encodings, im_known, im_known.split(".")[0])
    df.to_parquet("known_persons.parquet")
        

def generate_frames():
    while True:
        # Wait to not overwhelm the application with images
        time.sleep(0.4)
        success, frame = camera.read()
        if not success:
            break
        else:
            cv2.imwrite("templates\scene.jpg", frame)
            image = cv2.imread("templates\scene.jpg")
            start = time.time()
            face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1)
            end4 = time.time()
            print("Face recognitions duur: ",end4 - start)
            # Loop through each face found in the image and save it
            for i, (top, right, bottom, left) in enumerate(face_locations):
                # Extract the face using the bounding box
                face_image = image[top:bottom, left:right]

                # Save the face image
                output_path = f"templates\\face_{i+1}.jpg"
                cv2.imwrite(output_path, face_image)
                
            
            try:
                start2 = time.time()
                unknown_encodings = face_recognition.face_encodings(frame, known_face_locations=face_locations)
                known_persons = get_known_encodings()
                end2 = time.time()
                print("Face encodings duur: ",end2 - start2)
            except Exception as e:
                print(str(e))
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
            # Yield frame data in multipart format for video streaming
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                continue
            #print()
            if not unknown_encodings:
                print("Geen gezichten gevonden")
            for unknown_encoding in unknown_encodings:
                start3 = time.time()
                results = face_recognition.compare_faces(known_persons["128_encoding"].values.tolist(), unknown_encoding)
                end3 = time.time()
                print("Face compare duur: ",end3 - start3)
                print(results)
                if not any(results):
                    print("Geen gezicht gevonden")
                    continue
                for i in range(len(results)):
                    if results[i]:
                        name = known_persons["person_name"].values.tolist()[i]
                        print("Face recognised ", name)
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        # Definieer de tekst en de positie onder de rechthoek
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 1

                        # Bereken de grootte van de tekst om de positie nauwkeurig te bepalen
                        text_size, _ = cv2.getTextSize(name, font, font_scale, thickness)
                        text_x = left
                        text_y = bottom + text_size[1] + 5  # 5 pixels onder de rechthoek"""

                        # Schrijf de tekst onder de rechthoek
                        cv2.putText(frame, name, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

            
            # Zorg ervoor dat het frame correct wordt geconverteerd naar een JPEG
            end = time.time()
            print("Einde full cycle: ",end - start)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Yield frame data in multipart format for video streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Render a simple template
    save_known_encodings()
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Route to stream the video
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def send_message_to_telegram(message):
    # Replace with your bot's API token and group's chat ID
    bot_token = '7976460492:AAEzJJdzhHTTjOknKBAixY0vfZJHZZxn-Uk'
    chat_id = '-4610119384'

    # Telegram API URL
    url = f'https://api.telegram.org/bot{bot_token}/sendMessage'

    # Payload
    payload = {
        'chat_id': chat_id,
        'text': message
    }

    # Send the request
    response = requests.post(url, json=payload)

    # Check the response
    if response.status_code == 200:
        print("Message sent successfully!")
    else:
        print("Failed to send message:", response.text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
