import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to the folder with training images
path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

# Load each image and extract the class name from the file name
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Function to find encodings for the images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Function to mark attendance
def markAttendance(name, marked_names):
    if name in marked_names:
        return False  # User already marked attendance
    # Append the new entry
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')
    with open('Attendance.csv', 'a') as f:  # Append name and time
        f.writelines(f'\n{name},{dtString}')
    marked_names.add(name)  # Add name to the set
    return True  # Attendance marked

# Encode the known faces
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Set to keep track of marked names and shown messages
marked_names = set()
shown_messages = set()

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Reduce image size for faster processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces in the current frame and encode them
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Compare detected faces with known faces
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Rescale to original size

            # Draw rectangle and put the name of the recognized face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Mark attendance if not already marked
            if markAttendance(name, marked_names):
                print(f'Attendance marked for {name}')
            else:
                if name not in shown_messages:
                    print(f'{name} has already marked attendance.')
                    shown_messages.add(name)  # Add to shown messages set

    # Display the webcam feed
    cv2.imshow('Webcam', img)

    # If 'q' is pressed, break the loop and close the webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
