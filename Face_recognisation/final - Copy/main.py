import cv2
import face_recognition
import os
import csv
from datetime import datetime
import json

# Directories and files
TRAINING_DIR = "training_data"
USER_DATA_FILE = "user_data.json"

# Ensure directories and files exist
if not os.path.exists(TRAINING_DIR):
    os.makedirs(TRAINING_DIR)
if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, "w") as f:
        json.dump({}, f)

# Globals for known encodings and names
known_face_encodings = []
known_face_names = []
roll_number_map = {}

def load_training_data():
    """Load user data and face encodings."""
    global known_face_encodings, known_face_names, roll_number_map
    known_face_encodings.clear()
    known_face_names.clear()

    # Load user data from the JSON file
    with open(USER_DATA_FILE, "r") as f:
        user_data = json.load(f)
    roll_number_map = user_data

    for name, data in user_data.items():
        for i in range(50):  # here we assuming 50 images per user
            filepath = os.path.join(TRAINING_DIR, f"{name}_{i}.jpg")
            if os.path.exists(filepath):
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image)  # Get face encodings

                if encodings:  # Check if any encoding is found
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
               


def add_new_user():
    """Add a new user by capturing 50 photos."""
    user_name = input("Enter the name of the new user: ").strip()
    roll_no = input("Enter the roll number of the new user: ").strip()

    if not user_name or not roll_no:
        print("Name and roll number cannot be empty!")
        return

    # Check if the user already exists
    with open(USER_DATA_FILE, "r") as f:
        user_data = json.load(f)
    if user_name in user_data:
        print(f"User {user_name} already exists!")
        return

    # Initialize webcam and capture 50 photos
    cam = cv2.VideoCapture(0)
    print("Position yourself in front of the camera to start capturing images.")
    photo_count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to access the camera. Try again.")
            break

        # Detect face in the frame
        face_locations = face_recognition.face_locations(frame)

        for (top, right, bottom, left) in face_locations:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Crop the face
            face_image = frame[top:bottom, left:right]

            # Save the cropped face
            if photo_count < 50:
                filepath = os.path.join(TRAINING_DIR, f"{user_name}_{photo_count}.jpg")
                cv2.imwrite(filepath, face_image)
                print(f"Saved {filepath}")
                photo_count += 1

            if photo_count >= 50:
                print("Captured 50 photos successfully.")
                break

        # Display the frame with the bounding box
        cv2.imshow("Capture User Images", frame)

        # Press 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Image capture canceled.")
            break
        elif photo_count >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()

    # Save user data
    user_data[user_name] = {"roll_no": roll_no}
    with open(USER_DATA_FILE, "w") as f:
        json.dump(user_data, f)

    print(f"Training data updated with {user_name}.")
    load_training_data()  # Reload encodings for future attendance marking

def mark_attendance():
    """Mark attendance using face recognition."""
    global known_face_encodings, known_face_names, roll_number_map
    load_training_data()

    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    attendance_file = f"{current_date}.csv"

    # Open the attendance file
    with open(attendance_file, "w+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Roll Number", "Time"])

        # Initialize webcam
        cam = cv2.VideoCapture(0)
        faces_recognized = set()

        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to access the camera. Try again.")
                break

            # Detect faces in the frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                    if name not in faces_recognized:
                        faces_recognized.add(name)
                        roll_no = roll_number_map[name]["roll_no"]
                        current_time = now.strftime("%H:%M:%S")
                        writer.writerow([name, roll_no, current_time])
                        print(f"Marked attendance for {name} ({roll_no}) at {current_time}")

                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display the video feed
            cv2.imshow("Attendance", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        print("\nOptions:")
        print("1. Add a new user")
        print("2. Mark attendance")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            add_new_user()
        elif choice == "2":
            mark_attendance()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")
