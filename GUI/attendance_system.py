import tkinter as tk
from tkinter import ttk
import cv2
import requests
from PIL import Image, ImageTk


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance System")

        self.start_button = tk.Button(
            root, text="Start Attendance", command=self.start_attendance)
        self.start_button.pack(pady=10)

        self.video_frame = tk.Label(root)
        self.video_frame.pack(side=tk.LEFT, padx=10)

        self.attendance_list = ttk.Treeview(root, columns=(
            "Name", "Roll Number", "Attendance"), show="headings")
        self.attendance_list.heading("Name", text="Name")
        self.attendance_list.heading("Roll Number", text="Roll Number")
        self.attendance_list.heading("Attendance", text="Attendance")
        self.attendance_list.pack(side=tk.RIGHT, padx=10)

        # self.api_url = "http://10.51.227.94:6006/recognize"
        self.api_url = "http://127.0.0.1:6006/recognize"
        self.attendance_api_url = "http://127.0.0.1:5000/students"

        # Set to keep track of recognized students
        self.recognized_students = set()

        # For Testing Only
        # ------------------------------------------------------
        self.count = 0
        self.recognition_data = {
            "accuracy": "[0.76876688]",
            "id": "5367",
            "name": "sushil basi",
            "time": "13:15:31"}

    def start_attendance(self):
        # Destroy start button
        self.start_button.destroy()

        # Create video capture
        self.cap = cv2.VideoCapture(0)

        # Update video frame
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()

        # For Testing Only
        # ------------------------------------------------------
        self.count += 1
        if self.count % 1000 == 0:
            self.recognition_data = {
                "accuracy": "[0.76876688]",
                "id": "123",
                "name": "Samir Khanal",
                "time": "13:15:31"}
        # --------------------------------------------------

        if ret:
            # Convert the OpenCV frame to a JPEG image
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_bytes = img_encoded.tobytes()

            # # Send the image to the recognition API
            response = requests.post(self.api_url, files={'file': ('image.jpg', img_bytes)})

            if response.status_code == 200:
                # Process the response data only if the student has not been recognized before
                recognition_data = response.json()
                name = recognition_data.get('name', 'Unknown')
                if name not in self.recognized_students:
                    self.process_recognition_data(recognition_data)

            # # For Testing Only
            # # ------------------------------------------------------

            # name = self.recognition_data['name']
            # if name not in self.recognized_students:
            #     self.process_recognition_data(self.recognition_data)

            # # --------------------------------------------------

            # Convert the OpenCV frame to a PIL image
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            # Update video frame
            self.video_frame.configure(image=img)
            self.video_frame.image = img

            # Call update_video after 10 milliseconds
            self.video_frame.after(10, self.update_video)

    def process_recognition_data(self, recognition_data):
        # Extract relevant information from the recognition response and update the attendance list
        name = recognition_data.get('name', 'Unknown')
        roll_number = recognition_data.get('roll_number', 'N/A')
        status = recognition_data.get('status', 'Unknown')

        # # For Test only
        # # --------------------------------------------
        # name = recognition_data['name']
        # roll_number = recognition_data["id"]
        # attendance = self.get_attendance(roll_number)
        # # ----------------------------------------------

        # Update the attendance list
        self.attendance_list.insert(
            "", "end", values=(name, roll_number, attendance))

        # Add the recognized student to the set
        self.recognized_students.add(name)

        # Call the attendance API (replace this with your logic)
        self.call_attendance_api(name, roll_number)

    def get_attendance(self, roll_number):
        data = {"roll_number": roll_number}
        url = self.attendance_api_url+'/getattendance'
        response = requests.get(url=url, json=data)
        if response.status_code == 200:
            data = response.json()
            return data['attendance']
        else:
            return 1

    def call_attendance_api(self, name, roll_number):
        # Call your attendance API with the relevant data
        attendance_data = {
            "name": name,
            "roll_number": roll_number,
            "attendance": 1
        }
        url = self.attendance_api_url + '/attend'
        response = requests.post(url=url, json=attendance_data)
        # Handle the response from the attendance API as needed
        data = response.text
        print("update attendance data:", data)


if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()
