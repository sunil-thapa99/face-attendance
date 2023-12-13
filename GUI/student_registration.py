import tkinter as tk
from tkinter import messagebox
import cv2
import threading
import requests
from PIL import Image, ImageTk
import multiprocessing
from queue import Empty
import requests
import json


class VideoCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Registration")

        # Set window size to the screen size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        self.output_file = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # Frames per second
        resolution = (640, 480)  # Resolution of the video
        self.record_duration = 5
        self.out = cv2.VideoWriter(self.output_file, fourcc, fps, resolution)

        # Loading label
        self.loading_label = tk.Label(
            self.root, text="Video Processing...", font=("Helvetica", 12), fg="blue")

        # Variables
        # Open default camera (change to 1, 2, etc., for additional cameras)
        self.capture = cv2.VideoCapture(0)
        self.is_recording = False
        self.student_name_var = tk.StringVar()
        self.student_id_var = tk.StringVar()
        self.video_frames = []  # Store the captured video frames

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        # Label and Entry for Student Name
        tk.Label(self.root, text="Student Name:").pack()
        tk.Entry(self.root, textvariable=self.student_name_var).pack()

        # Label and Entry for Student ID
        tk.Label(self.root, text="Student ID:").pack()
        tk.Entry(self.root, textvariable=self.student_id_var).pack()

        # Canvas to display video
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack()

        # # Create a label to display the video frame
        # self.video_label = tk.Label(self.root)
        # self.video_label.pack(pady=10)

        # Start Registration button
        self.start_registration_button = tk.Button(self.root, text="Start Registration",
                                                   command=self.start_registration)
        self.start_registration_button.pack(pady=10)

        # Start Recording button
        self.start_recording_button = tk.Button(
            self.root, text="Start Recording", command=self.start_recording)
        self.start_recording_button.pack(pady=10)
        self.start_recording_button.pack_forget()

        # # Save button
        # tk.Button(self.root, text="Save Video", command=self.save_video).pack(
        #     side=tk.LEFT, padx=10)

        # Exit button
        tk.Button(self.root, text="Exit",
                  command=self.exit_application).pack(side=tk.RIGHT, padx=10)

        # Submit Video button
        self.submit_video_button = tk.Button(
            self.root, text="Submit Video", command=self.submit_video)
        self.submit_video_button.pack(padx=10)
        self.submit_video_button.pack_forget()

        # Train Video button
        self.train_video_button = tk.Button(
            self.root, text="Register to the system", command=self.train_image)
        self.train_video_button.pack(padx=10)
        self.train_video_button.pack_forget()

        # Training Loading label
        self.training_loading_label = tk.Label(
            self.root, text="Registering with the system...", font=("Helvetica", 12), fg="blue")
        self.training_loading_label.pack(pady=10)
        self.training_loading_label.pack_forget()

    def start_registration(self):

        # Validate that both name and ID are entered
        if not self.student_name_var.get() or not self.student_id_var.get():
            messagebox.showerror(
                "Error", "Please enter both Student Name and Student ID.")
            return
        self.start_recording_button.pack()
        self.start_registration_button.pack_forget()
        # Schedule a timer to stop recording after the specified duration

    def start_recording(self):
        # self.stop_recording_button.pack()
        # Enable recording flag
        self.is_recording = True
        self.root.after(int(self.record_duration * 1000), self.stop_recording)
        # Start video capture in a separate thread
        self.video_thread = threading.Thread(target=self.capture_video_thread)
        self.video_thread.start()

        # Update the Tkinter window with the video frames
        self.update_video()

    def stop_recording(self):
        print("Inside stop recording")
        # Disable recording flag
        self.is_recording = False

        # # Wait for the video thread to finish
        if self.video_thread.is_alive():
            self.video_thread.join()

        self.release_video()
        self.out.release()

        # # self.video_thread._stop()
        # self.video_label.pack_forget()

        print("after joining thread and before releasing capture")

        # Display recording stopped message and close the application
        messagebox.showinfo("Recording Stopped",
                            "Video Recorded successfully!")
        # Show submit video button
        self.submit_video_button.pack()
        self.start_registration_button.pack_forget()

    def capture_video_thread(self):
        while self.is_recording:
            ret, frame = self.capture.read()

            if not ret or frame is None or frame.size == 0:
                print("Error: Could not capture frame.")
                break

            self.out.write(frame)

            # # Convert the OpenCV frame to a format suitable for Tkinter
            cv2.imshow('Video Capture', frame)

            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # image = Image.fromarray(image)
            # photo = ImageTk.PhotoImage(image=image)

            # Update the label with the new image
            # self.video_label.config(image=photo)
            # self.video_label.image = photo

            # Press 'q' to stop the video capture
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def release_video(self):
        if hasattr(self, 'capture'):
            self.capture.release()

    def update_video(self):
        # Check if the recording is still in progress
        if self.is_recording and self.video_thread.is_alive():
            # Call the update_video method after a delay (in milliseconds)
            self.root.after(10, self.update_video())
        else:
            # Recording has stopped, do cleanup or any other necessary actions
            self.capture.release()

    def exit_application(self):
        # Release video capture object
        self.capture.release()

        # Close the Tkinter window
        self.root.destroy()

    def submit_video(self):
        # Release video capture object
        self.capture.release()
        self.out.release()

        self.start_recording_button.pack_forget()
        self.start_registration_button.pack_forget()
        cv2.waitKey(5)
        # Perform actions to submit the recorded video (add your logic here)
        # For example, you can save the video_frames list to a file or send it to a server.
        # Reset the video_frames list for the next registration if needed.
        # Send the video file along with student name and ID to the API\

        url = 'http://127.0.0.1:6006/upload'
        # url = 'http://10.51.227.94:6006/upload'
        student_name = self.student_name_var.get()
        student_id = self.student_id_var.get()
        print("student name:", student_name)
        print("student id:", student_id)

        self.loading_label.pack(pady=10)

        # Reopen the video file for sending
        with open('output_video.mp4', 'rb') as file:
            files = {'file': ('output_video.mp4', file)}
            data = {'name': student_name, 'id': student_id}

            response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            messagebox.showinfo("Video Submitted",
                                "Video submitted successfully!")
            response1 = requests.get('http://127.0.0.1:6006/dataprocess')
            messagebox.showinfo("Video Submitted",
                                "Video processed successfully!")
            self.loading_label.pack_forget()
            self.train_video_button.pack()

        else:
            self.submit_video()

        # # Check the API response
        # print(response.status_code, response.text)
        # messagebox.showinfo("Video Submitted", "Video submitted successfully!")
    def train_image(self):
        self.training_loading_label.pack()
        response1 = requests.post('http://127.0.0.1:6006/train')

        # Adding to the database
        url = 'http://127.0.0.1:5000/students'
        student_name = self.student_name_var.get()
        student_id = self.student_id_var.get()
        data = {'name': student_name, 'roll_number': student_id}
        data = json.dumps(data)
        headers = {"Content-Type": "application/json"}

        response = requests.post(url=url, data=data, headers=headers)

        messagebox.showinfo(
            "Model Trained", "Student Registered successfully!")
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoCaptureApp(root)
    root.mainloop()
