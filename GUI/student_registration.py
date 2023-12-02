import tkinter as tk
from tkinter import messagebox
import cv2
import threading
import requests
from PIL import Image, ImageTk
import multiprocessing
from queue import Empty


class VideoCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Registration")

        # Set window size to the screen size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        output_file = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # Frames per second
        resolution = (640, 480)  # Resolution of the video
        self.record_duration = 5
        self.out = cv2.VideoWriter(output_file, fourcc, fps, resolution)

        self.video_writer_queue = multiprocessing.Queue()

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

    def stop_recording(self):
        # Disable recording flag
        self.is_recording = False

        # Wait for the video thread to finish
        if self.video_thread.is_alive():
            self.video_thread.join()

        # Release video capture object
        self.capture.release()

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

            # Convert the OpenCV frame to a format suitable for Tkinter
            cv2.imshow('Video Capture', frame)

            # Press 'q' to stop the video capture
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def exit_application(self):
        # Release video capture object
        self.capture.release()

        # Close the Tkinter window
        self.root.destroy()

    def submit_video(self):
        self.start_recording_button.pack_forget()
        self.start_registration_button.pack_forget()
        # Perform actions to submit the recorded video (add your logic here)
        # For example, you can save the video_frames list to a file or send it to a server.
        # Reset the video_frames list for the next registration if needed.
        messagebox.showinfo("Video Submitted", "Video submitted successfully!")
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoCaptureApp(root)
    root.mainloop()