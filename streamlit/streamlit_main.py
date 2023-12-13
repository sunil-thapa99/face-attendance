import streamlit as st
import cv2
import requests

# URL = 'http://10.51.227.94:6006'
URL = 'http://127.0.0.1:6006'

def main():
    st.set_page_config(page_title="Face Attendance")
    st.title("Face Attendance")
    cap = cv2.VideoCapture(1)
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")
    if "recognized_response" not in st.session_state:
        st.session_state.recognized_response = ""
    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.write("Video Capture Ended")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        files = {'file': ('frame.jpg', frame_bytes, 'image/jpeg')}
        response = requests.post(f"{URL}/recognize", files=files)
        if response.status_code == 200:
            for res in response.json().get('response'):
                recognized_response = f"{res['id']} - {res['name']} [{res['time']}]"
                if recognized_response:
                    st.session_state.recognized_response = recognized_response
        else:
            st.session_state.recognized_response = "None"
        frame = cv2.putText(frame, st.session_state.recognized_response, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        frame_placeholder.image(frame,channels="RGB")
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()