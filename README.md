### Deep Learning Assignment

- Face Attendance System using Facenet
- Create a datasets folder 
- pip install -r requirements.txt
- python main.py

### WorkFlow API
#### Upload
```
/upload
Method [POST]: {
    name: "Sunil Thapa",
    id: "1",
    file: [file_type]
}
```

#### Dataprocess
```
/dataprocess
Method [GET]
```

#### Train
```
/train
Method [POST]
```

#### Recognition
```
/recognize
Method [POST]: {
    file: [file_type]
}
Response: {
    'id': '',
    'name': '',
    'time': '',
    'accuracy': ''
}
```

### WorkFlow
Run main.py from the root folder<br/>
Run app.py for database api from the GUI folder<br/>
Run student_registration.py from the GUI folder for registering the students video and information<br/>
Then, Run attendance_system.py to take attendance from the GUI folder.
