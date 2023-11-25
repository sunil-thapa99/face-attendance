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