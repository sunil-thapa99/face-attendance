from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.db'
db = SQLAlchemy(app)


class Student(db.Model):
    name = db.Column(db.String(50), nullable=False)
    roll_number = db.Column(db.String(20), unique=True,
                            nullable=False, primary_key=True)
    attendance = db.Column(db.Integer, default=0)


@app.route('/students', methods=['GET'])
def get_students():
    students = Student.query.all()
    result = [{'name': student.name, 'roll_number': student.roll_number,
               'attendance': student.attendance} for student in students]
    return jsonify(result)


@app.route('/students/<int:student_roll_number>', methods=['GET'])
def get_student(student_roll_number):
    student = Student.query.get_or_404(student_roll_number)
    result = {'name': student.name,
              'roll_number': student.roll_number, 'attendance': student.attendance}
    return jsonify(result)


@app.route('/students', methods=['POST'])
def add_student():
    data = request.get_json()

    if 'name' not in data or 'roll_number' not in data:
        return jsonify({'error': 'Name and Roll Number are required'}), 400

    new_student = Student(name=data['name'], roll_number=data['roll_number'])
    db.session.add(new_student)
    db.session.commit()

    return jsonify({'message': 'Student added successfully'}), 201


@app.route('/students/<int:student_roll_number>', methods=['PUT'])
def update_attendance(student_roll_number):
    student = Student.query.get_or_404(student_roll_number)
    data = request.get_json()

    if 'attendance' not in data:
        return jsonify({'error': 'Attendance is required'}), 400

    student.attendance = data['attendance']
    db.session.commit()

    return jsonify({'message': 'Attendance updated successfully'})


@app.route('/students/<int:student_roll_number>', methods=['DELETE'])
def delete_student(student_roll_number):
    student = Student.query.get_or_404(student_roll_number)
    db.session.delete(student)
    db.session.commit()

    return jsonify({'message': 'Student deleted successfully'})


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
