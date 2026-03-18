from app import app, Employee

with app.app_context():
    total = Employee.query.count()
    with_enc = Employee.query.filter(Employee.face_encoding != None).count()
    print('total', total)
    print('with encoding', with_enc)
