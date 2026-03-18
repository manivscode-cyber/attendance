from app import app, Employee

with app.app_context():
    count = Employee.query.count()
    print(f"Total employees: {count}")

    for emp in Employee.query.limit(3).all():
        print(f"- {emp.emp_id}: {emp.name} (has face encoding={bool(emp.face_encoding)})")
