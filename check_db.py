import sqlite3

conn = sqlite3.connect('attendance.db')
c = conn.cursor()

# Get tables
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [x[0] for x in c.fetchall()]
print(f"Tables: {tables}")

# Get employee columns
if 'employee' in tables:
    c.execute("PRAGMA table_info(employee)")
    columns = c.fetchall()
    print("\nEmployee columns:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")

# Check employee count
c.execute("SELECT COUNT(*) FROM employee")
count = c.fetchone()[0]
print(f"\nTotal employees: {count}")

# List employees
c.execute("SELECT emp_id, emp_name, face_encoding IS NOT NULL FROM employee")
employees = c.fetchall()
print("\nEmployees:")
for emp in employees:
    print(f"  - ID: {emp[0]}, Name: {emp[1]}, Has Face: {emp[2]}")

conn.close()
