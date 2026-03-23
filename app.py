
# GoDigital Attendance System - Connected Backend
# Run with: python app.py
# Visit: http://localhost:5000

import os
import json
import math
import pickle
import base64
import subprocess
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pytz
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, Blueprint
from flask_sqlalchemy import SQLAlchemy
from PIL import Image, ImageOps
from werkzeug.security import generate_password_hash

IST = pytz.timezone('Asia/Kolkata')
BASE_DIR = Path(__file__).resolve().parent
FACE_WORKER_SCRIPT = BASE_DIR / 'face_recognition_worker.py'
LOCAL_VENV_PYTHON_CANDIDATES = (
    BASE_DIR / '.venv' / 'Scripts' / 'python.exe',
    BASE_DIR / '.venv' / 'bin' / 'python',
)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config.from_object('config')
app.secret_key = "super-secret-key-change-this-in-production-2026"

# Initialize SQLAlchemy
db = SQLAlchemy(app)


@app.after_request
def add_no_cache_headers(response):
    """Prevent browsers from caching the scan page or API responses during active development."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# Define models before importing other modules


class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    emp_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100))
    department = db.Column(db.String(100))
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    face_encoding = db.Column(db.LargeBinary)  # Pickled numpy array for face encoding
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(IST))


class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False)
    employee = db.relationship('Employee', backref=db.backref('attendance', lazy=True))
    date = db.Column(db.Date, default=lambda: datetime.now(IST).date())
    check_in = db.Column(db.String(10), nullable=True)  # Store as "HH:MM" format
    check_out = db.Column(db.String(10), nullable=True)  # Store as "HH:MM" format
    status = db.Column(db.String(20), default='Pending')  # 'Present', 'Late', 'Absent'
    attendance_type = db.Column(db.String(20))  # 'WFH', 'Field', 'Onsite'
    department = db.Column(db.String(100))
    total_hours = db.Column(db.Float, default=0.0)


# ────────────────────────────────────────────────
#   DATA MANAGEMENT FUNCTIONS
# ────────────────────────────────────────────────


def init_default_employees():
    """Start with an empty employee table for first-time enrollment."""
    return


# Initialize database on app startup


@app.before_request
def init_db():
    """Initialize database tables on first request"""
    if not hasattr(app, 'db_initialized'):
        db.create_all()
        init_default_employees()
        app.db_initialized = True


@app.route('/')
def home():
    """Open home page first. If already logged in, go to dashboard."""
    if session.get('emp_id'):
        return redirect(url_for('attendance.dashboard'))
    return render_template('home.html', ist_now=current_ist_datetime_str())


# ────────────────────────────────────────────────
#   ATTENDANCE HELPER FUNCTIONS
# ────────────────────────────────────────────────


def get_attendance_status(check_in_time_str):
    """
    Determine if attendance is on-time or late
    On-time: 10:00 to 10:15
    Late: 10:16 onwards
    """
    try:
        check_in_hour, check_in_minute = map(int, check_in_time_str.split(':'))

        # Convert to minutes for easier comparison
        check_in_minutes = check_in_hour * 60 + check_in_minute
        on_time_limit = 10 * 60 + 15  # 10:15 in minutes

        if check_in_minutes <= on_time_limit:
            return "On-Time"
        else:
            return "Late"
    except (AttributeError, TypeError, ValueError):
        return "Unknown"


def calculate_total_hours(check_in_str, check_out_str):
    """Calculate total hours worked between check-in and check-out"""
    try:
        check_in = datetime.strptime(check_in_str, '%H:%M:%S')
        check_out = datetime.strptime(check_out_str, '%H:%M:%S')

        duration = check_out - check_in
        total_seconds = duration.total_seconds()
        total_hours = total_seconds / 3600

        return round(total_hours, 2)
    except (TypeError, ValueError):
        return 0.0


def now_ist():
    """Return the current Asia/Kolkata aware datetime."""
    return datetime.now(IST)


def current_ist_date():
    """Return the current date in Asia/Kolkata."""
    return now_ist().date()


def current_ist_time_str():
    """Return the current IST time in HH:MM format."""
    return now_ist().strftime('%H:%M')


def current_ist_datetime_str():
    """Return a friendly current IST datetime string."""
    return now_ist().strftime('%d %b %Y, %I:%M %p IST')


def serialize_attendance_record(record):
    """Return attendance details that the face-attendance UI can show directly."""
    if not record:
        return None

    return {
        'date': record.date.isoformat() if record.date else None,
        'check_in': record.check_in,
        'check_out': record.check_out,
        'status': record.status,
        'attendance_type': record.attendance_type,
        'department': record.department,
        'total_hours': record.total_hours,
    }


def get_face_worker_python():
    """Return the local venv Python used for face recognition, if available."""
    for candidate in LOCAL_VENV_PYTHON_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def get_face_restart_command():
    """Return the recommended command to start Flask with the project venv."""
    worker_python = get_face_worker_python()
    if worker_python and worker_python.name.lower() == 'python.exe':
        return r'.\.venv\Scripts\python.exe app.py'
    if worker_python:
        return './.venv/bin/python app.py'
    return None


def get_face_dependency_message():
    """Return a clear next step when the active interpreter lacks face dependencies."""
    active_python = Path(sys.executable).name or sys.executable
    restart_command = get_face_restart_command()
    if restart_command:
        return (
            f'Face recognition is not available in the active Python ({active_python}). '
            f'Restart Flask with `{restart_command}`.'
        )
    return (
        'Face recognition dependencies are missing. Install `face-recognition` '
        'and `face-recognition-models`, then restart Flask.'
    )


def load_native_face_modules():
    """Load native face-recognition modules when available in this interpreter."""
    try:
        import face_recognition
        import face_recognition_models  # noqa: F401
        import numpy as np
    except ImportError:
        return None, None
    return np, face_recognition


def normalize_face_encoding(encoding):
    """Convert a stored face encoding into a plain list of floats."""
    if encoding is None:
        return None

    if hasattr(encoding, 'tolist'):
        encoding = encoding.tolist()

    try:
        return [float(value) for value in encoding]
    except (TypeError, ValueError):
        return None


def calculate_face_distance(known_encoding, candidate_encoding):
    """Return Euclidean distance between two face encodings."""
    known_values = normalize_face_encoding(known_encoding)
    candidate_values = normalize_face_encoding(candidate_encoding)

    if not known_values or not candidate_values:
        return None

    if len(known_values) != len(candidate_values):
        return None

    squared_sum = sum(
        (known_value - candidate_value) ** 2
        for known_value, candidate_value in zip(known_values, candidate_values)
    )
    return math.sqrt(squared_sum)


def run_face_worker_extract(image_data):
    """Use the project venv to extract face data when this interpreter cannot."""
    worker_python = get_face_worker_python()
    if worker_python is None or not FACE_WORKER_SCRIPT.exists():
        return {
            'ok': False,
            'code': 'dependency_missing',
            'message': get_face_dependency_message()
        }

    payload = json.dumps({
        'action': 'extract_face',
        'image_data': image_data,
    })

    try:
        result = subprocess.run(
            [str(worker_python), str(FACE_WORKER_SCRIPT)],
            input=payload,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            timeout=45,
            check=False
        )
    except Exception as exc:
        print(f"Face worker launch error: {exc}")
        return {
            'ok': False,
            'code': 'dependency_missing',
            'message': get_face_dependency_message()
        }

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if stderr:
            print(f"Face worker error: {stderr}")
        return {
            'ok': False,
            'code': 'dependency_missing',
            'message': get_face_dependency_message()
        }

    try:
        face_result = json.loads(result.stdout or '{}')
    except json.JSONDecodeError as exc:
        print(f"Face worker response error: {exc}")
        return {
            'ok': False,
            'code': 'processing_error',
            'message': 'Face processing returned an invalid response.'
        }

    if not isinstance(face_result, dict):
        return {
            'ok': False,
            'code': 'processing_error',
            'message': 'Face processing returned an invalid response.'
        }

    if face_result.get('ok'):
        face_result['encoding'] = normalize_face_encoding(face_result.get('encoding'))
        location = face_result.get('location') or []
        face_result['location'] = tuple(int(value) for value in location)

    return face_result


def decode_image_from_base64(image_data):
    """Decode incoming base64 image data and normalize orientation for mobile uploads."""
    try:
        if ',' in image_data:
            _, encoded = image_data.split(',', 1)
        else:
            encoded = image_data

        img_bytes = base64.b64decode(encoded)
        pil_image = Image.open(BytesIO(img_bytes))
        pil_image = ImageOps.exif_transpose(pil_image).convert('RGB')
        return pil_image, img_bytes
    except Exception as exc:
        print(f"Image decode error: {exc}")
        return None, None


def extract_face_data_from_base64(image_data):
    """Validate an uploaded image and extract a single reliable face encoding."""
    if not image_data:
        return {'ok': False, 'code': 'missing_image', 'message': 'No image provided'}

    native_numpy, native_face_recognition = load_native_face_modules()
    if native_face_recognition is None:
        return run_face_worker_extract(image_data)

    try:
        pil_image, _ = decode_image_from_base64(image_data)
        if pil_image is None:
            return {'ok': False, 'code': 'invalid_image', 'message': 'Invalid image data'}

        rgb_img = native_numpy.array(pil_image)
        if rgb_img.size == 0:
            return {'ok': False, 'code': 'invalid_image', 'message': 'Invalid image data'}

        scale = 1.0
        max_width = 960
        if pil_image.width > max_width:
            scale = max_width / float(pil_image.width)
            resized_height = int(pil_image.height * scale)
            resized = pil_image.resize((max_width, resized_height))
            processing_img = native_numpy.array(resized)
        else:
            processing_img = rgb_img

        def locate_faces(image, upsample=1):
            return native_face_recognition.face_locations(
                image,
                number_of_times_to_upsample=upsample,
                model='hog'
            )

        face_locations = locate_faces(processing_img, upsample=1)
        if len(face_locations) == 0:
            face_locations = locate_faces(processing_img, upsample=2)
        if len(face_locations) == 0:
            return {'ok': False, 'code': 'no_face', 'message': 'No face detected. Move closer, face the camera, and use better lighting.'}
        if len(face_locations) > 1:
            return {'ok': False, 'code': 'multiple_faces', 'message': 'Multiple faces detected. Please scan only one face.'}

        original_face_locations = face_locations
        if scale != 1.0:
            original_face_locations = [
                (
                    max(0, min(rgb_img.shape[0], int(top / scale))),
                    max(0, min(rgb_img.shape[1], int(right / scale))),
                    max(0, min(rgb_img.shape[0], int(bottom / scale))),
                    max(0, min(rgb_img.shape[1], int(left / scale)))
                )
                for top, right, bottom, left in face_locations
            ]

        encodings = native_face_recognition.face_encodings(processing_img, face_locations)
        if len(encodings) != 1 and processing_img is not rgb_img:
            encodings = native_face_recognition.face_encodings(
                rgb_img,
                original_face_locations
            )
        if len(encodings) != 1:
            # Retry once on each image without supplied locations.
            encodings = native_face_recognition.face_encodings(processing_img)
        if len(encodings) != 1 and processing_img is not rgb_img:
            encodings = native_face_recognition.face_encodings(rgb_img)
        if len(encodings) != 1:
            return {
                'ok': False,
                'code': 'invalid_face',
                'message': 'Please retry with one clear face in the frame.'
            }

        return {
            'ok': True,
            'encoding': encodings[0],
            'location': original_face_locations[0],
            'landmarks': None
        }
    except Exception as e:
        print(f"Face extraction error: {e}")
        return {
            'ok': False,
            'code': 'processing_error',
            'message': 'Please retry with one clear face in the frame.'
        }


def check_face_image(image_data):
    """Run a lightweight face validation pass before full recognition/enrollment."""
    face_result = extract_face_data_from_base64(image_data)
    if not face_result.get('ok'):
        return face_result

    top, right, bottom, left = face_result['location']
    face_height = bottom - top
    face_width = right - left

    if face_height < 70 or face_width < 70:
        return {
            'ok': False,
            'code': 'face_too_small',
            'message': 'Move closer so your face fills more of the frame.'
        }

    return {
        'ok': True,
        'code': 'face_ready',
        'message': 'Face detected clearly.',
        'location': face_result['location']
    }


def get_face_encoding_from_base64(image_data):
    """Backward-compatible helper that returns only the encoding or None."""
    result = extract_face_data_from_base64(image_data)
    if result.get('ok'):
        return result['encoding']
    return None


def is_same_face(known_encoding, candidate_encoding, tolerance=0.5):
    """Return True when two encodings represent the same person."""
    distance = calculate_face_distance(known_encoding, candidate_encoding)
    return distance is not None and distance <= tolerance


def save_face_backup(emp_id, image_data):
    """Save the captured face image as a local backup."""
    if not image_data:
        return

    os.makedirs('data/faces', exist_ok=True)
    _, img_bytes = decode_image_from_base64(image_data)
    if not img_bytes:
        return
    with open(f'data/faces/{emp_id}_face.jpg', 'wb') as f:
        f.write(img_bytes)


def set_employee_session(employee):
    """Create an application session for a recognized employee."""
    session['emp_id'] = employee.emp_id
    session['emp_name'] = employee.name
    session['department'] = employee.department
    session['is_admin'] = employee.is_admin


def mark_face_attendance(employee, department=None, attendance_type='Onsite'):
    """Mark check-in or check-out for a recognized employee in IST."""
    today = current_ist_date()
    current_time_str = current_ist_time_str()
    normalized_department = (department or employee.department or 'General').strip() or 'General'

    today_attendance = Attendance.query.filter(
        Attendance.employee_id == employee.id,
        Attendance.date == today
    ).first()

    if today_attendance:
        if today_attendance.check_out:
            return {
                'success': False,
                'message': 'Attendance already completed for today.',
                'action': 'duplicate',
                'attendance': today_attendance
            }

        check_in_time = datetime.strptime(today_attendance.check_in, '%H:%M')
        check_out_time = datetime.strptime(current_time_str, '%H:%M')

        if check_out_time < check_in_time:
            check_out_time = check_out_time.replace(day=check_out_time.day + 1)

        total_hours = round((check_out_time - check_in_time).total_seconds() / 3600, 2)
        today_attendance.check_out = current_time_str
        today_attendance.total_hours = total_hours
        today_attendance.department = normalized_department
        today_attendance.attendance_type = attendance_type
        employee.department = normalized_department
        db.session.commit()

        return {
            'success': True,
            'message': f'Check-out marked successfully. Total hours: {total_hours:.2f}',
            'action': 'check_out',
            'attendance': today_attendance,
            'next_action': 'Attendance completed for today'
        }

    status = get_attendance_status(current_time_str)
    employee.department = normalized_department
    new_attendance = Attendance(
        employee_id=employee.id,
        date=today,
        check_in=current_time_str,
        status=status,
        attendance_type=attendance_type,
        department=normalized_department
    )
    db.session.add(new_attendance)
    db.session.commit()

    return {
        'success': True,
        'message': f'Check-in marked successfully. Status: {status}',
        'action': 'check_in',
        'attendance': new_attendance,
        'next_action': 'Scan again later to check out'
    }


def verify_face_recognition(face_image_data):
    """DEPRECATED placeholder - use get_face_encoding_from_base64() instead."""
    return bool(face_image_data)


# ────────────────────────────────────────────────
#   DECORATORS
# ────────────────────────────────────────────────


def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('emp_id') is None:
            flash('Please log in first', 'error')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Decorator to require admin access"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('emp_id') is None:
            flash('Please log in first', 'error')
            return redirect(url_for('auth.login'))

        emp = Employee.query.filter_by(emp_id=session.get('emp_id')).first()

        if not emp or not emp.is_admin:
            flash('Admin access required', 'error')
            return redirect(url_for('attendance.dashboard'))

        return f(*args, **kwargs)
    return decorated_function


# ────────────────────────────────────────────────
#   AUTH BLUEPRINT
# ────────────────────────────────────────────────

auth_bp = Blueprint('auth', __name__, url_prefix='')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        emp_id = request.form.get('emp_id', '').upper().strip()
        name = request.form.get('name', '').strip()

        if not emp_id or not name:
            flash('Please provide both Employee ID and Name', 'error')
            return render_template('login.html')

        employee = Employee.query.filter_by(emp_id=emp_id).first()

        if employee:
            if employee.name.strip().lower() == name.lower():
                session['emp_id'] = emp_id
                session['emp_name'] = employee.name
                session['department'] = employee.department
                session['is_admin'] = employee.is_admin
                flash(f'Welcome back, {employee.name}!', 'success')
                return redirect(url_for('attendance.dashboard'))
            else:
                flash('Employee ID and Name do not match', 'error')
        else:
            # Auto-register new employee (no password required)
            new_employee = Employee(
                emp_id=emp_id,
                name=name,
                email='',
                department='General',
                password_hash=generate_password_hash(name),
                is_admin=False
            )
            db.session.add(new_employee)
            db.session.commit()

            session['emp_id'] = emp_id
            session['emp_name'] = name
            session['department'] = 'General'
            session['is_admin'] = False
            flash('Account created and signed in successfully!', 'success')
            return redirect(url_for('attendance.dashboard'))

    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('auth.login'))


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        emp_id = request.form.get('emp_id', '').upper()
        name = request.form.get('name', '')
        email = request.form.get('email', '')
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if Employee.query.filter_by(emp_id=emp_id).first():
            flash('Employee ID already registered', 'error')
        elif password != confirm_password:
            flash('Passwords do not match', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
        else:
            new_employee = Employee(
                emp_id=emp_id,
                name=name,
                email=email,
                department='General',
                password_hash=generate_password_hash(password),
                is_admin=False
            )
            db.session.add(new_employee)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('auth.login'))

    return render_template('register.html')


app.register_blueprint(auth_bp)

# ────────────────────────────────────────────────
#   ATTENDANCE BLUEPRINT
# ────────────────────────────────────────────────

attendance_bp = Blueprint('attendance', __name__, url_prefix='')


@attendance_bp.route('/dashboard')
def dashboard():
    if session.get('emp_id') is None:
        return render_template('face_attendance.html', ist_now=current_ist_datetime_str())

    emp = Employee.query.filter_by(emp_id=session.get('emp_id')).first()
    if not emp:
        session.clear()
        return render_template('face_attendance.html', ist_now=current_ist_datetime_str())

    # Get today's attendance
    today = current_ist_date()
    today_attendance = Attendance.query.filter(
        Attendance.employee_id == emp.id,
        Attendance.date == today
    ).first()

    today_status = 'Not Marked'
    today_attendance_status = '-'
    today_hours = 0.0

    if today_attendance:
        if today_attendance.check_out:
            today_status = 'Checked Out'
            today_hours = today_attendance.total_hours or 0.0
        else:
            today_status = 'Checked In'

        today_attendance_status = today_attendance.status or 'Unknown'

    # Calculate monthly stats
    current_month = today.replace(day=1)
    next_month = current_month.replace(month=current_month.month % 12 + 1, year=current_month.year + (current_month.month // 12))

    month_attendance = Attendance.query.filter(
        Attendance.employee_id == emp.id,
        Attendance.date >= current_month,
        Attendance.date < next_month
    ).count()

    return render_template(
        'dashboard.html',
        emp=emp,
        today_status=today_status,
        today_attendance_status=today_attendance_status,
        today_hours=today_hours,
        monthly_days=month_attendance,
        avg_hours='8.5',
        ist_now=current_ist_datetime_str()
    )


@attendance_bp.route('/checkout/<emp_id>', methods=['POST'])
@login_required
def checkout(emp_id):
    emp = Employee.query.filter_by(emp_id=emp_id).first()
    if not emp:
        flash('Employee not found.', 'error')
        return redirect(url_for('attendance.profile'))

    today = current_ist_date()
    today_attendance = Attendance.query.filter(
        Attendance.employee_id == emp.id,
        Attendance.date == today
    ).first()

    if today_attendance and today_attendance.check_in and not today_attendance.check_out:
        now = now_ist()
        current_time_str = now.strftime('%H:%M')
        check_in_time = datetime.strptime(today_attendance.check_in, '%H:%M')
        check_out_time = datetime.strptime(current_time_str, '%H:%M')

        if check_out_time < check_in_time:
            check_out_time = check_out_time.replace(day=check_out_time.day + 1)

        total_hours = round((check_out_time - check_in_time).total_seconds() / 3600, 2)
        today_attendance.check_out = current_time_str
        today_attendance.total_hours = total_hours
        db.session.commit()

        flash(f'Checked out successfully! Total hours: {total_hours:.2f} hrs', 'success')
    else:
        flash('No check-in found or already checked out.', 'error')

    return redirect(url_for('attendance.profile', emp_id=emp_id))


@attendance_bp.route('/face-attendance')
def face_attendance():
    return render_template('face_attendance.html', ist_now=current_ist_datetime_str())


@attendance_bp.route('/profile')
@attendance_bp.route('/profile/<emp_id>')
@login_required
def profile(emp_id=None):
    # Support both path and query parameter for emp_id
    if not emp_id:
        emp_id = request.args.get('emp_id') or session.get('emp_id')

    emp = Employee.query.filter_by(emp_id=emp_id).first()
    if not emp:
        flash('Employee not found.', 'error')
        return redirect(url_for('attendance.dashboard'))

    emp_records = Attendance.query.filter_by(employee_id=emp.id).order_by(Attendance.date.desc()).limit(10).all()
    today = current_ist_date()
    today_record = next((record for record in emp_records if record.date == today), None)

    return render_template(
        'profile.html',
        emp=emp,
        records=emp_records,
        today=today,
        today_record=today_record,
        ist_now=current_ist_datetime_str()
    )


# Edit employee profile info (department, email)


@attendance_bp.route('/edit-profile', methods=['POST'])
@attendance_bp.route('/edit-profile/<emp_id>', methods=['POST'])
@login_required
def edit_profile(emp_id=None):
    if not emp_id:
        emp_id = request.form.get('emp_id') or session.get('emp_id')

    if not emp_id:
        flash('Employee not found.', 'error')
        return redirect(url_for('attendance.profile'))

    department = request.form.get('department', '').strip()
    email = request.form.get('email', '').strip()

    emp = Employee.query.filter_by(emp_id=emp_id).first()
    if not emp:
        flash('Employee not found.', 'error')
        return redirect(url_for('attendance.profile', emp_id=emp_id))

    emp.department = department
    emp.email = email
    db.session.commit()
    flash('Profile updated successfully!', 'success')
    return redirect(url_for('attendance.profile', emp_id=emp_id))


@attendance_bp.route('/enroll')
@login_required
def enroll():
    flash('Enrollment is disabled.', 'warning')
    return redirect(url_for('attendance.employee_list'))


@attendance_bp.route('/enroll', methods=['POST'])
@login_required
def enroll_post():
    flash('Enrollment is disabled.', 'warning')
    return redirect(url_for('attendance.employee_list'))


@attendance_bp.route('/employee-list')
@login_required
def employee_list():
    employees = Employee.query.order_by(Employee.created_at.desc()).all()
    return render_template('employee_list.html', employees=employees)


@attendance_bp.route('/bulk-enroll', methods=['POST'])
@login_required
def bulk_enroll():
    flash('Enrollment is disabled.', 'warning')
    return redirect(url_for('attendance.employee_list'))


@attendance_bp.route('/api/verify-face', methods=['POST'])
def verify_face():
    """Verify face recognition and return employee details if recognized"""
    try:
        data = request.get_json(silent=True) or {}
        if not data or 'image_data' not in data:
            return jsonify({'success': False, 'message': 'No face image data provided', 'detected': False}), 400

        image_data = data['image_data']

        face_result = extract_face_data_from_base64(image_data)
        if not face_result.get('ok'):
            if face_result.get('code') in {'dependency_missing', 'library_missing'}:
                return jsonify({
                    'success': False,
                    'message': face_result['message'],
                    'detected': False,
                    'recognized': False
                }), 503
            return jsonify({
                'success': False,
                'message': face_result['message'],
                'detected': face_result['code'] != 'no_face',
                'recognized': False
            }), 400

        unknown_encoding = face_result['encoding']

        employees = Employee.query.filter(Employee.face_encoding.isnot(None)).all()
        if not employees:
            return jsonify({
                'success': False,
                'message': 'No enrolled employees found',
                'detected': True,
                'recognized': False
            })

        best_match = None
        best_distance = 1.0
        TOLERANCE = 0.6

        for emp in employees:
            if not emp.face_encoding:
                continue
            try:
                known_encoding = pickle.loads(emp.face_encoding)
                distance = calculate_face_distance(known_encoding, unknown_encoding)
                if distance is not None and distance < best_distance:
                    best_distance = distance
                    if distance < TOLERANCE:
                        best_match = emp
            except Exception as compare_error:
                print(f"Error comparing face with {emp.emp_id}: {compare_error}")
                continue

        if best_match:
            set_employee_session(best_match)
            return jsonify({
                'success': True,
                'message': 'Face recognized successfully!',
                'detected': True,
                'recognized': True,
                'employee': {
                    'emp_id': best_match.emp_id,
                    'name': best_match.name,
                    'department': best_match.department or 'General'
                },
                'redirect_url': url_for('attendance.profile', emp_id=best_match.emp_id)
            })

        return jsonify({
            'success': False,
            'message': 'Face detected but not recognized. Please enroll.',
            'detected': True,
            'recognized': False
        })

    except Exception as e:
        print(f"Face verification error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}',
            'detected': False
        }), 500


@attendance_bp.route('/api/check-face', methods=['POST'])
def check_face():
    """Validate that a single usable face is visible before recognition or enrollment."""
    try:
        data = request.get_json(silent=True) or {}
        image_data = data.get('image_data')
        if not image_data:
            return jsonify({
                'success': False,
                'message': 'No face image data provided',
                'detected': False
            }), 400

        face_check = check_face_image(image_data)
        if face_check.get('ok'):
            return jsonify({
                'success': True,
                'message': face_check['message'],
                'detected': True
            })

        if face_check.get('code') in {'dependency_missing', 'library_missing'}:
            return jsonify({
                'success': False,
                'message': face_check['message'],
                'detected': False
            }), 503

        return jsonify({
            'success': False,
            'message': face_check['message'],
            'detected': face_check.get('code') not in {'no_face', 'missing_image', 'invalid_image'}
        })
    except Exception as e:
        print(f"Face precheck error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error while validating the face.',
            'detected': False
        }), 500


@attendance_bp.route('/api/enroll-face', methods=['POST'])
def enroll_face():
    """Enroll new employee with face recognition"""
    try:
        return jsonify({
            'success': False,
            'message': 'Enrollment is disabled.'
        }), 403

        payload = request.get_json(silent=True) or {}
        source = request.form if request.form else payload
        emp_id = source.get('emp_id', '').upper().strip()
        name = source.get('name', '').strip()
        department = source.get('department', 'General')
        image_data = source.get('image_data') or source.get('image')

        if not emp_id or not name:
            return jsonify({'success': False, 'message': 'Employee ID and name are required'}), 400

        if not image_data:
            return jsonify({'success': False, 'message': 'No face image provided'}), 400

        face_result = extract_face_data_from_base64(image_data)
        if not face_result.get('ok'):
            if face_result.get('code') in {'dependency_missing', 'library_missing'}:
                return jsonify({'success': False, 'message': face_result['message']}), 503
            return jsonify({'success': False, 'message': face_result['message']}), 400

        try:
            face_encoding_data = pickle.dumps(face_result['encoding'])
        except Exception as pickle_error:
            print(f"Face encoding serialization error: {pickle_error}")
            return jsonify({'success': False, 'message': 'Failed to store face encoding'}), 500

        existing_employee = Employee.query.filter_by(emp_id=emp_id).first()
        if existing_employee:
            if existing_employee.face_encoding:
                try:
                    stored_encoding = pickle.loads(existing_employee.face_encoding)
                except Exception as stored_face_error:
                    print(f"Stored face decode error for {emp_id}: {stored_face_error}")
                    return jsonify({
                        'success': False,
                        'message': 'Existing employee face data is invalid. Please contact admin.'
                    }), 500

                if not is_same_face(stored_encoding, face_result['encoding']):
                    return jsonify({
                        'success': False,
                        'message': f'Employee ID {emp_id} already belongs to another face. Access denied.'
                    }), 403

                if name and existing_employee.name.strip().lower() != name.lower():
                    return jsonify({
                        'success': False,
                        'message': f'Employee ID {emp_id} is already registered for {existing_employee.name}.'
                    }), 403
            elif name and existing_employee.name and existing_employee.name.strip().lower() != name.lower():
                return jsonify({
                    'success': False,
                    'message': f'Employee ID {emp_id} is already registered for {existing_employee.name}.'
                }), 403

            existing_employee.face_encoding = face_encoding_data
            if name:
                existing_employee.name = name
            if department:
                existing_employee.department = department
            db.session.commit()

            try:
                save_face_backup(emp_id, image_data)
            except Exception as save_error:
                print(f"Error saving face image: {str(save_error)}")

            set_employee_session(existing_employee)
            attendance_type = source.get('attendance_type', 'Onsite')
            attendance_result = mark_face_attendance(existing_employee, department=department, attendance_type=attendance_type)

            return jsonify({
                'success': attendance_result['success'],
                'message': f"Face updated for {existing_employee.name}. {attendance_result['message']}",
                'employee': {
                    'emp_id': existing_employee.emp_id,
                    'name': existing_employee.name,
                    'department': existing_employee.department or 'General'
                },
                'attendance_action': attendance_result['action'],
                'next_action': attendance_result.get('next_action'),
                'attendance': serialize_attendance_record(attendance_result.get('attendance')),
                'redirect_url': url_for('attendance.profile', emp_id=existing_employee.emp_id)
            })

        new_employee = Employee(
            emp_id=emp_id,
            name=name,
            email='',
            department=department,
            password_hash=generate_password_hash(name),
            is_admin=False,
            face_encoding=face_encoding_data
        )

        db.session.add(new_employee)
        db.session.commit()

        try:
            save_face_backup(emp_id, image_data)
        except Exception as save_error:
            print(f"Error saving face image: {str(save_error)}")

        set_employee_session(new_employee)
        attendance_type = source.get('attendance_type', 'Onsite')
        attendance_result = mark_face_attendance(new_employee, department=department, attendance_type=attendance_type)

        return jsonify({
            'success': attendance_result['success'],
            'message': f'Employee {name} enrolled successfully. {attendance_result["message"]}',
            'employee': {
                'emp_id': emp_id,
                'name': name,
                'department': department
            },
            'attendance_action': attendance_result['action'],
            'next_action': attendance_result.get('next_action'),
            'attendance': serialize_attendance_record(attendance_result.get('attendance')),
            'redirect_url': url_for('attendance.profile', emp_id=new_employee.emp_id)
        })
    except Exception as e:
        db.session.rollback()
        print(f"Enrollment API error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Enrollment failed: {str(e)}'
        }), 500


@attendance_bp.route('/api/face-attendance', methods=['POST'])
def face_attendance_api():
    """Mark attendance for recognized employee"""
    try:
        payload = request.get_json(silent=True) or {}
        source = request.form if request.form else payload
        emp_id = source.get('emp_id', '').upper().strip()
        department = source.get('department')
        attendance_type = source.get('attendance_type', 'Onsite')

        if not emp_id:
            return jsonify({'success': False, 'message': 'Employee ID is required'}), 400

        emp = Employee.query.filter_by(emp_id=emp_id).first()
        if not emp:
            return jsonify({'success': False, 'message': 'Employee not found'}), 404

        set_employee_session(emp)
        attendance_result = mark_face_attendance(emp, department=department, attendance_type=attendance_type)
        status_code = 200 if attendance_result['success'] else 400
        return jsonify({
            'success': attendance_result['success'],
            'message': attendance_result['message'],
            'action': attendance_result['action'],
            'next_action': attendance_result.get('next_action'),
            'attendance': serialize_attendance_record(attendance_result.get('attendance')),
            'employee': {
                'emp_id': emp.emp_id,
                'name': emp.name,
                'department': emp.department or 'General'
            },
            'redirect_url': url_for('attendance.profile', emp_id=emp.emp_id)
        }), status_code
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Attendance marking failed: {str(e)}'
        }), 500


app.register_blueprint(attendance_bp)

# ────────────────────────────────────────────────
#   ADMIN BLUEPRINT
# ────────────────────────────────────────────────

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


@admin_bp.route('/dashboard')
@admin_required
def admin_dashboard():
    employees = Employee.query.all()
    attendance_records = Attendance.query.filter(Attendance.date == current_ist_date()).all()

    # Get today's summary
    present = len([r for r in attendance_records if r.check_in])
    on_time = len([r for r in attendance_records if r.status == 'On-Time'])
    late = len([r for r in attendance_records if r.status == 'Late'])
    absent = len(employees) - present

    return render_template(
        'admin.html',
        employees=employees,
        attendance_records=attendance_records,
        today_present=present,
        today_on_time=on_time,
        today_late=late,
        today_absent=absent
    )


@admin_bp.route('/attendance-list')
@admin_required
def attendance_list():
    attendance_records = Attendance.query.all()
    return render_template('daily_report.html', records=attendance_records)


app.register_blueprint(admin_bp)

# ────────────────────────────────────────────────
#   API ENDPOINTS
# ────────────────────────────────────────────────


@app.route('/api/employees-status')
@login_required
def employees_status():
    """API endpoint for employee status"""
    employees = Employee.query.all()
    today = current_ist_date()

    results = []
    for emp in employees:
        attendance = Attendance.query.filter(
            Attendance.employee_id == emp.id,
            Attendance.date == today
        ).first()

        status = 'Absent'
        check_in = None

        if attendance and attendance.check_in:
            status = 'Present'
            check_in = attendance.check_in

        results.append({
            'emp_id': emp.emp_id,
            'name': emp.name,
            'status': status,
            'check_in': check_in
        })

    return jsonify(results)


@app.route('/api/attendance-today')
@login_required
def attendance_today():
    """API endpoint for today's attendance"""
    today_records = Attendance.query.filter(Attendance.date == current_ist_date()).all()
    return jsonify([{
        'id': r.id,
        'emp_id': r.employee.emp_id,
        'name': r.employee.name,
        'check_in': r.check_in,
        'check_out': r.check_out,
        'status': r.status,
        'attendance_type': r.attendance_type,
        'total_hours': r.total_hours
    } for r in today_records])


# ────────────────────────────────────────────────
#   ERROR HANDLERS
# ────────────────────────────────────────────────


@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Page not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '').lower() in ('1', 'true', 'yes')
    app.run(host='0.0.0.0', port=port, debug=debug)
