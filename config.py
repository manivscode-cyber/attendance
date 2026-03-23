import os

# In production (e.g. Heroku), set a secure SECRET_KEY and DATABASE_URL.
SECRET_KEY = os.environ.get("SECRET_KEY", "attendance_secret")

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    # Normalize legacy provider URLs so SQLAlchemy can load the Postgres dialect.
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
    elif DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

    SQLALCHEMY_DATABASE_URI = DATABASE_URL
else:
    # Local sqlite fallback (file stored in repo folder)
    DATABASE = "attendance.db"
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.abspath(DATABASE)}"

SQLALCHEMY_TRACK_MODIFICATIONS = False
