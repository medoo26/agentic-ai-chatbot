#!/usr/bin/env python3
"""
Setup script to initialize the database and create default admin user
"""
from database import init_db, SessionLocal, AdminUser
from passlib.context import CryptContext
import os
from dotenv import load_dotenv

load_dotenv()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def setup_database():
    """Initialize database and create default admin"""
    print("Initializing database...")
    init_db()
    print("Database initialized successfully!")
    
    # Create default admin
    db = SessionLocal()
    try:
        admin_username = os.getenv("ADMIN_USERNAME", "admin")
        admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
        
        existing_admin = db.query(AdminUser).filter(AdminUser.username == admin_username).first()
        if not existing_admin:
            hashed_password = pwd_context.hash(admin_password)
            new_admin = AdminUser(username=admin_username, password_hash=hashed_password)
            db.add(new_admin)
            db.commit()
            print(f"Created default admin user:")
            print(f"  Username: {admin_username}")
            print(f"  Password: {admin_password}")
        else:
            print(f"Admin user '{admin_username}' already exists")
    finally:
        db.close()
    
    print("\nSetup complete!")
    print("You can now start the server with: python run.py")

if __name__ == "__main__":
    setup_database()

