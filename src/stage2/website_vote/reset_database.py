#!/usr/bin/env python3
"""
Database Reset Utility for Thesis Voting System

This script safely backs up existing data and recreates the database schema.
Use this if you encounter database schema issues or need a clean start.
"""

import os
import shutil
import sqlite3
import psycopg2
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    url = urlparse(DATABASE_URL)
    DB_CONFIG = {
        'host': url.hostname,
        'port': url.port,
        'database': url.path[1:],
        'user': url.username,
        'password': url.password
    }
    USE_POSTGRES = True
else:
    DATABASE = Path("data/votes.db")
    USE_POSTGRES = False

BACKUP_DIR = Path("backups")
BACKUP_DIR.mkdir(exist_ok=True)


def backup_existing_data():
    """Backup existing data before reset"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("🔄 Creating backup of existing data...")

    if USE_POSTGRES:
        try:
            conn = psycopg2.connect(
                host=DB_CONFIG['host'],
                port=DB_CONFIG['port'],
                database=DB_CONFIG['database'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password']
            )

            # Check if votes table exists
            cursor = conn.cursor()
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'votes'
                );
            """)

            if cursor.fetchone()[0]:
                # Export existing votes
                cursor.execute("SELECT * FROM votes")
                votes = cursor.fetchall()

                if votes:
                    backup_file = BACKUP_DIR / f"votes_backup_{timestamp}.sql"
                    with open(backup_file, 'w') as f:
                        f.write("-- Backup created: " +
                                datetime.now().isoformat() + "\n")
                        for vote in votes:
                            f.write(f"INSERT INTO votes VALUES {vote};\n")
                    print(f"✅ PostgreSQL data backed up to: {backup_file}")
                else:
                    print("ℹ️  No existing votes to backup")
            else:
                print("ℹ️  No existing votes table found")

            conn.close()

        except Exception as e:
            print(f"⚠️  PostgreSQL backup failed: {e}")

    else:
        # SQLite backup
        if DATABASE.exists():
            backup_file = BACKUP_DIR / f"votes_backup_{timestamp}.db"
            shutil.copy2(DATABASE, backup_file)
            print(f"✅ SQLite database backed up to: {backup_file}")
        else:
            print("ℹ️  No existing SQLite database to backup")


def recreate_schema():
    """Recreate database schema with correct structure"""
    print("🔧 Recreating database schema...")

    if USE_POSTGRES:
        try:
            conn = psycopg2.connect(
                host=DB_CONFIG['host'],
                port=DB_CONFIG['port'],
                database=DB_CONFIG['database'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password']
            )
            cursor = conn.cursor()

            # Drop existing tables
            cursor.execute("DROP TABLE IF EXISTS votes CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS app_monitoring CASCADE;")

            # Create votes table with correct schema
            cursor.execute("""
            CREATE TABLE votes (
                user_session TEXT NOT NULL,
                contract_id TEXT NOT NULL,
                option1_key TEXT NOT NULL,
                option2_key TEXT NOT NULL,
                winner_key TEXT NOT NULL,
                voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_agent TEXT,
                ip_address TEXT,
                session_start_time TIMESTAMP,
                PRIMARY KEY (user_session, contract_id)
            );
            """)

            # Create monitoring table
            cursor.execute("""
            CREATE TABLE app_monitoring (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT NOT NULL,
                metric_value TEXT NOT NULL,
                details JSONB
            );
            """)

            # Create indexes for better performance
            cursor.execute(
                "CREATE INDEX idx_votes_contract_id ON votes(contract_id);")
            cursor.execute(
                "CREATE INDEX idx_votes_voted_at ON votes(voted_at);")
            cursor.execute(
                "CREATE INDEX idx_monitoring_timestamp ON app_monitoring(timestamp);")

            conn.commit()
            conn.close()
            print("✅ PostgreSQL schema recreated successfully")

        except Exception as e:
            print(f"❌ PostgreSQL schema creation failed: {e}")
            return False

    else:
        try:
            # Ensure data directory exists
            DATABASE.parent.mkdir(exist_ok=True)

            # Remove old database if it exists
            if DATABASE.exists():
                DATABASE.unlink()

            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()

            # Create votes table
            cursor.execute("""
            CREATE TABLE votes (
                user_session TEXT NOT NULL,
                contract_id TEXT NOT NULL,
                option1_key TEXT NOT NULL,
                option2_key TEXT NOT NULL,
                winner_key TEXT NOT NULL,
                voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_agent TEXT,
                ip_address TEXT,
                session_start_time TIMESTAMP,
                PRIMARY KEY (user_session, contract_id)
            );
            """)

            # Create monitoring table
            cursor.execute("""
            CREATE TABLE app_monitoring (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT NOT NULL,
                metric_value TEXT NOT NULL,
                details TEXT
            );
            """)

            # Create indexes
            cursor.execute(
                "CREATE INDEX idx_votes_contract_id ON votes(contract_id);")
            cursor.execute(
                "CREATE INDEX idx_votes_voted_at ON votes(voted_at);")
            cursor.execute(
                "CREATE INDEX idx_monitoring_timestamp ON app_monitoring(timestamp);")

            # Enable optimizations
            cursor.execute('PRAGMA journal_mode=WAL;')
            cursor.execute('PRAGMA synchronous=NORMAL;')
            cursor.execute('PRAGMA cache_size=10000;')

            conn.commit()
            conn.close()
            print("✅ SQLite schema recreated successfully")

        except Exception as e:
            print(f"❌ SQLite schema creation failed: {e}")
            return False

    return True


def verify_schema():
    """Verify the new schema is correct"""
    print("🔍 Verifying schema...")

    try:
        if USE_POSTGRES:
            conn = psycopg2.connect(
                host=DB_CONFIG['host'],
                port=DB_CONFIG['port'],
                database=DB_CONFIG['database'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password']
            )
            cursor = conn.cursor()

            # Check tables exist
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name IN ('votes', 'app_monitoring')
                ORDER BY table_name;
            """)
            tables = [row[0] for row in cursor.fetchall()]

            if 'votes' in tables and 'app_monitoring' in tables:
                print("✅ All tables created successfully")

                # Test insert
                cursor.execute("""
                    INSERT INTO votes (user_session, contract_id, option1_key, option2_key, winner_key)
                    VALUES ('test_session', 'test_contract', 'option1', 'option2', 'option1')
                """)
                cursor.execute(
                    "DELETE FROM votes WHERE user_session = 'test_session'")
                conn.commit()
                print("✅ Database write test successful")
            else:
                print(f"❌ Missing tables. Found: {tables}")
                return False

            conn.close()

        else:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()

            # Check tables
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables = [row[0] for row in cursor.fetchall()]

            if 'votes' in tables and 'app_monitoring' in tables:
                print("✅ All tables created successfully")

                # Test insert
                cursor.execute("""
                    INSERT INTO votes (user_session, contract_id, option1_key, option2_key, winner_key)
                    VALUES ('test_session', 'test_contract', 'option1', 'option2', 'option1')
                """)
                cursor.execute(
                    "DELETE FROM votes WHERE user_session = 'test_session'")
                conn.commit()
                print("✅ Database write test successful")
            else:
                print(f"❌ Missing tables. Found: {tables}")
                return False

            conn.close()

        return True

    except Exception as e:
        print(f"❌ Schema verification failed: {e}")
        return False


def main():
    """Main reset process"""
    print("🗃️  Thesis Voting System - Database Reset Utility")
    print("=" * 60)

    # Confirm operation
    response = input("⚠️  This will reset your database. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Operation cancelled")
        return

    print("\n🚀 Starting database reset process...")

    # Step 1: Backup existing data
    backup_existing_data()

    # Step 2: Recreate schema
    if not recreate_schema():
        print("❌ Schema recreation failed. Check the errors above.")
        return

    # Step 3: Verify schema
    if not verify_schema():
        print("❌ Schema verification failed. Check the errors above.")
        return

    print("\n🎉 Database reset completed successfully!")
    print("✅ Your voting system is ready for use.")
    print(f"📁 Backups are stored in: {BACKUP_DIR.absolute()}")

    print("\n📋 Next steps:")
    print("1. Start your application: python app.py")
    print("2. Test the voting flow")
    print("3. Check admin dashboard for system health")


if __name__ == "__main__":
    main()
