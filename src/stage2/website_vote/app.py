from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, g, jsonify, flash
import json
import csv
import os
from pathlib import Path
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import threading
import time
import shutil
import traceback
from functools import wraps, lru_cache
import hashlib
from itertools import combinations
import random  # For randomized pair selection
app = Flask(__name__, static_folder='public')
app.secret_key = os.urandom(24)  # Random secret key for user differentiation


load_dotenv()

# Performance optimization: Reduce logging overhead in production
DEBUG_MODE = os.environ.get('DEBUG', 'False').lower() == 'true'
DEBUG_MODE = True
LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.WARNING

# Configure streamlined logging for better performance
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Simplified logging setup
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'app.log'),
        logging.StreamHandler()
    ]
)

app.logger.setLevel(LOG_LEVEL)

# Lightweight loggers for production
user_logger = logging.getLogger('user_actions')
user_logger.setLevel(LOG_LEVEL)

error_logger = logging.getLogger('errors')
error_logger.setLevel(logging.ERROR)

# Database configuration with optimized connection pooling
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    url = urlparse(DATABASE_URL)
    DB_CONFIG = {
        'host': url.hostname,
        'port': url.port,
        'database': url.path[1:],
        'user': url.username,
        'password': url.password,
        'connect_timeout': 10,  # Reduced timeout
        'application_name': 'thesis_voting_app'
    }
    USE_POSTGRES = True
    if DEBUG_MODE:
        app.logger.info("Configured for PostgreSQL")
else:
    DATABASE = Path("data/votes.db")
    DATABASE.parent.mkdir(exist_ok=True)
    USE_POSTGRES = False
    if DEBUG_MODE:
        app.logger.info("Configured for SQLite")

DATA_DIR = Path("data/stage2_out/val")
PDF_DIR_NAME = "data/contracts_pdf"
SUMMARIES_FILE = DATA_DIR / "contract_summaries.json"
BACKUP_DIR = Path("backups")
BACKUP_DIR.mkdir(exist_ok=True)

# Add new data directories for three-way comparison
THREE_WAY_DATA_DIRS = {
    'fine_tuned_committee': Path("data/fine_tuned_committee_out"),
    'single_stage': Path("data/single_stage_out"),
    'stage2_out': Path("data/stage2_out/test")
}

# Global variables for monitoring
app_start_time = datetime.now()
error_count = 0
vote_count = 0

# Performance optimization: Cache for frequently accessed data
_contract_cache = {}
_summary_cache = {}
_cached_contract_ids = None

# Global variables for three-way comparison
_threeway_contract_cache = {}
_threeway_contract_ids = None


def log_error(func_name, error, extra_info=""):
    """Streamlined error logging"""
    global error_count
    error_count += 1
    error_msg = f"ERROR in {func_name}: {str(error)}"
    if extra_info:
        error_msg += f" | {extra_info}"
    error_logger.error(error_msg)
    if DEBUG_MODE:
        error_logger.error(traceback.format_exc())


def log_user_action(session_id, action, details=""):
    """Lightweight user action logging"""
    if DEBUG_MODE:
        user_logger.info(f"User {session_id[:8]} - {action} - {details}")


def optimized_file_operation(operation, *args, **kwargs):
    """Streamlined file operations with minimal retry for performance"""
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        # Single retry for critical operations only
        try:
            time.sleep(0.1)
            return operation(*args, **kwargs)
        except Exception as retry_error:
            raise retry_error


def validate_session():
    """Optimized session validation"""
    try:
        if 'user_session_id' not in session:
            session['user_session_id'] = hashlib.md5(
                (str(datetime.now()) + str(os.urandom(8))).encode()
            ).hexdigest()[:16]  # Shorter session ID
            session.permanent = True

        # Initialize missing session data in one go
        session_defaults = {
            'vote_history': [],
            'presented_contracts': [],
            'current_contract_idx': -1,
            'current_pair_idx': -1,
            'voted_pairs': {}  # contract_id -> set of voted pair identifiers
        }

        for key, default_value in session_defaults.items():
            if key not in session:
                session[key] = default_value

        session.modified = True
        return True
    except Exception as e:
        log_error("validate_session", e)
        return False


def require_session(f):
    """Lightweight session decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not validate_session():
            flash("Session error. Please restart.", "error")
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated_function


@lru_cache(maxsize=128)
def get_cached_contract_ids():
    """Cache contract IDs to avoid repeated file system operations"""
    global _cached_contract_ids
    if _cached_contract_ids is not None:
        return _cached_contract_ids

    try:
        if not DATA_DIR.exists():
            app.logger.error(
                f"Critical: Data directory {DATA_DIR} does not exist!")
            return []

        contract_files = sorted(DATA_DIR.glob("*.json"))
        contract_ids = []

        for p in contract_files:
            if p.stem != "contract_summaries":
                # Quick validation without full JSON parsing
                try:
                    if p.stat().st_size > 100:  # Basic size check
                        contract_ids.append(p.stem)
                except Exception:
                    continue

        _cached_contract_ids = contract_ids
        return contract_ids
    except Exception as e:
        log_error("get_cached_contract_ids", e)
        return []


# Load contract IDs with caching
contract_ids = get_cached_contract_ids()

# Load contract summaries with caching


@lru_cache(maxsize=1)
def get_cached_summaries():
    """Cache contract summaries"""
    try:
        if SUMMARIES_FILE.exists():
            with optimized_file_operation(open, SUMMARIES_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        log_error("get_cached_summaries", e)
        return {}


contract_summaries = get_cached_summaries()

if DEBUG_MODE:
    app.logger.info(
        f"Loaded {len(contract_ids)} contracts and {len(contract_summaries)} summaries")


# Optimized database connection with connection pooling
_db_pool = None


def get_db():
    """Optimized database connection with simple pooling"""
    db = getattr(g, '_database', None)
    if db is None:
        try:
            if USE_POSTGRES:
                db = g._database = psycopg2.connect(
                    host=DB_CONFIG['host'],
                    port=DB_CONFIG['port'],
                    database=DB_CONFIG['database'],
                    user=DB_CONFIG['user'],
                    password=DB_CONFIG['password'],
                    cursor_factory=RealDictCursor,
                    connect_timeout=DB_CONFIG['connect_timeout']
                )
                db.autocommit = False
            else:
                db = g._database = sqlite3.connect(
                    DATABASE,
                    timeout=10.0,  # Reduced timeout
                    check_same_thread=False
                )
                db.row_factory = sqlite3.Row
                # Optimized SQLite settings
                db.execute('PRAGMA journal_mode=WAL;')
                db.execute('PRAGMA synchronous=NORMAL;')
                db.execute('PRAGMA cache_size=10000;')
                db.execute('PRAGMA temp_store=MEMORY;')
        except Exception as e:
            log_error("get_db", e)
            raise e

    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        try:
            db.close()
        except Exception:
            pass


def ensure_schema():
    """Streamlined schema creation with proper migration support"""
    try:
        if USE_POSTGRES:
            conn = psycopg2.connect(
                host=DB_CONFIG['host'],
                port=DB_CONFIG['port'],
                database=DB_CONFIG['database'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                connect_timeout=DB_CONFIG['connect_timeout']
            )
            cursor = conn.cursor()

            # Check if votes table exists and what columns it has
            cursor.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'votes' AND table_schema = 'public';
            """)
            existing_columns = [row[0] for row in cursor.fetchall()]

            if existing_columns:
                # Table exists, check if we need to migrate
                migration_needed = False

                # Check each column individually and add only if missing
                if 'pair_identifier' not in existing_columns:
                    if DEBUG_MODE:
                        app.logger.info(
                            "Adding pair_identifier column to PostgreSQL votes table")
                    cursor.execute(
                        "ALTER TABLE votes ADD COLUMN pair_identifier TEXT;")
                    migration_needed = True

                if 'option1_key' not in existing_columns:
                    if DEBUG_MODE:
                        app.logger.info(
                            "Adding option1_key column to PostgreSQL votes table")
                    cursor.execute(
                        "ALTER TABLE votes ADD COLUMN option1_key TEXT;")
                    migration_needed = True

                if 'option2_key' not in existing_columns:
                    if DEBUG_MODE:
                        app.logger.info(
                            "Adding option2_key column to PostgreSQL votes table")
                    cursor.execute(
                        "ALTER TABLE votes ADD COLUMN option2_key TEXT;")
                    migration_needed = True

                # Only perform data migration if pair_identifier was added
                if migration_needed and 'pair_identifier' not in existing_columns:
                    # Update existing records with default pair_identifier
                    cursor.execute("""
                        UPDATE votes SET 
                        pair_identifier = COALESCE(option1_key, '0') || 'vs' || COALESCE(option2_key, '1')
                        WHERE pair_identifier IS NULL;
                    """)

                    # Drop the old primary key constraint and add the new one
                    cursor.execute(
                        "ALTER TABLE votes DROP CONSTRAINT IF EXISTS votes_pkey;")
                    cursor.execute(
                        "                    ALTER TABLE votes ADD PRIMARY KEY (user_session, contract_id, pair_identifier);")
            else:
                # Create new table
                cursor.execute("""
                CREATE TABLE votes (
                    user_session TEXT NOT NULL,
                    contract_id TEXT NOT NULL,
                    pair_identifier TEXT NOT NULL,
                    option1_key TEXT NOT NULL,
                    option2_key TEXT NOT NULL,
                    winner_key TEXT NOT NULL,
                    voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_agent TEXT,
                    ip_address TEXT,
                    session_start_time TIMESTAMP,
                    PRIMARY KEY (user_session, contract_id, pair_identifier)
                );
                """)

            # Handle three-way comparison votes table with proper migration
            cursor.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'threeway_votes' AND table_schema = 'public';
            """)
            threeway_existing_columns = [row[0] for row in cursor.fetchall()]

            if threeway_existing_columns:
                # Table exists, check if we need to migrate
                threeway_migration_needed = False

                # Check for new rating columns that were added in the latest version
                new_rating_columns = [
                    'fine_tuned_committee_clarity', 'fine_tuned_committee_legal', 'fine_tuned_committee_reasoning', 'fine_tuned_committee_alignment',
                    'single_stage_clarity', 'single_stage_legal', 'single_stage_reasoning', 'single_stage_alignment',
                    'non_fine_tuned_committee_clarity', 'non_fine_tuned_committee_legal', 'non_fine_tuned_committee_reasoning', 'non_fine_tuned_committee_alignment'
                ]

                for column in new_rating_columns:
                    if column not in threeway_existing_columns:
                        if DEBUG_MODE:
                            app.logger.info(
                                f"Adding {column} column to PostgreSQL threeway_votes table")
                        cursor.execute(
                            f"ALTER TABLE threeway_votes ADD COLUMN {column} INTEGER;")
                        threeway_migration_needed = True

                # Check for filename columns
                filename_columns = ['fine_tuned_committee_filename',
                                    'single_stage_filename', 'non_fine_tuned_committee_filename']
                for column in filename_columns:
                    if column not in threeway_existing_columns:
                        if DEBUG_MODE:
                            app.logger.info(
                                f"Adding {column} column to PostgreSQL threeway_votes table")
                        cursor.execute(
                            f"ALTER TABLE threeway_votes ADD COLUMN {column} TEXT;")
                        threeway_migration_needed = True

                # Check for source columns
                source_columns = ['fine_tuned_committee_source',
                                  'single_stage_source', 'non_fine_tuned_committee_source']
                for column in source_columns:
                    if column not in threeway_existing_columns:
                        if DEBUG_MODE:
                            app.logger.info(
                                f"Adding {column} column to PostgreSQL threeway_votes table")
                        cursor.execute(
                            f"ALTER TABLE threeway_votes ADD COLUMN {column} TEXT;")
                        threeway_migration_needed = True

                if threeway_migration_needed and DEBUG_MODE:
                    app.logger.info(
                        "PostgreSQL threeway_votes table migrated successfully")
            else:
                # Create new table
                cursor.execute("""
                CREATE TABLE threeway_votes (
                    user_session TEXT NOT NULL,
                    contract_id TEXT NOT NULL,
                    fine_tuned_committee_filename TEXT NOT NULL,
                    single_stage_filename TEXT NOT NULL,
                    non_fine_tuned_committee_filename TEXT NOT NULL,
                    fine_tuned_committee_source TEXT NOT NULL,
                    single_stage_source TEXT NOT NULL,
                    non_fine_tuned_committee_source TEXT NOT NULL,
                    winner_source TEXT NOT NULL,
                    fine_tuned_committee_clarity INTEGER,
                    fine_tuned_committee_legal INTEGER,
                    fine_tuned_committee_reasoning INTEGER,
                    fine_tuned_committee_alignment INTEGER,
                    single_stage_clarity INTEGER,
                    single_stage_legal INTEGER,
                    single_stage_reasoning INTEGER,
                    single_stage_alignment INTEGER,
                    non_fine_tuned_committee_clarity INTEGER,
                    non_fine_tuned_committee_legal INTEGER,
                    non_fine_tuned_committee_reasoning INTEGER,
                    non_fine_tuned_committee_alignment INTEGER,
                    voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_agent TEXT,
                    ip_address TEXT,
                    session_start_time TIMESTAMP,
                    PRIMARY KEY (user_session, contract_id)
                );
                """)
                if DEBUG_MODE:
                    app.logger.info("PostgreSQL threeway_votes table created")

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS app_monitoring (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT NOT NULL,
                metric_value TEXT NOT NULL,
                details JSONB
            );
            """)

            conn.commit()
            conn.close()
            if DEBUG_MODE:
                app.logger.info("PostgreSQL schema ensured")

        else:
            conn = sqlite3.connect(DATABASE, timeout=10.0)
            cursor = conn.cursor()

            # Check for existing table and migrate if needed
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='votes';")
            if cursor.fetchone():
                cursor.execute("PRAGMA table_info(votes);")
                columns = [row[1] for row in cursor.fetchall()]
                if 'pair_identifier' not in columns:
                    if DEBUG_MODE:
                        app.logger.info(
                            "Migrating SQLite votes table to include pair_identifier")
                    cursor.execute("DROP TABLE votes;")

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS votes (
                user_session TEXT NOT NULL,
                contract_id TEXT NOT NULL,
                pair_identifier TEXT NOT NULL,
                option1_key TEXT NOT NULL,
                option2_key TEXT NOT NULL,
                winner_key TEXT NOT NULL,
                voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_agent TEXT,
                ip_address TEXT,
                session_start_time TIMESTAMP,
                PRIMARY KEY (user_session, contract_id, pair_identifier)
            );
            """)

            # Handle three-way comparison votes table with proper migration for SQLite
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='threeway_votes';")
            if cursor.fetchone():
                # Table exists, check if we need to migrate
                cursor.execute("PRAGMA table_info(threeway_votes);")
                existing_columns = [row[1] for row in cursor.fetchall()]

                # Check for new rating columns that were added in the latest version
                new_rating_columns = [
                    'fine_tuned_committee_clarity', 'fine_tuned_committee_legal', 'fine_tuned_committee_reasoning', 'fine_tuned_committee_alignment',
                    'single_stage_clarity', 'single_stage_legal', 'single_stage_reasoning', 'single_stage_alignment',
                    'non_fine_tuned_committee_clarity', 'non_fine_tuned_committee_legal', 'non_fine_tuned_committee_reasoning', 'non_fine_tuned_committee_alignment'
                ]

                migration_needed = False
                for column in new_rating_columns:
                    if column not in existing_columns:
                        if DEBUG_MODE:
                            app.logger.info(
                                f"Adding {column} column to SQLite threeway_votes table")
                        cursor.execute(
                            f"ALTER TABLE threeway_votes ADD COLUMN {column} INTEGER;")
                        migration_needed = True

                # Check for filename columns
                filename_columns = ['fine_tuned_committee_filename',
                                    'single_stage_filename', 'non_fine_tuned_committee_filename']
                for column in filename_columns:
                    if column not in existing_columns:
                        if DEBUG_MODE:
                            app.logger.info(
                                f"Adding {column} column to SQLite threeway_votes table")
                        cursor.execute(
                            f"ALTER TABLE threeway_votes ADD COLUMN {column} TEXT;")
                        migration_needed = True

                # Check for source columns
                source_columns = ['fine_tuned_committee_source',
                                  'single_stage_source', 'non_fine_tuned_committee_source']
                for column in source_columns:
                    if column not in existing_columns:
                        if DEBUG_MODE:
                            app.logger.info(
                                f"Adding {column} column to SQLite threeway_votes table")
                        cursor.execute(
                            f"ALTER TABLE threeway_votes ADD COLUMN {column} TEXT;")
                        migration_needed = True

                if migration_needed and DEBUG_MODE:
                    app.logger.info(
                        "SQLite threeway_votes table migrated successfully")
            else:
                # Create new table
                cursor.execute("""
                CREATE TABLE threeway_votes (
                    user_session TEXT NOT NULL,
                    contract_id TEXT NOT NULL,
                    fine_tuned_committee_filename TEXT NOT NULL,
                    single_stage_filename TEXT NOT NULL,
                    non_fine_tuned_committee_filename TEXT NOT NULL,
                    fine_tuned_committee_source TEXT NOT NULL,
                    single_stage_source TEXT NOT NULL,
                    non_fine_tuned_committee_source TEXT NOT NULL,
                    winner_source TEXT NOT NULL,
                    fine_tuned_committee_clarity INTEGER,
                    fine_tuned_committee_legal INTEGER,
                    fine_tuned_committee_reasoning INTEGER,
                    fine_tuned_committee_alignment INTEGER,
                    single_stage_clarity INTEGER,
                    single_stage_legal INTEGER,
                    single_stage_reasoning INTEGER,
                    single_stage_alignment INTEGER,
                    non_fine_tuned_committee_clarity INTEGER,
                    non_fine_tuned_committee_legal INTEGER,
                    non_fine_tuned_committee_reasoning INTEGER,
                    non_fine_tuned_committee_alignment INTEGER,
                    voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_agent TEXT,
                    ip_address TEXT,
                    session_start_time TIMESTAMP,
                    PRIMARY KEY (user_session, contract_id)
                );
                """)
                if DEBUG_MODE:
                    app.logger.info("SQLite threeway_votes table created")

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS app_monitoring (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT NOT NULL,
                metric_value TEXT NOT NULL,
                details TEXT
            );
            """)

            conn.commit()
            conn.close()
            if DEBUG_MODE:
                app.logger.info("SQLite schema ensured")

    except Exception as e:
        log_error("ensure_schema", e)
        raise e


def backup_database():
    """Lightweight backup for production"""
    try:
        if not DEBUG_MODE:
            return  # Skip frequent backups in production

        with app.app_context():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if USE_POSTGRES:
                backup_votes_to_csv(f"backup_votes_{timestamp}.csv")
            else:
                backup_file = BACKUP_DIR / f"votes_backup_{timestamp}.db"
                if DATABASE.exists():
                    shutil.copy2(DATABASE, backup_file)
                    if DEBUG_MODE:
                        app.logger.info(f"Database backed up to {backup_file}")
    except Exception as e:
        log_error("backup_database", e)


def backup_votes_to_csv(filename=None):
    """Optimized CSV export - includes ALL data (human and AI votes)"""
    try:
        with app.app_context():
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"votes_export_{timestamp}.csv"

            backup_file = BACKUP_DIR / filename

            if USE_POSTGRES:
                conn = psycopg2.connect(
                    host=DB_CONFIG['host'],
                    port=DB_CONFIG['port'],
                    database=DB_CONFIG['database'],
                    user=DB_CONFIG['user'],
                    password=DB_CONFIG['password'],
                    cursor_factory=RealDictCursor
                )
            else:
                conn = sqlite3.connect(DATABASE, timeout=10.0)
                conn.row_factory = sqlite3.Row

            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_session, contract_id, pair_identifier, option1_key, option2_key, winner_key, 
                       voted_at, user_agent, ip_address, session_start_time 
                FROM votes ORDER BY voted_at
            """)

            votes = cursor.fetchall()

            with open(backup_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['user_session', 'contract_id', 'pair_identifier', 'option1_key', 'option2_key', 'winner_key',
                              'voted_at', 'user_agent', 'ip_address', 'session_start_time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for vote in votes:
                    writer.writerow(dict(vote))

            conn.close()
            if DEBUG_MODE:
                app.logger.info(f"Votes exported to {backup_file}")
            return backup_file
    except Exception as e:
        log_error("backup_votes_to_csv", e)
        return None


# Initialize database
try:
    ensure_schema()
    if DEBUG_MODE:
        backup_database()
except Exception as e:
    log_error("initial_setup", e)


def periodic_backup():
    """Reduced frequency backup for performance"""
    while True:
        try:
            # Backup every 4 hours instead of 1 hour
            time.sleep(14400)
            if DEBUG_MODE:
                backup_database()
                backup_votes_to_csv()
        except Exception as e:
            log_error("periodic_backup", e)


# Start backup thread only in debug mode
if DEBUG_MODE:
    backup_thread = threading.Thread(target=periodic_backup, daemon=True)
    backup_thread.start()


@lru_cache(maxsize=256)
def get_contract_summary(contract_id):
    """Cached contract summary retrieval"""
    try:
        return contract_summaries.get(contract_id, {}).get('summary', 'No summary available for this contract.')
    except Exception as e:
        log_error("get_contract_summary", e, f"contract_id: {contract_id}")
        return 'Error loading summary for this contract.'


def record_monitoring_metric(metric_name, metric_value, details=None):
    """Lightweight monitoring for production"""
    if not DEBUG_MODE:
        return  # Skip detailed monitoring in production

    try:
        with app.app_context():
            if USE_POSTGRES:
                conn = psycopg2.connect(
                    host=DB_CONFIG['host'],
                    port=DB_CONFIG['port'],
                    database=DB_CONFIG['database'],
                    user=DB_CONFIG['user'],
                    password=DB_CONFIG['password']
                )
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO app_monitoring (metric_name, metric_value, details) 
                    VALUES (%s, %s, %s)
                """, (metric_name, str(metric_value), json.dumps(details) if details else None))
            else:
                conn = sqlite3.connect(DATABASE, timeout=10.0)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO app_monitoring (metric_name, metric_value, details) 
                    VALUES (?, ?, ?)
                """, (metric_name, str(metric_value), json.dumps(details) if details else None))

            conn.commit()
            conn.close()
    except Exception as e:
        log_error("record_monitoring_metric", e)


@app.errorhandler(404)
def not_found(error):
    log_error("404_error", error, f"URL: {request.url}")
    return render_template('error.html',
                           error_type="Page Not Found",
                           error_message="The page you're looking for doesn't exist."), 404


@app.errorhandler(500)
def internal_error(error):
    log_error("500_error", error, f"URL: {request.url}")
    return render_template('error.html',
                           error_type="Internal Server Error",
                           error_message="Something went wrong. Please try again."), 500


@app.route("/")
def index():
    try:
        log_user_action(session.get(
            'user_session_id', 'unknown'), 'index_visit')
        if not session.get("info_acknowledged"):
            return render_template("info.html")
        return redirect(url_for("show_vote_item"))
    except Exception as e:
        log_error("index", e)
        return render_template('error.html',
                               error_type="System Error",
                               error_message="Unable to load the application. Please refresh the page.")


@app.route("/start_voting", methods=["POST"])
def start_voting():
    try:
        log_user_action('new_user', 'start_voting_attempt')

        if request.form.get("confirm_checkbox") == "confirmed":
            # Validate session
            if not validate_session():
                return render_template("info.html", error_message="Session error. Please try again.")

            session["info_acknowledged"] = True
            session.permanent = True
            session['vote_history'] = []
            session['presented_contracts'] = []
            session['current_contract_idx'] = -1
            session['current_pair_idx'] = -1
            session['voted_pairs'] = {}
            session['last_contract_id'] = None
            session['session_start_time'] = datetime.now().isoformat()
            session.modified = True

            log_user_action(session['user_session_id'], 'voting_started')
            record_monitoring_metric(
                'user_started', 1, {'session_id': session['user_session_id']})

            app.logger.info(
                f"Session initialized with user ID: {session['user_session_id']}")
            return redirect(url_for("show_vote_item"))
        else:
            app.logger.info("Checkbox not confirmed")
            return render_template("info.html", error_message="Please acknowledge the information by checking the box.")

    except Exception as e:
        log_error("start_voting", e)
        return render_template("info.html", error_message="An error occurred. Please try again.")


@app.route("/vote_item")
@app.route("/vote_item/<navigation_action>")
@require_session
def show_vote_item(navigation_action=None):
    try:
        if not session.get("info_acknowledged"):
            return redirect(url_for("index"))

        log_user_action(session['user_session_id'],
                        'vote_item_visit', f"action: {navigation_action}")

        # Get current session state
        voted_pairs = session.get('voted_pairs', {})

        # Find next pair to vote on
        current_contract_id = None
        current_pair = None

        if navigation_action == "next":
            current_contract_id, current_pair = find_next_pair_to_vote_on(
                voted_pairs)
        elif navigation_action == "prev":
            current_contract_id, current_pair = find_previous_pair(voted_pairs)
        else:
            # Default: find first available pair
            current_contract_id, current_pair = find_next_pair_to_vote_on(
                voted_pairs)

        if not current_contract_id or not current_pair:
            record_monitoring_metric('user_completed', 1, {
                'session_id': session['user_session_id']})
            return redirect(url_for("all_done_page"))

        # Load contract data
        contract_data_file = DATA_DIR / f"{current_contract_id}.json"
        if not contract_data_file.exists():
            return redirect(url_for("show_vote_item", navigation_action="next"))

        try:
            with optimized_file_operation(open, contract_data_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            log_error("load_contract_data", e, f"file: {contract_data_file}")
            return redirect(url_for("show_vote_item", navigation_action="next"))

        option1_key, option2_key = current_pair

        # Validate that both options exist
        if option1_key not in data or option2_key not in data:
            return redirect(url_for("show_vote_item", navigation_action="next"))

        option1_report = data[option1_key].get(
            "final_report", "No report available")
        option2_report = data[option2_key].get(
            "final_report", "No report available")

        if not option1_report.strip() or not option2_report.strip():
            return redirect(url_for("show_vote_item", navigation_action="next"))

        # Get cached contract summary
        contract_summary = get_contract_summary(current_contract_id)

        # Handle PDF with simplified logic
        pdf_filename_to_try = f"{current_contract_id}.pdf"
        pdf_display_url = None
        pdf_file_path = Path(app.root_path) / \
            PDF_DIR_NAME / pdf_filename_to_try

        if pdf_file_path.exists():
            pdf_display_url = url_for(
                'serve_contract_pdf', filename=pdf_filename_to_try)
        else:
            # Fallback to example PDF
            example_pdf_filename = "EMERALDHEALTHTHERAPEUTICSINC_06_10_2020-EX-4.5-CONSULTING AGREEMENT - DR. GAETANO MORELLO N.D. INC..PDF"
            example_pdf_path = Path(app.root_path) / \
                PDF_DIR_NAME / example_pdf_filename
            if example_pdf_path.exists():
                pdf_display_url = url_for(
                    'serve_contract_pdf', filename=example_pdf_filename)
                pdf_filename_to_try = example_pdf_filename
            else:
                pdf_filename_to_try = None

        # Store session data for current comparison
        session['current_contract_for_vote'] = {
            'contract_id': current_contract_id,
            'option1_key': option1_key,
            'option2_key': option2_key,
            'pair_identifier': get_pair_identifier(option1_key, option2_key),
            'pdf_url': pdf_display_url,
            'pdf_filename': pdf_filename_to_try
        }
        session.modified = True  # Ensure session changes are saved

        # Calculate navigation state
        can_go_prev = has_previous_pair(voted_pairs)
        can_go_next = has_next_pair(voted_pairs)

        # Check if this specific pair has been voted on
        pair_id = get_pair_identifier(option1_key, option2_key)
        contract_voted_pairs = voted_pairs.get(current_contract_id, set())
        if isinstance(contract_voted_pairs, list):
            contract_voted_pairs = set(contract_voted_pairs)

        voted_option_for_this_pair = None
        is_marked_unclear = False

        # Check if this specific pair was voted on
        for past_vote in session.get('vote_history', []):
            if (past_vote['contract_id'] == current_contract_id and
                    past_vote.get('pair_identifier') == pair_id):
                stored_winner = past_vote['winner']
                if stored_winner == "UNCLEAR":
                    is_marked_unclear = True
                elif stored_winner == option1_key:
                    voted_option_for_this_pair = 'w1'
                elif stored_winner == option2_key:
                    voted_option_for_this_pair = 'w2'
                break

        # Calculate progress information
        total_expected = get_total_expected_comparisons()
        completed_count = sum(len(pairs) if isinstance(pairs, (set, list)) else 0
                              for pairs in voted_pairs.values())

        progress_info = {
            'current_pair': f"{option1_key} vs {option2_key}",
            'completed': completed_count,
            'total': total_expected,
            'contract_id': current_contract_id
        }

        record_monitoring_metric('pair_viewed', 1, {
            'contract_id': current_contract_id,
            'pair': pair_id,
            'session_id': session['user_session_id']
        })

        return render_template("vote.html",
                               contract_id=current_contract_id,
                               option1_key=option1_key,
                               option2_key=option2_key,
                               pair_identifier=get_pair_identifier(
                                   option1_key, option2_key),
                               w1=option1_report,
                               w2=option2_report,
                               summary=contract_summary,
                               pdf_url=pdf_display_url,
                               can_go_prev=can_go_prev,
                               can_go_next=can_go_next,
                               voted_option=voted_option_for_this_pair,
                               is_marked_unclear=is_marked_unclear,
                               progress_info=progress_info)

    except Exception as e:
        log_error("show_vote_item", e,
                  f"navigation_action: {navigation_action}")
        flash("An error occurred loading the contract. Please try again.", "error")
        return redirect(url_for("index"))


def find_next_pair_to_vote_on(voted_pairs):
    """Select a random contract (avoiding the last one if possible) and a random unvoted pair."""
    try:
        last_contract_id = session.get('last_contract_id')

        # Build list of (contract_id, remaining_pairs)
        contract_candidates = []
        for contract_id in contract_ids:
            contract_voted_pairs = voted_pairs.get(contract_id, set())
            if isinstance(contract_voted_pairs, list):
                contract_voted_pairs = set(contract_voted_pairs)

            remaining_pairs = get_remaining_comparisons_for_contract(
                contract_id, contract_voted_pairs)
            if remaining_pairs:
                contract_candidates.append((contract_id, remaining_pairs))

        if not contract_candidates:
            return None, None

        # If there is more than one candidate contract, try to avoid repeating the last contract shown
        if last_contract_id and len(contract_candidates) > 1:
            contract_candidates_no_repeat = [
                c for c in contract_candidates if c[0] != last_contract_id]
            if contract_candidates_no_repeat:
                contract_candidates = contract_candidates_no_repeat

        # Choose a random contract, then a random pair within that contract
        chosen_contract_id, remaining_pairs = random.choice(
            contract_candidates)
        chosen_pair = random.choice(remaining_pairs)

        return chosen_contract_id, chosen_pair
    except Exception as e:
        log_error("find_next_pair_to_vote_on", e)
        return None, None


def find_previous_pair(voted_pairs):
    """Return the last pair the user voted on so that the 'Previous' button shows it."""
    try:
        vote_history = session.get('vote_history', [])
        if vote_history:
            last_vote = vote_history[-1]
            option1_key = last_vote.get('option1_key')
            option2_key = last_vote.get('option2_key')
            if option1_key and option2_key:
                return last_vote['contract_id'], (option1_key, option2_key)

        # Fallback to random next pair if no history is present
        return find_next_pair_to_vote_on(voted_pairs)
    except Exception as e:
        log_error("find_previous_pair", e)
        return None, None


def has_previous_pair(voted_pairs):
    """Check if there are any previous pairs that can be navigated to"""
    try:
        completed_count = sum(len(pairs) if isinstance(pairs, (set, list)) else 0
                              for pairs in voted_pairs.values())
        return completed_count > 0
    except Exception:
        return False


def has_next_pair(voted_pairs):
    """Check if there are more pairs to vote on"""
    try:
        total_expected = get_total_expected_comparisons()
        completed_count = sum(len(pairs) if isinstance(pairs, (set, list)) else 0
                              for pairs in voted_pairs.values())
        return completed_count < total_expected
    except Exception:
        return False


@app.route("/submit_vote", methods=["POST"])
@require_session
def submit_vote():
    try:
        app.logger.info("=== SUBMIT_VOTE START ===")
        if not session.get("info_acknowledged"):
            app.logger.info("INFO NOT ACKNOWLEDGED - redirecting")
            return redirect(url_for("index"))

        # Validate form data
        form_contract_id = request.form.get("contract_id", "").strip()
        form_option1_key = request.form.get("option1_key", "").strip()
        form_option2_key = request.form.get("option2_key", "").strip()
        form_pair_identifier = request.form.get("pair_identifier", "").strip()
        final_choice = request.form.get("final_choice", "").strip()
        app.logger.info(
            f"Form data: contract_id={form_contract_id}, option1_key={form_option1_key}, option2_key={form_option2_key}, pair_id={form_pair_identifier}, final_choice={final_choice}")

        if not form_contract_id or not final_choice or not form_option1_key or not form_option2_key or not form_pair_identifier:
            app.logger.info("INVALID FORM DATA - redirecting")
            flash("Invalid vote submission. Please try again.", "error")
            return redirect(url_for("show_vote_item"))

        log_user_action(session['user_session_id'], 'vote_submission',
                        f"contract: {form_contract_id}, choice: {final_choice}")

        # Use the form data directly since it contains the exact pair that was displayed
        option1_key = form_option1_key
        option2_key = form_option2_key
        pair_identifier = form_pair_identifier
        app.logger.info(
            f"Using form data - Option keys: {option1_key}, {option2_key}, pair_id: {pair_identifier}")

        winner_key = None

        if final_choice == "unclear":
            winner_key = "UNCLEAR"
        elif final_choice == "w1":
            winner_key = option1_key
        elif final_choice == "w2":
            winner_key = option2_key
        else:
            app.logger.info("INVALID CHOICE - redirecting")
            flash("Invalid vote choice. Please select an option.", "error")
            return redirect(url_for("show_vote_item"))

        app.logger.info(f"Winner key determined: {winner_key}")

        # Validate contract exists
        contract_data_file = DATA_DIR / f"{form_contract_id}.json"
        if not contract_data_file.exists():
            app.logger.info("CONTRACT FILE NOT EXISTS - redirecting")
            flash("Contract data not found. Please try again.", "error")
            return redirect(url_for("show_vote_item", navigation_action="next"))

        app.logger.info("About to get metadata and save vote")
        # Get metadata
        user_agent = request.headers.get('User-Agent', '')
        ip_address = request.environ.get(
            'HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', ''))
        session_start_time = session.get('session_start_time')

        # Save vote with pair identifier
        vote_saved = False
        try:
            app.logger.info(
                f"Attempting to save vote: {session['user_session_id']}, {form_contract_id}, {pair_identifier}, {winner_key}")
            db = get_db()
            cursor = db.cursor()
            user_session_id = session['user_session_id']
            contract_id_to_vote_on = form_contract_id

            if USE_POSTGRES:
                app.logger.info("Using PostgreSQL for vote save")
                cursor.execute("""
                    INSERT INTO votes (user_session, contract_id, pair_identifier, option1_key, option2_key, winner_key, 
                                     user_agent, ip_address, session_start_time) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_session, contract_id, pair_identifier) DO UPDATE SET 
                    option1_key = EXCLUDED.option1_key,
                    option2_key = EXCLUDED.option2_key,
                    winner_key = EXCLUDED.winner_key,
                    user_agent = EXCLUDED.user_agent,
                    ip_address = EXCLUDED.ip_address,
                    session_start_time = EXCLUDED.session_start_time
                """, (user_session_id, contract_id_to_vote_on, pair_identifier, option1_key, option2_key, winner_key,
                      user_agent, ip_address, session_start_time))
            else:
                app.logger.info("Using SQLite for vote save")
                cursor.execute("""
                    INSERT OR REPLACE INTO votes (user_session, contract_id, pair_identifier, option1_key, option2_key, winner_key,
                                                user_agent, ip_address, session_start_time) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_session_id, contract_id_to_vote_on, pair_identifier, option1_key, option2_key, winner_key,
                      user_agent, ip_address, session_start_time))

            app.logger.info("About to commit vote to database")
            db.commit()
            app.logger.info("Vote committed successfully to database")
            vote_saved = True
            global vote_count
            vote_count += 1

            if DEBUG_MODE:
                app.logger.info(
                    f"Vote saved for {contract_id_to_vote_on} pair {pair_identifier} by {user_session_id} - Winner: {winner_key}")

            record_monitoring_metric('vote_submitted', 1, {
                'contract_id': contract_id_to_vote_on,
                'pair_identifier': pair_identifier,
                'session_id': user_session_id,
                'winner': winner_key
            })

        except Exception as e:
            app.logger.error(f"Database error in submit_vote: {str(e)}")
            app.logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            app.logger.error(f"Full traceback: {traceback.format_exc()}")
            log_error("submit_vote_db", e)
            flash("Failed to save vote. Please try again.", "error")
            return redirect(url_for("show_vote_item"))

        if vote_saved:
            # Update session tracking for voted pairs
            voted_pairs = session.get('voted_pairs', {})
            if contract_id_to_vote_on not in voted_pairs:
                voted_pairs[contract_id_to_vote_on] = set()
            elif isinstance(voted_pairs[contract_id_to_vote_on], list):
                voted_pairs[contract_id_to_vote_on] = set(
                    voted_pairs[contract_id_to_vote_on])

            voted_pairs[contract_id_to_vote_on].add(pair_identifier)
            session['voted_pairs'] = {k: list(v) if isinstance(v, set) else v
                                      for k, v in voted_pairs.items()}

            # Update session history efficiently
            vote_history = session.get('vote_history', [])
            updated_history = False

            for i, history_item in enumerate(vote_history):
                if (history_item['contract_id'] == contract_id_to_vote_on and
                        history_item.get('pair_identifier') == pair_identifier):
                    vote_history[i]['winner'] = winner_key
                    updated_history = True
                    break

            if not updated_history:
                vote_history.append({
                    'contract_id': contract_id_to_vote_on,
                    'pair_identifier': pair_identifier,
                    'option1_key': option1_key,
                    'option2_key': option2_key,
                    'winner': winner_key,
                    'pdf_filename': session.get('current_contract_for_vote', {}).get('pdf_filename')
                })

            session['vote_history'] = vote_history
            # Remember the last contract voted on to help randomize the next selection
            session['last_contract_id'] = contract_id_to_vote_on
            session.modified = True

            # Check if there are more pairs to vote on
            if has_next_pair(voted_pairs):
                return redirect(url_for("show_vote_item", navigation_action="next"))
            else:
                return redirect(url_for("all_done_page"))
        else:
            flash("An error occurred submitting your vote. Please try again.", "error")
            return redirect(url_for("show_vote_item"))

    except Exception as e:
        log_error("submit_vote", e)
        flash("An error occurred submitting your vote. Please try again.", "error")
        return redirect(url_for("show_vote_item"))

# New route to serve PDF files


@app.route('/contract_pdfs/<path:filename>')
def serve_contract_pdf(filename):
    try:
        # Validate filename to prevent path traversal
        if '..' in filename or filename.startswith('/'):
            app.logger.warning(f"Invalid PDF filename requested: {filename}")
            return "Invalid filename", 400

        pdf_path = Path(app.root_path) / PDF_DIR_NAME / filename
        if not pdf_path.exists():
            app.logger.warning(f"PDF file not found: {pdf_path}")
            return "PDF not found", 404

        log_user_action(session.get('user_session_id',
                        'unknown'), 'pdf_view', filename)
        return send_from_directory(Path(app.root_path) / PDF_DIR_NAME, filename)
    except Exception as e:
        log_error("serve_contract_pdf", e, f"filename: {filename}")
        return "Error serving PDF", 500


@app.route("/history")
@require_session
def view_history():
    try:
        if not session.get("info_acknowledged"):
            return redirect(url_for("index"))

        log_user_action(session['user_session_id'], 'history_view')

        history = session.get('vote_history', [])
        enriched_history = []

        for item in history:
            try:
                # Reload contract data for each history item
                contract_data_file = DATA_DIR / f"{item['contract_id']}.json"
                if contract_data_file.exists():
                    with optimized_file_operation(open, contract_data_file, 'r') as f:
                        data = json.load(f)

                    # Get the specific options that were compared
                    option1_key = item.get('option1_key')
                    option2_key = item.get('option2_key')

                    if option1_key and option2_key and option1_key in data and option2_key in data:
                        w1 = data[option1_key].get(
                            "final_report", "No report available")
                        w2 = data[option2_key].get(
                            "final_report", "No report available")
                        pair_description = f"Option {option1_key} vs Option {option2_key}"
                    else:
                        # Fallback for old format or missing data
                        available_keys = [
                            k for k in data.keys() if k.isdigit()]
                        if len(available_keys) >= 2:
                            option1_key = available_keys[0]
                            option2_key = available_keys[1]
                            w1 = data[option1_key].get(
                                "final_report", "No report available")
                            w2 = data[option2_key].get(
                                "final_report", "No report available")
                            pair_description = f"Option {option1_key} vs Option {option2_key}"
                        else:
                            w1 = w2 = "Contract data unavailable"
                            pair_description = "Unknown comparison"

                    summary = get_contract_summary(item['contract_id'])
                else:
                    w1 = w2 = "Contract file not found"
                    summary = "Summary unavailable"
                    pair_description = "Unknown comparison"
            except Exception as e:
                app.logger.error(
                    f"Error loading history for {item['contract_id']}: {e}")
                w1 = w2 = "Error loading contract"
                summary = "Summary unavailable"
                pair_description = "Unknown comparison"

            pdf_url = None
            if item.get('pdf_filename'):
                pdf_path_check = Path(app.root_path) / \
                    PDF_DIR_NAME / item['pdf_filename']
                if pdf_path_check.exists():
                    pdf_url = url_for('serve_contract_pdf',
                                      filename=item['pdf_filename'])
                else:
                    app.logger.warning(
                        f"PDF {item['pdf_filename']} for history item {item['contract_id']} not found")

            enriched_history.append({
                'contract_id': item['contract_id'],
                'pair_identifier': item.get('pair_identifier', 'N/A'),
                'pair_description': pair_description,
                'winner': item['winner'],
                'w1': w1,
                'w2': w2,
                'summary': summary,
                'pdf_filename': item.get('pdf_filename'),
                'pdf_url': pdf_url
            })

        return render_template("history.html", history=enriched_history)
    except Exception as e:
        log_error("view_history", e)
        flash("Error loading voting history.", "error")
        return redirect(url_for("index"))


@app.route("/history/view/<contract_id_to_review>")
@require_session
def review_vote_from_history(contract_id_to_review):
    try:
        if not session.get("info_acknowledged"):
            return redirect(url_for("index"))

        # Validate contract ID
        if not contract_id_to_review or '..' in contract_id_to_review:
            return redirect(url_for('view_history'))

        log_user_action(session['user_session_id'],
                        'history_review', contract_id_to_review)

        history = session.get('vote_history', [])
        vote_to_review = None
        for vote in history:
            if vote['contract_id'] == contract_id_to_review:
                vote_to_review = vote
                break

        if not vote_to_review:
            flash("Vote not found in history.", "error")
            return redirect(url_for('view_history'))

        # Reload contract data for the review
        try:
            contract_data_file = DATA_DIR / f"{contract_id_to_review}.json"
            if contract_data_file.exists():
                with optimized_file_operation(open, contract_data_file, 'r') as f:
                    data = json.load(f)
                available_keys = [k for k in data.keys() if k.isdigit()]
                if len(available_keys) >= 2:
                    option1_key = available_keys[0]
                    option2_key = available_keys[1]
                    w1 = data[option1_key].get(
                        "final_report", "No report available")
                    w2 = data[option2_key].get(
                        "final_report", "No report available")
                    summary = get_contract_summary(contract_id_to_review)

                    # Convert the stored winner key back to w1/w2 format for template
                    winner_for_template = None
                    if vote_to_review['winner'] == option1_key:
                        winner_for_template = 'w1'
                    elif vote_to_review['winner'] == option2_key:
                        winner_for_template = 'w2'
                    else:
                        winner_for_template = vote_to_review['winner']
                else:
                    w1 = w2 = "Contract data unavailable"
                    summary = "Summary unavailable"
                    winner_for_template = vote_to_review['winner']
            else:
                w1 = w2 = "Contract file not found"
                summary = "Summary unavailable"
                winner_for_template = vote_to_review['winner']
        except Exception as e:
            app.logger.error(
                f"Error loading contract for review {contract_id_to_review}: {e}")
            w1 = w2 = "Error loading contract"
            summary = "Summary unavailable"
            winner_for_template = vote_to_review['winner']

        pdf_url = None
        if vote_to_review.get('pdf_filename'):
            pdf_path_check = Path(app.root_path) / \
                PDF_DIR_NAME / vote_to_review['pdf_filename']
            if pdf_path_check.exists():
                pdf_url = url_for('serve_contract_pdf',
                                  filename=vote_to_review['pdf_filename'])

        vote_data = {
            'contract_id': contract_id_to_review,
            'winner': winner_for_template,
            'w1': w1,
            'w2': w2,
            'summary': summary,
            'pdf_filename': vote_to_review.get('pdf_filename')
        }

        return render_template("review_vote.html", vote=vote_data, pdf_url=pdf_url)
    except Exception as e:
        log_error("review_vote_from_history", e,
                  f"contract_id: {contract_id_to_review}")
        flash("Error loading vote review.", "error")
        return redirect(url_for('view_history'))


@app.route("/all_done")
def all_done_page():
    try:
        log_user_action(session.get('user_session_id',
                        'unknown'), 'all_done_visit')

        # Record completion metrics
        if session.get('user_session_id'):
            record_monitoring_metric('user_completed_all', 1, {
                'session_id': session['user_session_id'],
                'votes_cast': len(session.get('vote_history', []))
            })

        return render_template("all_done.html")
    except Exception as e:
        log_error("all_done_page", e)
        return render_template("all_done.html")

# Admin and monitoring routes


@app.route("/admin")
def admin_dashboard():
    try:
        # Simple auth check (you might want to implement proper authentication)
        auth_key = request.args.get('key')
        if auth_key != os.environ.get('ADMIN_KEY', 'thesis_admin_2024'):
            return "Unauthorized", 401

        # Get overall statistics
        db = get_db()
        cursor = db.cursor()

        # Total votes (now pair votes) - excluding AI agents
        cursor.execute(
            "SELECT COUNT(*) as count FROM votes WHERE user_agent NOT LIKE 'AI_%' AND user_agent NOT LIKE '%ai_gemini%'")
        total_votes = cursor.fetchone()['count']

        # Unique users - excluding AI agents
        cursor.execute(
            "SELECT COUNT(DISTINCT user_session) as count FROM votes WHERE user_agent NOT LIKE 'AI_%' AND user_agent NOT LIKE '%ai_gemini%'")
        unique_users = cursor.fetchone()['count']

        # Three-way comparison statistics
        cursor.execute("SELECT COUNT(*) as count FROM threeway_votes")
        threeway_total_votes = cursor.fetchone()['count']

        cursor.execute(
            "SELECT COUNT(DISTINCT user_session) as count FROM threeway_votes")
        threeway_unique_users = cursor.fetchone()['count']

        cursor.execute("""
            SELECT winner_source, COUNT(*) as count 
            FROM threeway_votes 
            GROUP BY winner_source 
            ORDER BY count DESC
        """)
        threeway_winner_stats = cursor.fetchall()

        cursor.execute("""
            SELECT 
                AVG(fine_tuned_committee_clarity) as avg_ftc_clarity,
                AVG(fine_tuned_committee_legal) as avg_ftc_legal,
                AVG(fine_tuned_committee_reasoning) as avg_ftc_reasoning,
                AVG(fine_tuned_committee_alignment) as avg_ftc_alignment,
                AVG(single_stage_clarity) as avg_ss_clarity,
                AVG(single_stage_legal) as avg_ss_legal,
                AVG(single_stage_reasoning) as avg_ss_reasoning,
                AVG(single_stage_alignment) as avg_ss_alignment,
                AVG(non_fine_tuned_committee_clarity) as avg_nftc_clarity,
                AVG(non_fine_tuned_committee_legal) as avg_nftc_legal,
                AVG(non_fine_tuned_committee_reasoning) as avg_nftc_reasoning,
                AVG(non_fine_tuned_committee_alignment) as avg_nftc_alignment
            FROM threeway_votes
        """)
        rating_averages = cursor.fetchone()

        # Votes per contract (now shows pair counts) - excluding AI agents
        cursor.execute("""
            SELECT contract_id, COUNT(*) as vote_count, 
                   COUNT(DISTINCT pair_identifier) as unique_pairs
            FROM votes 
            WHERE user_agent NOT LIKE 'AI_%' AND user_agent NOT LIKE '%ai_gemini%'
            GROUP BY contract_id 
            ORDER BY vote_count DESC
        """)
        votes_per_contract = cursor.fetchall()

        # Votes per pair across all contracts - excluding AI agents
        cursor.execute("""
            SELECT pair_identifier, COUNT(*) as vote_count 
            FROM votes 
            WHERE user_agent NOT LIKE 'AI_%' AND user_agent NOT LIKE '%ai_gemini%'
            GROUP BY pair_identifier 
            ORDER BY vote_count DESC
        """)
        votes_per_pair = cursor.fetchall()

        # Recent activity - excluding AI agents
        cursor.execute("""
            SELECT user_session, contract_id, pair_identifier, winner_key, voted_at 
            FROM votes 
            WHERE user_agent NOT LIKE 'AI_%' AND user_agent NOT LIKE '%ai_gemini%'
            ORDER BY voted_at DESC 
            LIMIT 20
        """)
        recent_votes = cursor.fetchall()

        # Unclear votes per pair - excluding AI agents
        cursor.execute("""
            SELECT contract_id, pair_identifier, COUNT(*) as unclear_count 
            FROM votes 
            WHERE winner_key = 'UNCLEAR' 
            AND user_agent NOT LIKE 'AI_%' AND user_agent NOT LIKE '%ai_gemini%'
            GROUP BY contract_id, pair_identifier 
            ORDER BY unclear_count DESC
        """)
        unclear_votes = cursor.fetchall()

        # Calculate completion statistics - excluding AI agents
        total_expected_pairs = get_total_expected_comparisons()
        cursor.execute(
            "SELECT COUNT(DISTINCT contract_id || '|' || pair_identifier) as unique_pairs FROM votes WHERE user_agent NOT LIKE 'AI_%' AND user_agent NOT LIKE '%ai_gemini%'")
        unique_pairs_voted = cursor.fetchone()['unique_pairs']

        completion_percentage = (
            unique_pairs_voted / total_expected_pairs * 100) if total_expected_pairs > 0 else 0

        # System stats
        uptime = datetime.now() - app_start_time

        stats = {
            'total_votes': total_votes,
            'unique_users': unique_users,
            'total_contracts': len(contract_ids),
            'total_expected_pairs': total_expected_pairs,
            'unique_pairs_voted': unique_pairs_voted,
            'completion_percentage': round(completion_percentage, 2),
            'votes_per_contract': votes_per_contract,
            'votes_per_pair': votes_per_pair,
            'recent_votes': recent_votes,
            'unclear_votes': unclear_votes,
            'threeway_total_votes': threeway_total_votes,
            'threeway_unique_users': threeway_unique_users,
            'threeway_winner_stats': threeway_winner_stats,
            'rating_averages': rating_averages,
            'uptime': str(uptime),
            'error_count': error_count,
            'vote_count': vote_count
        }

        return render_template("admin.html", stats=stats)
    except Exception as e:
        log_error("admin_dashboard", e)
        return f"Error loading admin dashboard: {str(e)}", 500


@app.route("/admin/export")
def admin_export():
    try:
        auth_key = request.args.get('key')
        if auth_key != os.environ.get('ADMIN_KEY', 'thesis_admin_2024'):
            return "Unauthorized", 401

        # Export all data including AI agent votes for comprehensive analysis
        export_file = backup_votes_to_csv()
        if export_file:
            return send_from_directory(BACKUP_DIR, export_file.name, as_attachment=True)
        else:
            return "Export failed", 500
    except Exception as e:
        log_error("admin_export", e)
        return f"Export error: {str(e)}", 500


@app.route("/admin/status")
def admin_status():
    """JSON endpoint for monitoring"""
    try:
        auth_key = request.args.get('key')
        if auth_key != os.environ.get('ADMIN_KEY', 'thesis_admin_2024'):

            return jsonify({"error": "Unauthorized"}), 401

        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            "SELECT COUNT(*) as count FROM votes WHERE user_agent NOT LIKE 'AI_%' AND user_agent NOT LIKE '%ai_gemini%'")
        total_votes = cursor.fetchone()['count']

        cursor.execute(
            "SELECT COUNT(DISTINCT user_session) as count FROM votes WHERE user_agent NOT LIKE 'AI_%' AND user_agent NOT LIKE '%ai_gemini%'")
        unique_users = cursor.fetchone()['count']

        # Calculate pair completion - excluding AI agents
        total_expected_pairs = get_total_expected_comparisons()
        cursor.execute(
            "SELECT COUNT(DISTINCT contract_id || '|' || pair_identifier) as unique_pairs FROM votes WHERE user_agent NOT LIKE 'AI_%' AND user_agent NOT LIKE '%ai_gemini%'")
        unique_pairs_voted = cursor.fetchone()['unique_pairs']
        completion_percentage = (
            unique_pairs_voted / total_expected_pairs * 100) if total_expected_pairs > 0 else 0

        return jsonify({
            "status": "online",
            "total_votes": total_votes,
            "unique_users": unique_users,
            "total_expected_pairs": total_expected_pairs,
            "unique_pairs_voted": unique_pairs_voted,
            "completion_percentage": round(completion_percentage, 2),
            "uptime_seconds": (datetime.now() - app_start_time).total_seconds(),
            "error_count": error_count,
            "contracts_loaded": len(contract_ids)
        })
    except Exception as e:
        log_error("admin_status", e)
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Basic health checks
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT 1")

        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "contracts_loaded": len(contract_ids)
        })
    except Exception as e:
        log_error("health_check", e)
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

# Recovery endpoint for lost sessions


@app.route("/recover_session", methods=["GET", "POST"])
def recover_session():
    try:
        if request.method == "POST":
            session_id = request.form.get("session_id", "").strip()
            if session_id:
                # Validate session exists in database
                db = get_db()
                cursor = db.cursor()
                cursor.execute(
                    "SELECT DISTINCT contract_id, winner_key FROM votes WHERE user_session = ?", (session_id,))
                votes = cursor.fetchall()

                if votes:
                    # Restore session
                    session['user_session_id'] = session_id
                    session['info_acknowledged'] = True
                    session.permanent = True
                    session['vote_history'] = [
                        {'contract_id': vote['contract_id'],
                            'winner': vote['winner_key']}
                        for vote in votes
                    ]
                    session.modified = True

                    log_user_action(session_id, 'session_recovered')
                    flash("Session recovered successfully!", "success")
                    return redirect(url_for("show_vote_item"))
                else:
                    flash("Session ID not found.", "error")

        return render_template("recover_session.html")
    except Exception as e:
        log_error("recover_session", e)
        flash("Error recovering session.", "error")
        return redirect(url_for("index"))


def generate_all_pairs(available_keys):
    """Generate all possible pairs from available keys (should be 0,1,2,3)"""
    if len(available_keys) < 2:
        return []

    # Ensure we have exactly 4 options and they are the expected ones
    expected_keys = ['0', '1', '2', '3']
    if len(available_keys) >= 4:
        # Use the first 4 keys if more are available
        keys_to_use = sorted(available_keys)[:4]
    else:
        keys_to_use = sorted(available_keys)

    # Generate all possible pairs
    pairs = list(combinations(keys_to_use, 2))
    return pairs


def get_pair_identifier(option1_key, option2_key):
    """Create a consistent identifier for a pair"""
    return f"{min(option1_key, option2_key)}vs{max(option1_key, option2_key)}"


def get_total_expected_comparisons():
    """Calculate total expected comparisons across all contracts"""
    return len(contract_ids) * 6  # 6 pairs per contract


def get_remaining_comparisons_for_contract(contract_id, voted_pairs):
    """Get remaining pairs to vote on for a specific contract"""
    try:
        contract_data_file = DATA_DIR / f"{contract_id}.json"
        if not contract_data_file.exists():
            return []

        with optimized_file_operation(open, contract_data_file, 'r') as f:
            data = json.load(f)

        available_keys = [k for k in data.keys() if k.isdigit()]
        all_pairs = generate_all_pairs(available_keys)

        # Filter out already voted pairs
        remaining_pairs = []
        for pair in all_pairs:
            pair_id = get_pair_identifier(pair[0], pair[1])
            if pair_id not in voted_pairs:
                remaining_pairs.append(pair)

        return remaining_pairs
    except Exception as e:
        log_error("get_remaining_comparisons_for_contract",
                  e, f"contract_id: {contract_id}")
        return []


@lru_cache(maxsize=128)
def get_threeway_contract_ids():
    """Cache contract IDs for three-way comparison using fuzzy matching"""
    global _threeway_contract_ids
    if _threeway_contract_ids is not None:
        return _threeway_contract_ids

    try:
        from difflib import SequenceMatcher

        def normalize_filename(filename):
            """Normalize filenames for better matching"""
            # Remove common suffixes and normalize punctuation
            normalized = filename.lower()
            normalized = normalized.replace('_pdf_report', '')
            # Convert underscores to dots for consistency
            normalized = normalized.replace('_', '.')
            # Normalize exhibit references
            normalized = normalized.replace('-ex-', '_ex_')
            return normalized

        def fuzzy_match(a, b, threshold=0.85):
            """Check if two filenames are similar enough"""
            norm_a = normalize_filename(a)
            norm_b = normalize_filename(b)
            return SequenceMatcher(None, norm_a, norm_b).ratio() >= threshold

        # Get all contract files from each directory
        all_contracts = {}

        for source_name, source_dir in THREE_WAY_DATA_DIRS.items():
            if source_dir.exists():
                contracts = []
                for p in source_dir.glob("*.json"):
                    if p.stem != "contract_summaries":
                        contracts.append(p.stem)
                all_contracts[source_name] = contracts
                if DEBUG_MODE:
                    app.logger.info(
                        f"Found {len(contracts)} contracts in {source_name}: {source_dir}")
            else:
                all_contracts[source_name] = []
                if DEBUG_MODE:
                    app.logger.warning(
                        f"Directory {source_name} does not exist: {source_dir}")

        # Check if we have contracts in all three directories
        missing_dirs = [
            name for name, contracts in all_contracts.items() if len(contracts) == 0]
        if missing_dirs:
            if DEBUG_MODE:
                app.logger.error(
                    f"Missing contracts in directories: {missing_dirs}")
                app.logger.error(
                    "Three-way comparison requires contracts in ALL three directories:")
                for name, path in THREE_WAY_DATA_DIRS.items():
                    exists = path.exists()
                    count = len(all_contracts.get(name, []))
                    app.logger.error(
                        f"  {name}: {path} (exists: {exists}, files: {count})")
            return []

        if not all_contracts:
            if DEBUG_MODE:
                app.logger.warning(
                    "No contract directories found, using fallback")
            return []

        # Find contracts that have matches across all three sources using fuzzy matching
        matched_contracts = []

        if DEBUG_MODE:
            app.logger.info(f"All contracts found: {all_contracts}")

        # Use the source with the fewest contracts as the base to minimize comparisons
        base_source = min(all_contracts.keys(),
                          key=lambda k: len(all_contracts[k]))
        base_contracts = all_contracts[base_source]

        if DEBUG_MODE:
            app.logger.info(
                f"Using {base_source} as base source with {len(base_contracts)} contracts")

        for base_contract in base_contracts:
            matches = {base_source: base_contract}

            # Find matches in other sources
            for other_source, other_contracts in all_contracts.items():
                if other_source == base_source:
                    continue

                best_match = None
                best_ratio = 0

                for other_contract in other_contracts:
                    norm_base = normalize_filename(base_contract)
                    norm_other = normalize_filename(other_contract)
                    ratio = SequenceMatcher(
                        None, norm_base, norm_other).ratio()
                    # 85% similarity threshold (raised for better accuracy)
                    if ratio > best_ratio and ratio >= 0.85:
                        best_ratio = ratio
                        best_match = other_contract

                if best_match:
                    matches[other_source] = best_match
                    if DEBUG_MODE:
                        app.logger.info(
                            f"Found match for {base_contract}: {other_source}={best_match} (ratio: {best_ratio:.2f})")
                else:
                    if DEBUG_MODE:
                        app.logger.info(
                            f"No match found for {base_contract} in {other_source}")

            # Only include if we found matches in ALL three sources
            if len(matches) == 3:
                matched_contracts.append(matches)
                if DEBUG_MODE:
                    app.logger.info(f"✓ Complete match set: {matches}")
            else:
                if DEBUG_MODE:
                    app.logger.info(
                        f"✗ Incomplete match (found {len(matches)}/3): {matches}")

        # Convert to a list of tuples (base_id, matches_dict) for consistent access
        _threeway_contract_ids = [(i, matches)
                                  for i, matches in enumerate(matched_contracts)]

        if DEBUG_MODE:
            app.logger.info(
                f"FINAL RESULT: Found {len(_threeway_contract_ids)} matching contract sets across all three sources")
            if len(_threeway_contract_ids) == 0:
                app.logger.warning(
                    "No matching contracts found! This will cause the 'Complete' page to show immediately.")

        return _threeway_contract_ids
    except Exception as e:
        log_error("get_threeway_contract_ids", e)
        return []


def load_threeway_contract_data(contract_match_info):
    """Load contract data from all three sources using fuzzy matched filenames"""
    try:
        contract_idx, matches = contract_match_info
        contract_data = {
            'non_fine_tuned_committee': None,
            'fine_tuned_committee': None,
            'single_stage': None,
            'contract_idx': contract_idx,
            'filenames': matches,  # Store the actual filenames used
            # Use stage2_out filename as display ID
            'display_id': matches.get('stage2_out', matches.get('single_stage', matches.get('fine_tuned_committee', f'contract_{contract_idx}')))
        }

        # Load from fine_tuned_committee using matched filename
        fine_tuned_committee_filename = matches.get('fine_tuned_committee')
        if fine_tuned_committee_filename:
            fine_tuned_committee_file = THREE_WAY_DATA_DIRS['fine_tuned_committee'] / \
                f"{fine_tuned_committee_filename}.json"
            if fine_tuned_committee_file.exists():
                try:
                    with open(fine_tuned_committee_file, 'r') as f:
                        fine_tuned_committee_data = json.load(f)
                    # Assume single variant or take first available
                    if isinstance(fine_tuned_committee_data, dict):
                        if '0' in fine_tuned_committee_data:
                            contract_data['fine_tuned_committee'] = fine_tuned_committee_data['0'].get(
                                'final_report', 'No report available')
                        else:
                            # If it's a direct report
                            contract_data['fine_tuned_committee'] = fine_tuned_committee_data.get(
                                'final_report', 'No report available')
                except Exception as e:
                    log_error("load_fine_tuned_committee_data", e,
                              f"filename: {fine_tuned_committee_filename}")

        # Load from single_stage using matched filename
        single_filename = matches.get('single_stage')
        if single_filename:
            single_file = THREE_WAY_DATA_DIRS['single_stage'] / \
                f"{single_filename}.json"
            if DEBUG_MODE:
                app.logger.info(
                    f"Looking for single_stage file: {single_file}")
            if single_file.exists():
                try:
                    with open(single_file, 'r') as f:
                        single_data = json.load(f)
                    # Handle different JSON structures for single_stage files
                    if isinstance(single_data, dict):
                        if 'analysis' in single_data and 'final_report' in single_data['analysis']:
                            # Single stage structure: {"analysis": {"final_report": "..."}}
                            contract_data['single_stage'] = single_data['analysis'].get(
                                'final_report', 'No report available')
                            if DEBUG_MODE:
                                app.logger.info(
                                    f"Loaded single_stage from analysis structure: {len(contract_data['single_stage'])} chars")
                        elif '0' in single_data:
                            # Committee structure: {"0": {"final_report": "..."}}
                            contract_data['single_stage'] = single_data['0'].get(
                                'final_report', 'No report available')
                            if DEBUG_MODE:
                                app.logger.info(
                                    f"Loaded single_stage from committee structure: {len(contract_data['single_stage'])} chars")
                        else:
                            # Direct structure: {"final_report": "..."}
                            contract_data['single_stage'] = single_data.get(
                                'final_report', 'No report available')
                            if DEBUG_MODE:
                                app.logger.info(
                                    f"Loaded single_stage from direct structure: {len(contract_data['single_stage'])} chars")
                except Exception as e:
                    log_error("load_single_stage_data", e,
                              f"filename: {single_filename}")
            else:
                if DEBUG_MODE:
                    app.logger.warning(
                        f"Single_stage file not found: {single_file}")

        # Load from stage2_out (non-fine-tuned committee) using matched filename (use option "0")
        non_fine_tuned_committee_filename = matches.get('stage2_out')
        if non_fine_tuned_committee_filename:
            non_fine_tuned_committee_file = THREE_WAY_DATA_DIRS['stage2_out'] / \
                f"{non_fine_tuned_committee_filename}.json"
            if DEBUG_MODE:
                app.logger.info(
                    f"Looking for non_fine_tuned_committee file: {non_fine_tuned_committee_file}")
            if non_fine_tuned_committee_file.exists():
                try:
                    with open(non_fine_tuned_committee_file, 'r') as f:
                        non_fine_tuned_committee_data = json.load(f)
                    if '0' in non_fine_tuned_committee_data:
                        contract_data['non_fine_tuned_committee'] = non_fine_tuned_committee_data['0'].get(
                            'final_report', 'No report available')
                        if DEBUG_MODE:
                            app.logger.info(
                                f"Loaded non_fine_tuned_committee from key '0': {len(contract_data['non_fine_tuned_committee'])} chars")
                except Exception as e:
                    log_error("load_non_fine_tuned_committee_data", e,
                              f"filename: {non_fine_tuned_committee_filename}")

        # Add debug logging to identify missing data
        if DEBUG_MODE:
            if not contract_data['fine_tuned_committee']:
                app.logger.warning(
                    f"Missing fine_tuned_committee data for {fine_tuned_committee_filename}")
            if not contract_data['single_stage']:
                app.logger.warning(
                    f"Missing single_stage data for {single_filename}")
            if not contract_data['non_fine_tuned_committee']:
                app.logger.warning(
                    f"Missing non_fine_tuned_committee data for {non_fine_tuned_committee_filename}")

        # Fallback: create dummy data if sources don't exist (for testing)
        if not contract_data['fine_tuned_committee']:
            contract_data['fine_tuned_committee'] = f"Fine-tuned committee analysis for {fine_tuned_committee_filename or 'unknown'} - placeholder text for testing."
        if not contract_data['single_stage']:
            contract_data['single_stage'] = f"Single-stage analysis for {single_filename or 'unknown'} - placeholder text for testing."
        if not contract_data['non_fine_tuned_committee']:
            contract_data[
                'non_fine_tuned_committee'] = f"Non-fine-tuned committee analysis for {non_fine_tuned_committee_filename or 'unknown'} - placeholder text for testing."

        return contract_data

    except Exception as e:
        log_error("load_threeway_contract_data",
                  e, f"contract_match_info: {contract_match_info}")
        return None


def find_next_threeway_contract(user_session_id):
    """Find next contract for three-way comparison that hasn't been voted on by ANY user (randomized selection)"""
    try:
        threeway_contract_matches = get_threeway_contract_ids()
        if DEBUG_MODE:
            app.logger.info(
                f"Available threeway contract matches: {len(threeway_contract_matches)}")

        if not threeway_contract_matches:
            if DEBUG_MODE:
                app.logger.info("No threeway contract matches found")
            return None

        # Get contracts already voted on by ANY user (not just this user)
        db = get_db()
        cursor = db.cursor()

        if USE_POSTGRES:
            cursor.execute("SELECT DISTINCT contract_id FROM threeway_votes")
        else:
            cursor.execute("SELECT DISTINCT contract_id FROM threeway_votes")

        voted_contracts = set(row['contract_id'] for row in cursor.fetchall())

        if DEBUG_MODE:
            app.logger.info(
                f"Contracts already voted on by any user: {voted_contracts}")

        # Build list of unvoted contracts
        unvoted_contracts = []
        for contract_idx, matches in threeway_contract_matches:
            # Use the display_id as contract_id for voting tracking
            display_id = matches.get('stage2_out', matches.get(
                'single_stage', matches.get('fine_tuned_committee', f'contract_{contract_idx}')))

            if display_id not in voted_contracts:
                unvoted_contracts.append((contract_idx, matches))
                if DEBUG_MODE:
                    app.logger.info(
                        f"Available unvoted contract: {display_id}")
            else:
                if DEBUG_MODE:
                    app.logger.info(
                        f"Contract {display_id} already voted on by someone")

        if not unvoted_contracts:
            if DEBUG_MODE:
                app.logger.info("All contracts have been voted on by users")
            return None

        # Randomly select from unvoted contracts
        selected_contract = random.choice(unvoted_contracts)
        contract_idx, matches = selected_contract
        display_id = matches.get('stage2_out', matches.get(
            'single_stage', matches.get('fine_tuned_committee', f'contract_{contract_idx}')))

        if DEBUG_MODE:
            app.logger.info(
                f"Randomly selected unvoted contract: {display_id} (from {len(unvoted_contracts)} available)")

        return selected_contract

    except Exception as e:
        log_error("find_next_threeway_contract", e)
        return None


def has_voted_on_contract_threeway(user_session_id, contract_id):
    """Check if user has already voted on this contract in three-way comparison"""
    try:
        db = get_db()
        cursor = db.cursor()

        if USE_POSTGRES:
            cursor.execute(
                "SELECT 1 FROM threeway_votes WHERE user_session = %s AND contract_id = %s",
                (user_session_id, contract_id)
            )
        else:
            cursor.execute(
                "SELECT 1 FROM threeway_votes WHERE user_session = ? AND contract_id = ?",
                (user_session_id, contract_id)
            )
        return cursor.fetchone() is not None
    except Exception as e:
        log_error("has_voted_on_contract_threeway", e)
        return False


# Three-way comparison routes
@app.route("/compare")
def threeway_index():
    """Entry point for three-way contract comparison"""
    try:
        log_user_action(session.get('user_session_id',
                        'unknown'), 'threeway_index_visit')
        if not session.get("threeway_info_acknowledged"):
            return render_template("threeway_info.html")
        return redirect(url_for("show_threeway_comparison"))
    except Exception as e:
        log_error("threeway_index", e)
        return render_template('error.html',
                               error_type="System Error",
                               error_message="Unable to load the comparison system. Please refresh the page.")


@app.route("/compare/start", methods=["POST"])
def start_threeway_comparison():
    """Start three-way comparison session"""
    try:
        log_user_action('new_user', 'start_threeway_comparison')

        if request.form.get("confirm_checkbox") == "confirmed":
            # Validate session
            if not validate_session():
                return render_template("threeway_info.html", error_message="Session error. Please try again.")

            session["threeway_info_acknowledged"] = True
            session.permanent = True
            session['threeway_vote_history'] = []
            session['threeway_session_start_time'] = datetime.now().isoformat()
            session.modified = True

            log_user_action(session['user_session_id'],
                            'threeway_voting_started')
            record_monitoring_metric('threeway_user_started', 1, {
                                     'session_id': session['user_session_id']})

            return redirect(url_for("show_threeway_comparison"))
        else:
            return render_template("threeway_info.html", error_message="Please acknowledge the information by checking the box.")

    except Exception as e:
        log_error("start_threeway_comparison", e)
        return render_template("threeway_info.html", error_message="An error occurred. Please try again.")


@app.route("/compare/vote")
@app.route("/compare/vote/<navigation_action>")
@require_session
def show_threeway_comparison(navigation_action=None):
    """Show three-way contract comparison page"""
    try:
        if not session.get("threeway_info_acknowledged"):
            return redirect(url_for("threeway_index"))

        log_user_action(session['user_session_id'],
                        'threeway_comparison_visit', f"action: {navigation_action}")

        # Find next contract to compare
        current_contract_match = find_next_threeway_contract(
            session['user_session_id'])

        if DEBUG_MODE:
            app.logger.info(
                f"Current contract match result: {current_contract_match}")

        if not current_contract_match:
            if DEBUG_MODE:
                app.logger.info(
                    f"No more contracts to vote on for user {session['user_session_id']}")
            record_monitoring_metric('threeway_user_completed', 1, {
                                     'session_id': session['user_session_id']})
            return redirect(url_for("threeway_all_done"))

        # Load contract data from all three sources
        contract_data = load_threeway_contract_data(current_contract_match)
        if not contract_data:
            flash("Error loading contract data. Please try again.", "error")
            return redirect(url_for("threeway_index"))

        current_contract_id = contract_data['display_id']

        # Get contract summary (try with display_id first, fallback to stage2 filename)
        contract_summary = get_contract_summary(current_contract_id)
        if contract_summary == 'No summary available for this contract.':
            # Try with stage2 filename
            stage2_filename = contract_data['filenames'].get('stage2_out')
            if stage2_filename:
                contract_summary = get_contract_summary(stage2_filename)

        # Handle PDF with simplified logic (same as original)
        pdf_filename_to_try = f"{current_contract_id}.pdf"
        pdf_display_url = None
        pdf_file_path = Path(app.root_path) / \
            PDF_DIR_NAME / pdf_filename_to_try

        if pdf_file_path.exists():
            pdf_display_url = url_for(
                'serve_contract_pdf', filename=pdf_filename_to_try)
        else:
            # Fallback to example PDF
            example_pdf_filename = "EMERALDHEALTHTHERAPEUTICSINC_06_10_2020-EX-4.5-CONSULTING AGREEMENT - DR. GAETANO MORELLO N.D. INC..PDF"
            example_pdf_path = Path(app.root_path) / \
                PDF_DIR_NAME / example_pdf_filename
            if example_pdf_path.exists():
                pdf_display_url = url_for(
                    'serve_contract_pdf', filename=example_pdf_filename)
                pdf_filename_to_try = example_pdf_filename
            else:
                pdf_filename_to_try = None

        # Store current contract data in session
        session['current_threeway_contract'] = {
            'contract_id': current_contract_id,
            'contract_match': current_contract_match,
            'pdf_url': pdf_display_url,
            'pdf_filename': pdf_filename_to_try
        }
        session.modified = True

        # Calculate progress
        threeway_contract_matches = get_threeway_contract_ids()
        total_contracts = len(threeway_contract_matches)

        # Count completed contracts by checking display IDs
        completed_count = 0
        for contract_idx, matches in threeway_contract_matches:
            display_id = matches.get('stage2_out', matches.get(
                'single_stage', matches.get('fine_tuned_committee', f'contract_{contract_idx}')))
            if has_voted_on_contract_threeway(session['user_session_id'], display_id):
                completed_count += 1
        completed_contracts = completed_count

        progress_info = {
            'completed': completed_contracts,
            'total': total_contracts,
            'contract_id': current_contract_id
        }

        # Check if already voted (for back navigation)
        existing_vote = None
        if has_voted_on_contract_threeway(session['user_session_id'], current_contract_id):
            db = get_db()
            cursor = db.cursor()

            if USE_POSTGRES:
                cursor.execute("""
                    SELECT winner_source,
                           fine_tuned_committee_clarity, fine_tuned_committee_legal, fine_tuned_committee_reasoning, fine_tuned_committee_alignment,
                           single_stage_clarity, single_stage_legal, single_stage_reasoning, single_stage_alignment,
                           non_fine_tuned_committee_clarity, non_fine_tuned_committee_legal, non_fine_tuned_committee_reasoning, non_fine_tuned_committee_alignment
                    FROM threeway_votes 
                    WHERE user_session = %s AND contract_id = %s
                """, (session['user_session_id'], current_contract_id))
            else:
                cursor.execute("""
                    SELECT winner_source,
                           fine_tuned_committee_clarity, fine_tuned_committee_legal, fine_tuned_committee_reasoning, fine_tuned_committee_alignment,
                           single_stage_clarity, single_stage_legal, single_stage_reasoning, single_stage_alignment,
                           non_fine_tuned_committee_clarity, non_fine_tuned_committee_legal, non_fine_tuned_committee_reasoning, non_fine_tuned_committee_alignment
                    FROM threeway_votes 
                    WHERE user_session = ? AND contract_id = ?
                """, (session['user_session_id'], current_contract_id))

            vote_row = cursor.fetchone()
            if vote_row:
                existing_vote = dict(vote_row)

        record_monitoring_metric('threeway_contract_viewed', 1, {
            'contract_id': current_contract_id,
            'session_id': session['user_session_id']
        })

        return render_template("threeway_vote.html",
                               contract_id=current_contract_id,
                               fine_tuned_committee_report=contract_data['fine_tuned_committee'],
                               single_stage_report=contract_data['single_stage'],
                               non_fine_tuned_committee_report=contract_data['non_fine_tuned_committee'],
                               summary=contract_summary,
                               pdf_url=pdf_display_url,
                               progress_info=progress_info,
                               existing_vote=existing_vote)

    except Exception as e:
        log_error("show_threeway_comparison", e)
        flash("An error occurred loading the comparison. Please try again.", "error")
        return redirect(url_for("threeway_index"))


@app.route("/compare/submit", methods=["POST"])
@require_session
def submit_threeway_vote():
    """Submit three-way comparison vote with ratings"""
    try:
        if not session.get("threeway_info_acknowledged"):
            return redirect(url_for("threeway_index"))

        # Validate form data
        contract_id = request.form.get("contract_id", "").strip()
        winner_source = request.form.get("winner_source", "").strip()

        # Get all individual ratings
        rating_fields = {}
        for analysis in ['fine_tuned_committee', 'single_stage', 'non_fine_tuned_committee']:
            for dimension in ['clarity', 'legal', 'reasoning', 'alignment']:
                field_name = f"{analysis}_{dimension}"
                field_value = request.form.get(field_name, "").strip()
                rating_fields[field_name] = field_value

        # Validate required fields
        missing_fields = []
        if not contract_id:
            missing_fields.append("contract_id")
        if not winner_source:
            missing_fields.append("winner_source")

        for field_name, field_value in rating_fields.items():
            if not field_value:
                missing_fields.append(field_name)

        if missing_fields:
            flash("Please complete all fields including all ratings.", "error")
            return redirect(url_for("show_threeway_comparison"))

        # Validate winner source
        if winner_source not in ['fine_tuned_committee', 'single_stage', 'non_fine_tuned_committee']:
            flash("Invalid selection. Please try again.", "error")
            return redirect(url_for("show_threeway_comparison"))

        # Validate ratings (1-5)
        try:
            for field_name, field_value in rating_fields.items():
                rating_value = int(field_value)
                if not (1 <= rating_value <= 5):
                    raise ValueError(
                        f"Rating {field_name} must be between 1 and 5")
                rating_fields[field_name] = rating_value
        except ValueError as e:
            flash(f"Invalid ratings: {str(e)}", "error")
            return redirect(url_for("show_threeway_comparison"))

        log_user_action(session['user_session_id'], 'threeway_vote_submission',
                        f"contract: {contract_id}, winner: {winner_source}")

        # Get metadata
        user_agent = request.headers.get('User-Agent', '')
        ip_address = request.environ.get('HTTP_X_FORWARDED_FOR',
                                         request.environ.get('REMOTE_ADDR', ''))
        session_start_time = session.get('threeway_session_start_time')

        # Get contract data from session or reload
        current_contract_match = session.get(
            'current_threeway_contract', {}).get('contract_match')
        if current_contract_match:
            contract_data = load_threeway_contract_data(current_contract_match)
        else:
            # Fallback: try to find the contract match by ID
            threeway_contract_matches = get_threeway_contract_ids()
            contract_data = None
            for contract_idx, matches in threeway_contract_matches:
                display_id = matches.get('stage2_out', matches.get(
                    'single_stage', matches.get('fine_tuned_committee', f'contract_{contract_idx}')))
                if display_id == contract_id:
                    contract_data = load_threeway_contract_data(
                        (contract_idx, matches))
                    break

        if not contract_data:
            flash("Error loading contract data. Please try again.", "error")
            return redirect(url_for("show_threeway_comparison"))

        # Save vote to database
        try:
            db = get_db()
            cursor = db.cursor()
            user_session_id = session['user_session_id']

            # Get filenames from contract data
            filenames = contract_data.get('filenames', {})
            fine_tuned_committee_filename = filenames.get(
                'fine_tuned_committee', '')
            single_stage_filename = filenames.get('single_stage', '')
            non_fine_tuned_committee_filename = filenames.get('stage2_out', '')

            if USE_POSTGRES:
                cursor.execute("""
                    INSERT INTO threeway_votes 
                    (user_session, contract_id, fine_tuned_committee_filename, single_stage_filename, non_fine_tuned_committee_filename,
                     fine_tuned_committee_source, single_stage_source, non_fine_tuned_committee_source, 
                     winner_source, 
                     fine_tuned_committee_clarity, fine_tuned_committee_legal, fine_tuned_committee_reasoning, fine_tuned_committee_alignment,
                     single_stage_clarity, single_stage_legal, single_stage_reasoning, single_stage_alignment,
                     non_fine_tuned_committee_clarity, non_fine_tuned_committee_legal, non_fine_tuned_committee_reasoning, non_fine_tuned_committee_alignment,
                     user_agent, ip_address, session_start_time) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_session, contract_id) DO UPDATE SET 
                    fine_tuned_committee_filename = EXCLUDED.fine_tuned_committee_filename,
                    single_stage_filename = EXCLUDED.single_stage_filename,
                    non_fine_tuned_committee_filename = EXCLUDED.non_fine_tuned_committee_filename,
                    fine_tuned_committee_source = EXCLUDED.fine_tuned_committee_source,
                    single_stage_source = EXCLUDED.single_stage_source,
                    non_fine_tuned_committee_source = EXCLUDED.non_fine_tuned_committee_source,
                    winner_source = EXCLUDED.winner_source,
                    fine_tuned_committee_clarity = EXCLUDED.fine_tuned_committee_clarity,
                    fine_tuned_committee_legal = EXCLUDED.fine_tuned_committee_legal,
                    fine_tuned_committee_reasoning = EXCLUDED.fine_tuned_committee_reasoning,
                    fine_tuned_committee_alignment = EXCLUDED.fine_tuned_committee_alignment,
                    single_stage_clarity = EXCLUDED.single_stage_clarity,
                    single_stage_legal = EXCLUDED.single_stage_legal,
                    single_stage_reasoning = EXCLUDED.single_stage_reasoning,
                    single_stage_alignment = EXCLUDED.single_stage_alignment,
                    non_fine_tuned_committee_clarity = EXCLUDED.non_fine_tuned_committee_clarity,
                    non_fine_tuned_committee_legal = EXCLUDED.non_fine_tuned_committee_legal,
                    non_fine_tuned_committee_reasoning = EXCLUDED.non_fine_tuned_committee_reasoning,
                    non_fine_tuned_committee_alignment = EXCLUDED.non_fine_tuned_committee_alignment,
                    user_agent = EXCLUDED.user_agent,
                    ip_address = EXCLUDED.ip_address,
                    session_start_time = EXCLUDED.session_start_time
                """, (user_session_id, contract_id, fine_tuned_committee_filename, single_stage_filename, non_fine_tuned_committee_filename,
                      contract_data['fine_tuned_committee'], contract_data['single_stage'], contract_data['non_fine_tuned_committee'],
                      winner_source,
                      rating_fields['fine_tuned_committee_clarity'], rating_fields['fine_tuned_committee_legal'],
                      rating_fields['fine_tuned_committee_reasoning'], rating_fields['fine_tuned_committee_alignment'],
                      rating_fields['single_stage_clarity'], rating_fields['single_stage_legal'],
                      rating_fields['single_stage_reasoning'], rating_fields['single_stage_alignment'],
                      rating_fields['non_fine_tuned_committee_clarity'], rating_fields['non_fine_tuned_committee_legal'],
                      rating_fields['non_fine_tuned_committee_reasoning'], rating_fields['non_fine_tuned_committee_alignment'],
                      user_agent, ip_address, session_start_time))
            else:
                cursor.execute("""
                    INSERT OR REPLACE INTO threeway_votes 
                    (user_session, contract_id, fine_tuned_committee_filename, single_stage_filename, non_fine_tuned_committee_filename,
                     fine_tuned_committee_source, single_stage_source, non_fine_tuned_committee_source,
                     winner_source,
                     fine_tuned_committee_clarity, fine_tuned_committee_legal, fine_tuned_committee_reasoning, fine_tuned_committee_alignment,
                     single_stage_clarity, single_stage_legal, single_stage_reasoning, single_stage_alignment,
                     non_fine_tuned_committee_clarity, non_fine_tuned_committee_legal, non_fine_tuned_committee_reasoning, non_fine_tuned_committee_alignment,
                     user_agent, ip_address, session_start_time) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_session_id, contract_id, fine_tuned_committee_filename, single_stage_filename, non_fine_tuned_committee_filename,
                      contract_data['fine_tuned_committee'], contract_data['single_stage'], contract_data['non_fine_tuned_committee'],
                      winner_source,
                      rating_fields['fine_tuned_committee_clarity'], rating_fields['fine_tuned_committee_legal'],
                      rating_fields['fine_tuned_committee_reasoning'], rating_fields['fine_tuned_committee_alignment'],
                      rating_fields['single_stage_clarity'], rating_fields['single_stage_legal'],
                      rating_fields['single_stage_reasoning'], rating_fields['single_stage_alignment'],
                      rating_fields['non_fine_tuned_committee_clarity'], rating_fields['non_fine_tuned_committee_legal'],
                      rating_fields['non_fine_tuned_committee_reasoning'], rating_fields['non_fine_tuned_committee_alignment'],
                      user_agent, ip_address, session_start_time))

            db.commit()

            if DEBUG_MODE:
                app.logger.info(
                    f"Three-way vote saved: {contract_id} by {user_session_id} - Winner: {winner_source}")

            record_monitoring_metric('threeway_vote_submitted', 1, {
                'contract_id': contract_id,
                'winner_source': winner_source,
                'session_id': user_session_id
            })

            # Update session history
            vote_history = session.get('threeway_vote_history', [])
            history_entry = {
                'contract_id': contract_id,
                'winner_source': winner_source
            }
            # Add all individual ratings to history
            history_entry.update(rating_fields)
            vote_history.append(history_entry)
            session['threeway_vote_history'] = vote_history
            session.modified = True

            # Check if more contracts to vote on
            next_contract = find_next_threeway_contract(user_session_id)
            if next_contract:
                return redirect(url_for("show_threeway_comparison"))
            else:
                return redirect(url_for("threeway_all_done"))

        except Exception as e:
            log_error("submit_threeway_vote_db", e)
            flash("Failed to save vote. Please try again.", "error")
            return redirect(url_for("show_threeway_comparison"))

    except Exception as e:
        log_error("submit_threeway_vote", e)
        flash("An error occurred submitting your vote. Please try again.", "error")
        return redirect(url_for("show_threeway_comparison"))


@app.route("/compare/done")
def threeway_all_done():
    """Three-way comparison completion page"""
    try:
        log_user_action(session.get('user_session_id',
                        'unknown'), 'threeway_all_done_visit')

        if session.get('user_session_id'):
            record_monitoring_metric('threeway_user_completed_all', 1, {
                'session_id': session['user_session_id'],
                'votes_cast': len(session.get('threeway_vote_history', []))
            })

        return render_template("threeway_all_done.html")
    except Exception as e:
        log_error("threeway_all_done", e)
        return render_template("threeway_all_done.html")


@app.route("/compare/debug")
def threeway_debug():
    """Debug endpoint to check three-way comparison setup"""
    try:
        debug_info = {}

        # Check directory existence
        for name, path in THREE_WAY_DATA_DIRS.items():
            debug_info[f"{name}_exists"] = path.exists()
            if path.exists():
                files = list(path.glob("*.json"))
                debug_info[f"{name}_files"] = [
                    f.stem for f in files if f.stem != "contract_summaries"]
                debug_info[f"{name}_count"] = len(debug_info[f"{name}_files"])
            else:
                debug_info[f"{name}_files"] = []
                debug_info[f"{name}_count"] = 0

        # Get matching results
        matches = get_threeway_contract_ids()
        debug_info["total_matches"] = len(matches)
        debug_info["matches"] = matches[:5]  # Show first 5 matches

        return f"<pre>{json.dumps(debug_info, indent=2)}</pre>"
    except Exception as e:
        return f"Error: {str(e)}"


@app.route("/compare/history")
@require_session
def threeway_history():
    """View three-way comparison history"""
    try:
        if not session.get("threeway_info_acknowledged"):
            return redirect(url_for("threeway_index"))

        log_user_action(session['user_session_id'], 'threeway_history_view')

        # Get vote history from database
        db = get_db()
        cursor = db.cursor()

        if USE_POSTGRES:
            cursor.execute("""
                SELECT contract_id, fine_tuned_committee_filename, single_stage_filename, non_fine_tuned_committee_filename,
                       winner_source, 
                       fine_tuned_committee_clarity, fine_tuned_committee_legal, fine_tuned_committee_reasoning, fine_tuned_committee_alignment,
                       single_stage_clarity, single_stage_legal, single_stage_reasoning, single_stage_alignment,
                       non_fine_tuned_committee_clarity, non_fine_tuned_committee_legal, non_fine_tuned_committee_reasoning, non_fine_tuned_committee_alignment,
                       voted_at
                FROM threeway_votes 
                WHERE user_session = %s
                ORDER BY voted_at DESC
            """, (session['user_session_id'],))
        else:
            cursor.execute("""
                SELECT contract_id, fine_tuned_committee_filename, single_stage_filename, non_fine_tuned_committee_filename,
                       winner_source,
                       fine_tuned_committee_clarity, fine_tuned_committee_legal, fine_tuned_committee_reasoning, fine_tuned_committee_alignment,
                       single_stage_clarity, single_stage_legal, single_stage_reasoning, single_stage_alignment,
                       non_fine_tuned_committee_clarity, non_fine_tuned_committee_legal, non_fine_tuned_committee_reasoning, non_fine_tuned_committee_alignment,
                       voted_at
                FROM threeway_votes 
                WHERE user_session = ?
                ORDER BY voted_at DESC
            """, (session['user_session_id'],))

        history = [dict(row) for row in cursor.fetchall()]

        return render_template("threeway_history.html", history=history)
    except Exception as e:
        log_error("threeway_history", e)
        flash("Error loading voting history.", "error")
        return redirect(url_for("threeway_index"))


if __name__ == "__main__":
    app.logger.info("Starting voting application...")
    record_monitoring_metric(
        'app_started', 1, {'timestamp': datetime.now().isoformat()})
    app.run(host="0.0.0.0", port=7860, debug=True)
