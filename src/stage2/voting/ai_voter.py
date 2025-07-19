import os
import requests
import json
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse
from pathlib import Path
from itertools import combinations
from datetime import datetime
import hashlib
import logging
from dotenv import load_dotenv

load_dotenv()

# ── Model Configuration ──────────────────────────────────────────────────────
# Available models - change _MODEL_ID to switch models
AVAILABLE_MODELS = {
    "gemini-flash": "google/gemini-2.0-flash-exp:free",
    "gemini-pro": "google/gemini-2.5-pro-preview",
    "llama-4": "meta-llama/llama-4-maverick",
    "claude-sonnet": "anthropic/claude-3.5-sonnet",
    "gpt": "openai/o4-mini-2025-04-16",
    "deepseek": "deepseek/deepseek-chat-v3-0324",
    "qwen": "qwen/qwen3-14b:free",
}

# ── Main Configuration ───────────────────────────────────────────────────────
_MODEL_ID = AVAILABLE_MODELS["gpt"]  # Change this to switch models
_BASE_URL = "https://openrouter.ai/api/v1"
_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ── Provider Configuration Toggle ─────────────────────────────────────────────
USE_CUSTOM_PROVIDER = False  # Set to False to disable custom provider settings
CUSTOM_PROVIDER_CONFIG = {
    "sort": "price",
}

if not _API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is required")

_HEADERS = {
    "Authorization": f"Bearer {_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com",
    "X-Title": "Gemini Contract Voter"
}

# ── Database Configuration ─────────────────────────────────────────────────────
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    url = urlparse(DATABASE_URL)
    DB_CONFIG = {
        'host': url.hostname,
        'port': url.port,
        'database': url.path[1:],
        'user': url.username,
        'password': url.password,
        'connect_timeout': 10,
        'application_name': 'ai_voting_gemini'
    }
    USE_POSTGRES = True
    print("Using PostgreSQL database")
else:
    DATABASE = Path("src/stage2/website_vote/data/votes.db")
    DATABASE.parent.mkdir(exist_ok=True)
    USE_POSTGRES = False
    print("Using SQLite database")

# ── Data Configuration ─────────────────────────────────────────────────────────
DATA_DIR = Path("src/stage2/website_vote/data/stage2_out")

# Dynamic session ID and user agent based on model


def get_model_short_name(model_id):
    """Extract a short name from the model ID for session naming"""
    for short_name, full_id in AVAILABLE_MODELS.items():
        if full_id == model_id:
            return short_name
    # Fallback: extract model name from the ID
    if "/" in model_id:
        return model_id.split("/")[1].split(":")[0].replace("-", "_")
    return model_id.replace("-", "_").replace(":", "_")


MODEL_SHORT_NAME = get_model_short_name(_MODEL_ID)
AI_SESSION_ID = f"ai_{MODEL_SHORT_NAME}_" + \
    hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
AI_USER_AGENT = f"AI_Voter_{MODEL_SHORT_NAME}/{_MODEL_ID}"

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Database Functions ─────────────────────────────────────────────────────────


def get_db():
    """Get database connection"""
    if USE_POSTGRES:
        return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    else:
        return sqlite3.connect(DATABASE)


def ensure_database_schema():
    """Ensure the votes table exists with proper schema"""
    try:
        db = get_db()
        cursor = db.cursor()

        if USE_POSTGRES:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS votes (
                    user_session TEXT,
                    contract_id TEXT,
                    pair_identifier TEXT,
                    option1_key TEXT,
                    option2_key TEXT,
                    winner_key TEXT,
                    voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_agent TEXT,
                    ip_address TEXT,
                    session_start_time TIMESTAMP,
                    PRIMARY KEY (user_session, contract_id, pair_identifier)
                );
            """)
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS votes (
                    user_session TEXT,
                    contract_id TEXT,
                    pair_identifier TEXT,
                    option1_key TEXT,
                    option2_key TEXT,
                    winner_key TEXT,
                    voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_agent TEXT,
                    ip_address TEXT,
                    session_start_time TIMESTAMP,
                    PRIMARY KEY (user_session, contract_id, pair_identifier)
                );
            """)

        db.commit()
        db.close()
        logger.info("Database schema ensured")
    except Exception as e:
        logger.error(f"Error ensuring database schema: {e}")
        raise

# ── AI LLM Functions ─────────────────────────────────────────────────────────


def chat_with_gemini(messages, temperature=0.3):
    """Chat completion using Google Gemini 2.5 Flash via OpenRouter"""
    payload = {
        "model": _MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "reasoning": {
            "effort": "medium"
        }
    }

    # Add custom provider configuration if enabled
    if USE_CUSTOM_PROVIDER and CUSTOM_PROVIDER_CONFIG:
        payload["provider"] = CUSTOM_PROVIDER_CONFIG
        logger.debug(f"Using custom provider config: {CUSTOM_PROVIDER_CONFIG}")

    response = requests.post(
        f"{_BASE_URL}/chat/completions",
        headers=_HEADERS,
        data=json.dumps(payload)
    )

    if response.status_code != 200:
        raise Exception(
            f"OpenRouter API error: {response.status_code} - {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()

# ── Contract Processing Functions ─────────────────────────────────────────────


def get_contract_ids():
    """Get all contract IDs from the data directory"""
    if not DATA_DIR.exists():
        raise Exception(f"Data directory {DATA_DIR} does not exist!")

    contract_files = sorted(DATA_DIR.glob("*.json"))
    contract_ids = []
    excluded_files = []

    for p in contract_files:
        if p.stem == "contract_summaries":
            excluded_files.append(p.name)
            continue
        contract_ids.append(p.stem)

    logger.info(f"Found {len(contract_ids)} contracts")
    if excluded_files:
        logger.info(f"Excluded files: {excluded_files}")

    return contract_ids


def load_contract(contract_id):
    """Load contract data from JSON file"""
    contract_file = DATA_DIR / f"{contract_id}.json"
    if not contract_file.exists():
        raise Exception(f"Contract file not found: {contract_file}")

    with open(contract_file, 'r') as f:
        data = json.load(f)

    # Extract the contract options (numbered keys 0, 1, 2, 3)
    options = {}
    for key in ['0', '1', '2', '3']:
        if key in data:
            options[key] = data[key]

    return options


def generate_all_pairs(available_keys):
    """Generate all possible pairs from available keys"""
    if len(available_keys) < 2:
        return []

    # Use the first 4 keys if more are available
    keys_to_use = sorted(available_keys)[:4]

    # Generate all possible pairs (combinations of 2)
    pairs = list(combinations(keys_to_use, 2))
    return pairs


def get_pair_identifier(option1_key, option2_key):
    """Create a consistent identifier for a pair"""
    return f"{min(option1_key, option2_key)}vs{max(option1_key, option2_key)}"

# ── AI Voting Logic ─────────────────────────────────────────────────────────


def create_voting_prompt(option1_text, option2_text, contract_id):
    """Create a prompt for the AI to compare two contract options"""
    prompt = f"""You are an expert legal analyst comparing two versions of a contract clause for: {contract_id}

Please carefully analyze these two contract options and determine which one is better from a legal perspective. Consider factors such as:
- Clarity and precision of language
- Risk allocation and protection for parties
- Enforceability and compliance
- Standard industry practices
- Overall legal soundness

Option 1:
{option1_text}

Option 2:
{option2_text}

Based on your analysis, which option is better? Respond with exactly one of these:
- "Option 1" if the first option is better
- "Option 2" if the second option is better  
- "Unclear" if you cannot determine a clear winner. This option is not recommended.
"""

    return prompt


def vote_on_pair(option1_key, option1_data, option2_key, option2_data, contract_id):
    """Use AI to vote on a pair of contract options"""
    try:
        # Extract the final report from each option
        option1_text = option1_data.get('final_report', 'No report available')
        option2_text = option2_data.get('final_report', 'No report available')

        prompt = create_voting_prompt(option1_text, option2_text, contract_id)

        messages = [
            {"role": "user", "content": prompt}
        ]

        response = chat_with_gemini(messages)
        print(response)

        # Parse the response to determine the winner
        response_lower = response.lower()
        if "option 1" in response_lower:
            winner = option1_key
        elif "option 2" in response_lower:
            winner = option2_key
        elif "unclear" in response_lower:
            winner = "UNCLEAR"

        else:
            # Default to unclear if we can't parse the response
            logger.warning(
                f"Could not parse AI response for {contract_id} {option1_key}vs{option2_key}: {response}")
            winner = "UNCLEAR"

        logger.info(
            f"AI voted: {contract_id} {option1_key}vs{option2_key} -> {winner}")
        return winner, response

    except Exception as e:
        logger.error(
            f"Error voting on pair {option1_key}vs{option2_key} for {contract_id}: {e}")
        return "UNCLEAR", f"Error: {str(e)}"


def save_vote(contract_id, option1_key, option2_key, winner_key):
    """Save vote to database in the same format as the website"""
    try:
        db = get_db()
        cursor = db.cursor()

        pair_identifier = get_pair_identifier(option1_key, option2_key)
        user_agent = AI_USER_AGENT
        ip_address = "127.0.0.1"
        session_start_time = datetime.now()

        if USE_POSTGRES:
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
            """, (AI_SESSION_ID, contract_id, pair_identifier, option1_key, option2_key, winner_key,
                  user_agent, ip_address, session_start_time))
        else:
            cursor.execute("""
                INSERT OR REPLACE INTO votes (user_session, contract_id, pair_identifier, option1_key, option2_key, winner_key,
                                            user_agent, ip_address, session_start_time) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (AI_SESSION_ID, contract_id, pair_identifier, option1_key, option2_key, winner_key,
                  user_agent, ip_address, session_start_time))

        db.commit()
        db.close()
        logger.info(
            f"Saved vote: {contract_id} {pair_identifier} -> {winner_key}")

    except Exception as e:
        logger.error(
            f"Error saving vote for {contract_id} {pair_identifier}: {e}")
        raise


def check_existing_votes():
    """Check how many votes already exist for this AI session"""
    try:
        db = get_db()
        cursor = db.cursor()

        cursor.execute("SELECT COUNT(*) as count FROM votes WHERE user_session = %s" if USE_POSTGRES
                       else "SELECT COUNT(*) as count FROM votes WHERE user_session = ?", (AI_SESSION_ID,))

        result = cursor.fetchone()
        count = result['count'] if USE_POSTGRES else result[0]

        db.close()
        return count

    except Exception as e:
        logger.error(f"Error checking existing votes: {e}")
        return 0


def print_configuration():
    """Print current configuration settings"""
    print(f"🔧 Configuration:")
    print(f"   Model: {_MODEL_ID}")
    print(f"   Model Short Name: {MODEL_SHORT_NAME}")
    print(f"   Session ID: {AI_SESSION_ID}")
    print(f"   User Agent: {AI_USER_AGENT}")
    print(
        f"   Custom Provider: {'Enabled' if USE_CUSTOM_PROVIDER else 'Disabled'}")
    if USE_CUSTOM_PROVIDER and CUSTOM_PROVIDER_CONFIG:
        print(f"   Provider Config: {CUSTOM_PROVIDER_CONFIG}")
    print(f"   Database: {'PostgreSQL' if USE_POSTGRES else 'SQLite'}")

# ── Main Execution ─────────────────────────────────────────────────────────


def main():
    """Main function to run AI voting on all contracts"""
    print(f"🤖 Starting AI voting with Google Gemini 2.5 Flash")
    print_configuration()
    print("-" * 60)

    # Ensure database schema
    ensure_database_schema()

    # Check existing votes
    existing_votes = check_existing_votes()
    if existing_votes > 0:
        print(f"⚠️  Found {existing_votes} existing votes for this session")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Get all contracts
    contract_ids = get_contract_ids()

    total_votes = 0
    total_pairs = 0
    errors = 0

    for i, contract_id in enumerate(contract_ids, 1):
        print(
            f"\n📄 Processing contract {i}/{len(contract_ids)}: {contract_id}")

        try:
            # Load contract data
            contract_options = load_contract(contract_id)
            available_keys = list(contract_options.keys())

            if len(available_keys) < 2:
                logger.warning(
                    f"Contract {contract_id} has less than 2 options, skipping")
                continue

            # Generate all pairs
            pairs = generate_all_pairs(available_keys)
            total_pairs += len(pairs)

            print(
                f"   Found {len(available_keys)} options, generating {len(pairs)} pairs")

            for j, (option1_key, option2_key) in enumerate(pairs, 1):
                pair_id = get_pair_identifier(option1_key, option2_key)
                print(f"   📊 Voting on pair {j}/{len(pairs)}: {pair_id}")

                # Get AI vote
                winner_key, explanation = vote_on_pair(
                    option1_key, contract_options[option1_key],
                    option2_key, contract_options[option2_key],
                    contract_id
                )

                # Save vote to database
                save_vote(contract_id, option1_key, option2_key, winner_key)
                total_votes += 1

                print(f"      Result: {winner_key}")

        except Exception as e:
            logger.error(f"Error processing contract {contract_id}: {e}")
            errors += 1
            continue

    print("\n" + "=" * 60)
    print(f"🎉 AI Voting Complete!")
    print(f"   Contracts processed: {len(contract_ids) - errors}")
    print(f"   Total pairs evaluated: {total_pairs}")
    print(f"   Total votes cast: {total_votes}")
    print(f"   Errors: {errors}")
    print(f"   Session ID: {AI_SESSION_ID}")
    print("=" * 60)


if __name__ == "__main__":
    main()
