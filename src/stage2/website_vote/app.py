from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, g
import json
import csv
import os
from pathlib import Path
import sqlite3

app = Flask(__name__, static_folder='public')
app.secret_key = os.urandom(24)  # Needed for session management

DATABASE = Path("data/votes.db")
DATA_DIR = Path("data/stage2_out")
PDF_DIR_NAME = "data/contracts_pdf"  # Relative to app.root_path
# VOTES_CSV = Path("data/votes.csv") # No longer needed

# Load all contract IDs once from JSON filenames
contract_files = sorted(DATA_DIR.glob("*.json"))
contract_ids = [p.stem for p in contract_files]


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row  # Access columns by name
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

# Create schema.sql file if it doesn't exist (or ensure it's correct)
# For now, I'll define the schema inline for the first run, then we can move to schema.sql


def ensure_schema():
    db_path = DATABASE
    needs_init = not db_path.exists()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if needs_init:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS votes (
            contract_id TEXT PRIMARY KEY,
            winner TEXT NOT NULL,
            voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.commit()
        print("Database initialized.")
    # Check if table exists, as an extra measure for subsequent runs
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='votes';")
    if not cursor.fetchone():
        print("'votes' table missing, re-initializing schema.")
        cursor.execute("""
        CREATE TABLE votes (
            contract_id TEXT PRIMARY KEY,
            winner TEXT NOT NULL,
            voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.commit()
        print("'votes' table created.")
    conn.close()


ensure_schema()  # Call it once at app startup to ensure DB and table exist

# Ensure votes.csv exists with header - NO LONGER NEEDED
# if not VOTES_CSV.exists():
#     VOTES_CSV.parent.mkdir(parents=True, exist_ok=True)
#     with open(VOTES_CSV, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["contract_id", "winner"])  # header


@app.route("/")
def index():
    if not session.get("info_acknowledged"):
        return render_template("info.html")
    return redirect(url_for("show_vote_item"))


@app.route("/start_voting", methods=["POST"])
def start_voting():
    if request.form.get("confirm_checkbox") == "confirmed":
        session["info_acknowledged"] = True
        session.permanent = True  # Make session last longer
        session['vote_history'] = []
        session['presented_cids'] = []
        session['current_cid_idx'] = -1
        session.modified = True
        return redirect(url_for("show_vote_item"))
    else:
        return render_template("info.html", error_message="Please acknowledge the information by checking the box.")


@app.route("/vote_item")
@app.route("/vote_item/<navigation_action>")
def show_vote_item(navigation_action=None):
    if not session.get("info_acknowledged"):
        return redirect(url_for("index"))

    presented_cids = session.get('presented_cids', [])
    current_cid_idx = session.get('current_cid_idx', -1)
    target_idx = current_cid_idx

    if navigation_action == "next":
        target_idx += 1
    elif navigation_action == "prev":
        target_idx -= 1
    elif navigation_action is None:
        if current_cid_idx == -1 and not presented_cids:
            target_idx = 0
        else:
            target_idx = current_cid_idx if current_cid_idx != -1 else 0
    else:
        return redirect(url_for("show_vote_item"))

    if target_idx < 0:
        target_idx = 0

    cid_to_serve = None

    if target_idx < len(presented_cids):
        cid_to_serve = presented_cids[target_idx]
        session['current_cid_idx'] = target_idx
    else:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT contract_id FROM votes")
        voted_globally_cids_tuples = cursor.fetchall()
        voted_globally_cids = {row['contract_id']
                               for row in voted_globally_cids_tuples}

        next_new_cid = None
        if target_idx >= len(presented_cids):
            for potential_cid in contract_ids:
                if potential_cid not in voted_globally_cids and potential_cid not in presented_cids:
                    next_new_cid = potential_cid
                    break

            if next_new_cid:
                cid_to_serve = next_new_cid
                presented_cids.append(next_new_cid)
                session['presented_cids'] = presented_cids
                session['current_cid_idx'] = len(presented_cids) - 1
            elif presented_cids:
                session['current_cid_idx'] = len(presented_cids) - 1
                cid_to_serve = presented_cids[session['current_cid_idx']]
            else:
                return redirect(url_for("all_done_page"))
        else:
            if presented_cids:
                actual_idx_to_serve = current_cid_idx if current_cid_idx != -1 else 0
                if actual_idx_to_serve < len(presented_cids):
                    cid_to_serve = presented_cids[actual_idx_to_serve]
                    session['current_cid_idx'] = actual_idx_to_serve
                else:
                    return redirect(url_for("all_done_page"))
            else:
                return redirect(url_for("all_done_page"))

    if not cid_to_serve:
        return redirect(url_for("all_done_page"))

    session.modified = True

    data = json.load(open(DATA_DIR / f"{cid_to_serve}.json"))
    pdf_filename_to_try = f"{cid_to_serve}.pdf"
    pdf_display_url = None
    pdf_file_path = Path(app.root_path) / PDF_DIR_NAME / pdf_filename_to_try
    if pdf_file_path.exists():
        pdf_display_url = url_for(
            'serve_contract_pdf', filename=pdf_filename_to_try)
    else:
        example_pdf_filename = "EMERALDHEALTHTHERAPEUTICSINC_06_10_2020-EX-4.5-CONSULTING AGREEMENT - DR. GAETANO MORELLO N.D. INC..PDF"
        example_pdf_path = Path(app.root_path) / \
            PDF_DIR_NAME / example_pdf_filename
        if example_pdf_path.exists():
            pdf_display_url = url_for(
                'serve_contract_pdf', filename=example_pdf_filename)
            pdf_filename_to_try = example_pdf_filename
        else:
            pdf_filename_to_try = None
            print(
                f"Warning: PDF for {cid_to_serve} (tried {pdf_filename_to_try}) and example PDF not found.")

    session['current_contract_for_vote'] = {
        'contract_id': cid_to_serve,
        'w1': data["w1"]["final_report"],
        'w2': data["w2"]["final_report"],
        'pdf_url': pdf_display_url,
        'pdf_filename': pdf_filename_to_try
    }

    can_go_prev = session.get('current_cid_idx', 0) > 0

    can_go_next = False
    current_idx_val = session.get('current_cid_idx', -1)
    if (current_idx_val + 1) < len(session.get('presented_cids', [])):
        can_go_next = True
    else:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT contract_id FROM votes")
        voted_globally_cids_tuples = cursor.fetchall()
        voted_globally_cids = {row['contract_id']
                               for row in voted_globally_cids_tuples}
        for potential_cid in contract_ids:
            if potential_cid not in voted_globally_cids and potential_cid not in session.get('presented_cids', []):
                can_go_next = True
                break

    voted_option_for_this_contract = None
    if 'vote_history' in session:
        for past_vote in session['vote_history']:
            if past_vote['contract_id'] == cid_to_serve:
                voted_option_for_this_contract = past_vote['winner']
                break

    return render_template("vote.html",
                           contract_id=cid_to_serve,
                           w1=data["w1"]["final_report"],
                           w2=data["w2"]["final_report"],
                           pdf_url=pdf_display_url,
                           can_go_prev=can_go_prev,
                           can_go_next=can_go_next,
                           voted_option=voted_option_for_this_contract)


@app.route("/submit_vote", methods=["POST"])
def submit_vote():
    if not session.get("info_acknowledged"):
        return redirect(url_for("index"))

    form_contract_id = request.form["contract_id"]
    winner = request.form["winner"]
    current_contract_details = session.get('current_contract_for_vote')

    if not current_contract_details or current_contract_details['contract_id'] != form_contract_id:
        return redirect(url_for("show_vote_item", navigation_action="next"))

    db = get_db()
    try:
        cursor = db.cursor()
        contract_id_to_vote_on = current_contract_details['contract_id']
        cursor.execute("INSERT OR REPLACE INTO votes (contract_id, winner) VALUES (?, ?)",
                       (contract_id_to_vote_on, winner))
        db.commit()

        updated_history = False
        if 'vote_history' in session:
            for i, history_item in enumerate(session['vote_history']):
                if history_item['contract_id'] == contract_id_to_vote_on:
                    session['vote_history'][i]['winner'] = winner
                    updated_history = True
                    break

        if not updated_history:
            if 'vote_history' not in session:
                session['vote_history'] = []
            history_entry = {
                'contract_id': contract_id_to_vote_on,
                'winner': winner,
                'w1': current_contract_details['w1'],
                'w2': current_contract_details['w2'],
                'pdf_filename': current_contract_details['pdf_filename']
            }
            session['vote_history'].append(history_entry)

        session.modified = True
    except sqlite3.IntegrityError:
        print(
            f"IntegrityError during INSERT OR REPLACE for {current_contract_details['contract_id']}")
        # pass # It's often better to let the user know or log more verbosely
    except Exception as e:
        print(
            f"Error submitting vote for {current_contract_details['contract_id']}: {e}")
        # pass # Similar to above, consider error handling strategy

    # Check if there are more contracts to vote on
    more_contracts_exist = False
    current_idx_in_session = session.get('current_cid_idx', -1)
    presented_cids_in_session = session.get('presented_cids', [])

    # Check if there's a next item in the already presented list that we haven't "nexted" to yet
    # The current_idx_in_session is the index of the item *just voted on*.
    # So, if current_idx_in_session + 1 is a valid index in presented_cids_in_session,
    # it means there's an item after the current one in the presented list.
    if (current_idx_in_session + 1) < len(presented_cids_in_session):
        more_contracts_exist = True
    else:
        # If we are at the end of the presented list, check for new, unvoted contracts
        # This 'db' is the same one used for submitting the vote, already available.
        cursor = db.cursor()  # Re-use cursor or get a new one if state is an issue
        cursor.execute("SELECT contract_id FROM votes")
        voted_globally_cids_tuples = cursor.fetchall()
        voted_globally_cids = {row['contract_id']
                               for row in voted_globally_cids_tuples}

        for potential_cid in contract_ids:  # contract_ids is the global list of all contract IDs
            if potential_cid not in voted_globally_cids and potential_cid not in presented_cids_in_session:
                more_contracts_exist = True
                break

    if more_contracts_exist:
        return redirect(url_for("show_vote_item", navigation_action="next"))
    else:
        return redirect(url_for("all_done_page"))

# New route to serve PDF files


@app.route('/contract_pdfs/<path:filename>')
def serve_contract_pdf(filename):
    return send_from_directory(Path(app.root_path) / PDF_DIR_NAME, filename)

# Part 2: User should be able to go back in their own session


@app.route("/history")
def view_history():
    if not session.get("info_acknowledged"):
        return redirect(url_for("index"))

    history = session.get('vote_history', [])
    enriched_history = []
    for item in history:
        pdf_url = None
        if item.get('pdf_filename'):
            pdf_path_check = Path(app.root_path) / \
                PDF_DIR_NAME / item['pdf_filename']
            if pdf_path_check.exists():
                pdf_url = url_for('serve_contract_pdf',
                                  filename=item['pdf_filename'])
            else:
                print(
                    f"Warning: PDF {item['pdf_filename']} for history item {item['contract_id']} not found.")
        enriched_history.append({**item, 'pdf_url': pdf_url})

    return render_template("history.html", history=enriched_history)


@app.route("/history/view/<contract_id_to_review>")
def review_vote_from_history(contract_id_to_review):
    if not session.get("info_acknowledged"):
        return redirect(url_for("index"))

    history = session.get('vote_history', [])
    vote_to_review = None
    for vote in history:
        if vote['contract_id'] == contract_id_to_review:
            vote_to_review = vote
            break

    if not vote_to_review:
        return redirect(url_for('view_history'))

    pdf_url = None
    if vote_to_review.get('pdf_filename'):
        pdf_path_check = Path(app.root_path) / PDF_DIR_NAME / \
            vote_to_review['pdf_filename']
        if pdf_path_check.exists():
            pdf_url = url_for('serve_contract_pdf',
                              filename=vote_to_review['pdf_filename'])

    return render_template("review_vote.html", vote=vote_to_review, pdf_url=pdf_url)


@app.route("/all_done")
def all_done_page():
    return render_template("all_done.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
