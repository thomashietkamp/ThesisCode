# Contract Voting Website

A Flask-based web application for conducting contract evaluations where users can vote between different contract analysis options.

## Features

- Interactive voting interface with markdown-rendered contract analysis
- Contract summaries from automated analysis
- PDF document viewing
- Session-based voting history
- Database support for both SQLite (local) and PostgreSQL (production)
- Heroku-ready deployment configuration

## Setup Instructions

### Local Development

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**

   - Ensure your contract data is in `data/stage2_out/` as JSON files
   - Contract summaries should be in `data/stage2_out/contract_summaries.json`
   - PDF files should be in `data/contracts_pdf/`

3. **Run the Application**

   ```bash
   python app.py
   ```

   The application will run on `http://localhost:7860`

### Heroku Deployment

1. **Prerequisites**

   - Install Heroku CLI
   - Create a Heroku account

2. **Setup Heroku App**

   ```bash
   heroku create your-app-name
   heroku addons:create heroku-postgresql:hobby-dev
   ```

3. **Deploy**

   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

4. **Environment Variables**
   The app automatically detects the `DATABASE_URL` environment variable provided by Heroku PostgreSQL.

## Data Structure

### Contract Data Files

Each contract should have a JSON file in `data/stage2_out/` with the structure:

```json
{
  "w1": {
    "final_report": "Contract analysis option 1..."
  },
  "w2": {
    "final_report": "Contract analysis option 2..."
  }
}
```

### Contract Summaries

The `contract_summaries.json` file should contain:

```json
{
  "contract_id": {
    "summary": "Brief summary of the contract..."
  }
}
```

## Database Schema

The application uses a simple database schema:

```sql
CREATE TABLE votes (
    contract_id TEXT PRIMARY KEY,
    winner TEXT NOT NULL,
    voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Configuration

The application automatically configures based on environment:

- **Local Development**: Uses SQLite database (`data/votes.db`)
- **Heroku Production**: Uses PostgreSQL from `DATABASE_URL` environment variable

## File Structure

```
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Procfile              # Heroku deployment configuration
├── templates/            # HTML templates
│   ├── vote.html         # Main voting interface
│   ├── history.html      # Voting history
│   └── review_vote.html  # Vote review page
├── data/
│   ├── stage2_out/       # Contract data JSON files
│   └── contracts_pdf/    # PDF documents
└── public/               # Static files (if any)
```

## Technologies Used

- Flask (Python web framework)
- SQLite/PostgreSQL (database)
- JavaScript (client-side interactivity)
- Marked.js (markdown rendering)
- HTML/CSS (frontend)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is part of academic research and follows institutional guidelines.
