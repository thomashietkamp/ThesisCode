# Three-Way Contract Comparison System

This document describes the new three-way contract comparison system that has been added to the voting application.

## Overview

The three-way comparison system allows users to compare three different AI-powered contract analysis approaches:

1. **Committee Analysis** - Multi-agent committee approach with fine-tuned models
2. **Single-Stage Analysis** - Direct single-pass analysis
3. **Stage 2 Analysis** - Two-stage processing approach (using option "1")

## Features

- **Side-by-side comparison** of three different analysis approaches
- **Detailed rating system** with four dimensions (1-5 scale):
  - Clarity: How clear and understandable is the analysis?
  - Legal Soundness: How legally accurate and reliable is the analysis?
  - Reasoning Depth: How thorough and detailed is the analysis?
  - Human Alignment: How well does the analysis match what a human lawyer would focus on?
- **Winner selection** to identify the best overall analysis
- **Progress tracking** and history viewing
- **Admin dashboard integration** with statistics and monitoring

## Routes

The new system adds the following routes to the Flask application:

- `/compare` - Entry point and info page
- `/compare/start` - Start comparison session (POST)
- `/compare/vote` - Main comparison page
- `/compare/submit` - Submit comparison vote (POST)
- `/compare/done` - Completion page
- `/compare/history` - View comparison history

## Database Schema

A new table `threeway_votes` stores the comparison results:

```sql
CREATE TABLE threeway_votes (
    user_session TEXT NOT NULL,
    contract_id TEXT NOT NULL,
    committee_source TEXT NOT NULL,     -- Full text of committee analysis
    single_source TEXT NOT NULL,       -- Full text of single-stage analysis
    stage2_source TEXT NOT NULL,       -- Full text of stage2 analysis (option 1)
    winner_source TEXT NOT NULL,       -- 'committee', 'single_stage', or 'stage2_out'
    clarity_rating INTEGER,            -- 1-5 rating
    legal_soundness_rating INTEGER,    -- 1-5 rating
    reasoning_depth_rating INTEGER,    -- 1-5 rating
    human_alignment_rating INTEGER,    -- 1-5 rating
    voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_agent TEXT,
    ip_address TEXT,
    session_start_time TIMESTAMP,
    PRIMARY KEY (user_session, contract_id)
);
```

## Data Sources Configuration

The system expects data from three sources configured in `THREE_WAY_DATA_DIRS`:

```python
THREE_WAY_DATA_DIRS = {
    'fine_tuned_committee': Path("data/fine_tuned_committee_out"),
    'single_stage': Path("data/single_stage_out"),
    'stage2_out': Path("data/stage2_out")
}
```

### Expected Data Format

For each contract, the system expects:

1. **Committee Analysis**: JSON file with single variant or key "0" containing `final_report`
2. **Single-Stage Analysis**: JSON file with single variant or key "0" containing `final_report`
3. **Stage 2 Analysis**: JSON file with key "1" containing `final_report`

Example JSON structure:

```json
{
  "0": {
    "final_report": "Analysis text here...",
    "other_fields": "..."
  },
  "1": {
    "final_report": "Different analysis text...",
    "other_fields": "..."
  }
}
```

## Templates

New templates have been created:

- `threeway_info.html` - Information and consent page
- `threeway_vote.html` - Main comparison interface with three panels and rating forms
- `threeway_all_done.html` - Completion page
- `threeway_history.html` - User's comparison history
- Updated `admin.html` - Added three-way comparison statistics

## Admin Dashboard

The admin dashboard now includes:

- Total three-way votes and unique users
- Average ratings across all dimensions
- Winner distribution (which approach wins most often)
- Detailed statistics and monitoring

## Setup Instructions

1. **Ensure data directories exist** and contain the appropriate contract analysis files
2. **Database migration** will happen automatically when the app starts
3. **Access the system** at `/compare`
4. **Monitor usage** via the admin dashboard

## Data Export

Three-way comparison data is included in the existing admin export functionality and can be analyzed alongside the pairwise comparison data.

## Fallback Behavior

If the expected data directories don't exist, the system will:

- Use placeholder text for missing analyses
- Fall back to the first 10 contracts from `stage2_out` for testing
- Log errors appropriately for debugging

## Integration with Existing System

The three-way comparison system runs independently alongside the existing pairwise comparison system. Users can access both through their respective entry points, and both systems maintain separate session tracking while sharing the same user identification system.
