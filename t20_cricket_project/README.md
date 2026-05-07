# T20 Cricket Analytics — FAST NUCES Stat & Prob Project 2026

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python app.py
```

### 3. Open in browser
Go to: http://localhost:5000

## Pages
| URL | Description |
|-----|-------------|
| `/` | Dashboard — Top players, overview stats, player search |
| `/descriptive` | Descriptive Stats — Mean, median, CI, correlations |
| `/probability` | Probability — Normality tests, empirical probabilities, Q-Q plot |
| `/regression` | Regression — Linear model, scatter, residuals, predictor |
| `/rawdata` | Raw Data — Full 2006-player table with search/sort/pagination |

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/overview-stats` | GET | Summary cards |
| `/api/top-players` | GET | Top 10 by runs & average |
| `/api/search-player?q=` | GET | Player search |
| `/api/descriptive-stats` | GET | All stats, CI, histogram |
| `/api/probability-analysis` | GET | Normality tests, probabilities |
| `/api/regression-model` | GET | Model, scatter, residuals |
| `/api/predict-runs` | POST | `{sr, mat}` → predicted runs |
| `/api/all-players` | GET | Paginated player table |

## Dataset
- Source: Kaggle — ICC T20 Cricket Player Stats
- Size: 2006 players, 15 variables
