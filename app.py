"""
T20 Cricket Analytics - Flask Backend
Stat & Prob Project - FAST NUCES Spring 2026
Handles all data loading, statistical analysis, and API endpoints.
"""

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, lognorm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ─── Load & Clean Dataset ─────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv('t20.csv')
    df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c], errors='ignore')
    df = df.reset_index(drop=True)
    numeric_cols = ['Mat','Inns','NO','Runs','HS','Ave','BF','SR','100','50','0','4s','6s']
    for col in numeric_cols:
        if col in df.columns:
            # '-' in the CSV means the player never batted/fielded in that stat — treat as 0
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace('*', '', regex=False).str.replace('-', '', regex=False),
                errors='coerce'
            ).fillna(0)
    return df

DF = load_data()

# ─── Routes: Pages ────────────────────────────────────────────────────────────

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/descriptive')
def descriptive():
    return render_template('descriptive.html')

@app.route('/probability')
def probability():
    return render_template('probability.html')

@app.route('/regression')
def regression():
    return render_template('regression.html')

@app.route('/rawdata')
def rawdata():
    return render_template('rawdata.html')

# ─── API: Dashboard ───────────────────────────────────────────────────────────

@app.route('/api/top-players')
def top_players():
    top_runs = DF.nlargest(10, 'Runs')[['Player','Runs','Ave','SR','Mat']].fillna(0)
    top_avg  = DF[DF['Inns'] >= 20].nlargest(10, 'Ave')[['Player','Ave','Runs','SR']].fillna(0)
    return jsonify({
        'top_by_runs': top_runs.to_dict(orient='records'),
        'top_by_avg':  top_avg.to_dict(orient='records')
    })

@app.route('/api/overview-stats')
def overview_stats():
    # Country Distribution
    countries = DF['Player'].str.extract(r'\((.*?)\)')[0].str.split('/').str[-1].value_counts()
    country_data = {
        'labels': countries.head(15).index.tolist(),
        'values': countries.head(15).tolist()
    }
    return jsonify({
        'total_players':       int(len(DF)),
        'total_countries':     int(countries.nunique()),
        'highest_runs':        int(DF['Runs'].max()),
        'highest_runs_player': str(DF.loc[DF['Runs'].idxmax(), 'Player']),
        'avg_strike_rate':     round(float(DF['SR'].mean()), 2),
        'total_sixes':         int(DF['6s'].sum()),
        'country_dist':        country_data
    })

@app.route('/api/search-player')
def search_player():
    q = request.args.get('q', '').lower()
    results = DF[DF['Player'].str.lower().str.contains(q, na=False)].head(10)
    return jsonify(results.fillna(0).to_dict(orient='records'))

# ─── API: Descriptive Statistics ──────────────────────────────────────────────

@app.route('/api/descriptive-stats')
def descriptive_stats():
    cols = ['Runs','Ave','SR','BF','Mat','4s','6s','50']
    result = {}
    z95 = 1.96
    for col in cols:
        data = DF[col].dropna()
        mean = float(data.mean())
        std  = float(data.std())
        se   = std / np.sqrt(len(data))
        try:    mode_val = float(data.mode()[0])
        except: mode_val = None
        q1 = float(data.quantile(0.25))
        q3 = float(data.quantile(0.75))
        iqr = q3 - q1
        outliers = data[(data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))]
        result[col] = {
            'mean':     round(mean, 2),
            'median':   round(float(data.median()), 2),
            'mode':     round(mode_val, 2) if mode_val is not None else None,
            'std_dev':  round(std, 2),
            'variance': round(float(data.var()), 2),
            'min':      round(float(data.min()), 2),
            'max':      round(float(data.max()), 2),
            'q1':       round(q1, 2),
            'q3':       round(q3, 2),
            'outliers': int(len(outliers)),
            'ci_lower': round(mean - z95*se, 2),
            'ci_upper': round(mean + z95*se, 2),
            'skewness': round(float(data.skew()), 3),
            'kurtosis': round(float(data.kurt()), 3),
        }
    corr = DF[cols].corr()['Runs'].drop('Runs').sort_values(ascending=False)
    result['correlations_with_runs'] = {k: round(float(v), 3) for k, v in corr.items()}
    hist_data, bin_edges = np.histogram(DF['Runs'].dropna(), bins=15)
    result['runs_histogram'] = {
        'counts': hist_data.tolist(),
        'bins':   [round(x,1) for x in bin_edges.tolist()]
    }
    return jsonify(result)

# ─── API: Probability ─────────────────────────────────────────────────────────

@app.route('/api/probability-analysis')
def probability_analysis():
    runs = DF['Runs'].dropna()
    sr   = DF['SR'].dropna()
    ave  = DF['Ave'].dropna()

    shape, loc, scale = lognorm.fit(runs + 1, floc=0)
    sample = runs.sample(min(500, len(runs)), random_state=42)
    stat, p_value = stats.shapiro(sample)

    qq_sample  = runs.sample(min(200, len(runs)), random_state=1).sort_values()
    theoretical = norm.ppf(
        np.linspace(0.01, 0.99, len(qq_sample)),
        loc=float(runs.mean()), scale=float(runs.std())
    )

    def emp_prob(series, threshold):
        return round(float((series > threshold).mean()), 4)

    ks_stat, ks_p = stats.kstest(runs, 'norm', args=(float(runs.mean()), float(runs.std())))

    return jsonify({
        'distribution_shape': {
            'skewness':     round(float(runs.skew()), 3),
            'kurtosis':     round(float(runs.kurt()), 3),
            'is_normal':    bool(abs(runs.skew()) < 0.5), # Simple rule of thumb
        },
        'probabilities': {
            'runs_gt_500':  emp_prob(runs, 500),
            'runs_gt_1000': emp_prob(runs, 1000),
            'avg_gt_30':    emp_prob(ave,  30),
            'avg_gt_40':    emp_prob(ave,  40),
            'sr_gt_130':    emp_prob(sr,   130),
            'sr_gt_150':    emp_prob(sr,   150),
        },
        'lognorm_fit':  {'shape': round(float(shape),4), 'loc': round(float(loc),4), 'scale': round(float(scale),4)},
        'qq_plot': {
            'observed':    [round(x,2) for x in qq_sample.tolist()],
            'theoretical': [round(x,2) for x in theoretical.tolist()],
        },
        'conditional_probs': {
            'runs_gt_1000_given_mat_gt_50': round(float((DF[DF['Mat'] > 50]['Runs'] > 1000).mean()), 4),
            'avg_gt_40_given_runs_gt_500': round(float((DF[DF['Runs'] > 500]['Ave'] > 40).mean()), 4),
        },
        'runs_stats':      {'mean': round(float(runs.mean()),2), 'std': round(float(runs.std()),2), 'total': int(len(runs))},
        'runs_density': {
            'x': np.linspace(0, float(runs.max()), 50).tolist(),
            'y': stats.gaussian_kde(runs)(np.linspace(0, float(runs.max()), 50)).tolist()
        }
    })

# ─── API: Distributions (Binomial & Poisson) ─────────────────────────────────

@app.route('/api/distributions')
def distributions():
    runs = DF['Runs'].dropna()
    centuries = DF['100'].dropna()

    # Binomial: P(exactly k of 10 players score 500+ runs)
    p_500 = float((runs > 500).mean())   # empirical probability
    n = 10
    binom_pmf = [
        {'k': k, 'prob': round(float(stats.binom.pmf(k, n, p_500)), 4)}
        for k in range(n + 1)
    ]

    # Poisson: P(player scores exactly k centuries), lambda = mean centuries
    lam = float(centuries.mean())
    poisson_pmf = [
        {'k': k, 'prob': round(float(stats.poisson.pmf(k, lam)), 4)}
        for k in range(7)
    ]

    return jsonify({
        'binomial': {
            'n': n, 'p': round(p_500, 4),
            'description': f'P(exactly k of {n} random players score 500+ runs)',
            'pmf': binom_pmf
        },
        'poisson': {
            'lambda': round(lam, 4),
            'description': 'P(player scores exactly k centuries in career)',
            'pmf': poisson_pmf
        }
    })

# ─── API: Regression ──────────────────────────────────────────────────────────

@app.route('/api/regression-model')
def regression_model():
    subset = DF[['Runs','SR','Mat']].dropna()
    X = subset[['SR','Mat']].values
    y = subset['Runs'].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred  = model.predict(X)
    r2      = r2_score(y, y_pred)
    rmse    = np.sqrt(mean_squared_error(y, y_pred))
    idx     = np.random.RandomState(42).choice(len(y), min(300, len(y)), replace=False)
    return jsonify({
        'coefficients': {
            'SR':        round(float(model.coef_[0]),4),
            'Mat':       round(float(model.coef_[1]),4),
            'intercept': round(float(model.intercept_),4),
        },
        'formula': f"Runs = {model.coef_[0]:.4f}×SR + {model.coef_[1]:.4f}×Mat + ({model.intercept_:.4f})",
        'r2':   round(float(r2),   4),
        'rmse': round(float(rmse), 2),
        'n':    int(len(y)),
        'scatter': {
            'actual':    [round(float(y[i]),1)       for i in idx],
            'predicted': [round(float(y_pred[i]),1)  for i in idx],
            'sr':        [round(float(X[i,0]),1)     for i in idx],
            'mat':       [round(float(X[i,1]),1)     for i in idx],
        },
        'residuals_sample': [round(float(r),2) for r in (y - y_pred)[:200]],
    })

@app.route('/api/predict-runs', methods=['POST'])
def predict_runs():
    data   = request.get_json()
    sr     = float(data.get('sr', 130))
    mat    = float(data.get('mat', 30))
    subset = DF[['Runs','SR','Mat']].dropna()
    model  = LinearRegression()
    model.fit(subset[['SR','Mat']].values, subset['Runs'].values)
    pred   = float(model.predict([[sr, mat]])[0])
    pct    = float(stats.percentileofscore(DF['Runs'].dropna(), pred))
    return jsonify({'predicted_runs': round(max(0,pred),0), 'percentile': round(pct,1), 'sr': sr, 'mat': mat})

# ─── API: Raw Data ────────────────────────────────────────────────────────────

@app.route('/api/all-players')
def all_players():
    page     = int(request.args.get('page',1))
    per_page = int(request.args.get('per_page',50))
    search   = request.args.get('search','').lower()
    sort_by  = request.args.get('sort','Runs')
    sort_dir = request.args.get('dir','desc')
    df = DF.copy()
    if search:
        df = df[df['Player'].str.lower().str.contains(search, na=False)]
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=(sort_dir=='asc'))
    total    = len(df)
    start    = (page-1)*per_page
    page_df  = df.iloc[start:start+per_page]
    return jsonify({'total': total, 'page': page, 'per_page': per_page,
                    'players': page_df.fillna('—').to_dict(orient='records')})

if __name__ == '__main__':
    app.run(debug=True, port=5001)