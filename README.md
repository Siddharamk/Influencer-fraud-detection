# Influencer Fraud Detection

A machine learning system for detecting fraudulent influencers on Instagram and TikTok using anomaly detection techniques.

---

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Methodology](#methodology)
4. [Feature Engineering](#feature-engineering)
5. [Output Files](#output-files)
6. [Usage](#usage)
7. [Results](#results)
8. [Dependencies](#dependencies)
9. [License](#license)

## Overview

This project detects fraudulent influencers by analyzing their engagement patterns, follower dynamics, and content behavior. It combines two complementary anomaly detection algorithms:

- **Isolation Forest** — isolates anomalies by randomly partitioning features; anomalies are easier to isolate and thus require fewer splits.
- **Local Outlier Factor (LOF)** — identifies local density deviations; points with significantly lower local density than their neighbors are flagged as outliers.

The system scores each influencer on a **0–100 fraud risk scale** and assigns a **vetting action** (`Reject / Escalate` or `Manual Review — High Concern`) to guide downstream moderation workflows.

## Project Structure

```
Influencer_Fraud_Detection/
├── Index.ipynb                          # Main analysis notebook (entry point)
├── README.ipynb                         # This file — project documentation
└── outputs/
    ├── influencer_fraud_scores.csv      # Scored influencer dataset (5,000 rows)
    ├── precision_recall_sweep.png       # Precision-recall curve across thresholds
    └── score_distribution.png           # Distribution of fraud risk scores
```

## Methodology

### 1. Data Collection
Influencer profile data is collected from Instagram and TikTok APIs, including follower counts, engagement metrics, posting history, and account metadata.

### 2. Feature Engineering
Raw metrics are transformed into behavioral features that capture suspicious patterns:

| Feature | Description |
|---|---|
| `engagement_rate` | (avg_likes + avg_comments) / followers — abnormally high or low values are suspicious |
| `comment_like_ratio` | avg_comments / avg_likes — bot accounts often have unnatural ratios |
| `follower_following_ratio` | followers / following — very high ratios suggest purchased followers |
| `pct_comments_generic` | Percentage of generic/repetitive comments — high values indicate bot engagement |
| `posting_regularity_cv` | Coefficient of variation in posting intervals — irregular patterns are suspicious |
| `audience_overlap_score` | Degree of audience overlap across accounts — high overlap suggests shared bot networks |
| `follower_growth_30d_pct` | 30-day follower growth percentage — sudden spikes indicate artificial growth |

### 3. Anomaly Detection
Two unsupervised models are trained on the feature set:

- **Isolation Forest** (`iso_forest_score`): Scores based on path length in random trees. Lower scores = more anomalous.
- **Local Outlier Factor** (`lof_score`): Scores based on local density deviation. Higher scores = more anomalous.

### 4. Risk Score Aggregation
The two model scores are normalized and combined into a single `fraud_risk_score` (0–100):

```
fraud_risk_score = 100 × (0.5 × normalized_iso_forest + 0.5 × normalized_lof)
```

### 5. Vetting Action Assignment
Based on the risk score and manually labeled ground truth (where available):

| Risk Score Range | Vetting Action |
|---|---|
| ≥ 40 | Reject / Escalate |
| 25–40 | Manual Review — High Concern |
| < 25 | Low Risk (no action) |

The threshold was tuned using a **precision-recall sweep** to balance false positives against fraud detection coverage.

## Feature Engineering

The following features are computed from raw influencer data:

```python
import pandas as pd
import numpy as np

# Load the scored dataset
df = pd.read_csv('outputs/influencer_fraud_scores.csv')

# Display the feature columns
feature_columns = [
    'followers', 'following', 'posts', 'avg_likes', 'avg_comments',
    'account_age_days', 'follower_growth_30d_pct', 'engagement_rate',
    'comment_like_ratio', 'follower_following_ratio', 'pct_comments_generic',
    'posting_regularity_cv', 'audience_overlap_score',
    'iso_forest_score', 'lof_score'
]

print(f"Dataset shape: {df.shape}")
print(f"Number of features: {len(feature_columns)}")
print(f"Platforms: {df['platform'].unique().tolist()}")
print(f"Vetting actions: {df['vetting_action'].unique().tolist()}")
```


```python
import pandas as pd
import numpy as np

# Load the scored dataset
df = pd.read_csv('outputs/influencer_fraud_scores.csv')

# Display the feature columns
feature_columns = [
    'followers', 'following', 'posts', 'avg_likes', 'avg_comments',
    'account_age_days', 'follower_growth_30d_pct', 'engagement_rate',
    'comment_like_ratio', 'follower_following_ratio', 'pct_comments_generic',
    'posting_regularity_cv', 'audience_overlap_score',
    'iso_forest_score', 'lof_score'
]

print(f"Dataset shape: {df.shape}")
print(f"Number of features: {len(feature_columns)}")
print(f"Platforms: {df['platform'].unique().tolist()}")
print(f"Vetting actions: {df['vetting_action'].unique().tolist()}")
```

## Output Files

### `outputs/influencer_fraud_scores.csv`
The primary output — a scored dataset of 5,000 influencers with the following columns:

| Column | Type | Description |
|---|---|---|
| `handle` | string | Influencer handle (e.g., `@influencer_03978`) |
| `platform` | string | Social platform (`Instagram` or `TikTok`) |
| `followers` | int | Total follower count |
| `following` | int | Accounts followed |
| `posts` | int | Total posts |
| `avg_likes` | float | Average likes per post |
| `avg_comments` | float | Average comments per post |
| `account_age_days` | float | Account age in days |
| `follower_growth_30d_pct` | float | 30-day follower growth percentage |
| `engagement_rate` | float | (likes + comments) / followers |
| `comment_like_ratio` | float | Comments / likes ratio |
| `follower_following_ratio` | float | Followers / following ratio |
| `pct_comments_generic` | float | Percentage of generic comments |
| `posting_regularity_cv` | float | Coefficient of variation in posting intervals |
| `audience_overlap_score` | float | Audience overlap with other accounts |
| `iso_forest_score` | float | Isolation Forest anomaly score |
| `lof_score` | float | Local Outlier Factor score |
| `fraud_risk_score` | float | Composite fraud risk score (0–100) |
| `vetting_action` | string | Recommended action (`Reject / Escalate` or `Manual Review — High Concern`) |
| `manually_labeled` | bool | Whether the account was manually labeled for ground truth |

### `outputs/precision_recall_sweep.png`
Precision-recall curve showing model performance across different risk score thresholds. Used to select the optimal threshold for the `fraud_risk_score`.

### `outputs/score_distribution.png`
Histogram of fraud risk scores across all 5,000 influencers, showing the distribution of high-risk vs. low-risk accounts.

## Usage

### Prerequisites
Install the required Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Analysis
1. Open `Index.ipynb` in Jupyter:

```bash
jupyter notebook Index.ipynb
```

2. Run all cells to:
   - Load and preprocess influencer data
   - Engineer behavioral features
   - Train Isolation Forest and LOF models
   - Compute fraud risk scores
   - Generate visualizations
   - Export results to `outputs/influencer_fraud_scores.csv`

### Loading the Results
```python
import pandas as pd

df = pd.read_csv('outputs/influencer_fraud_scores.csv')

# View top 10 highest-risk influencers
top_risk = df.nlargest(10, 'fraud_risk_score')
print(top_risk[['handle', 'platform', 'fraud_risk_score', 'vetting_action']])

# Filter by vetting action
reject = df[df['vetting_action'] == 'Reject / Escalate']
review = df[df['vetting_action'] == 'Manual Review — High Concern']
print(f"Reject/Escalate: {len(reject)} accounts")
print(f"Manual Review: {len(review)} accounts")
```


```python
import pandas as pd

df = pd.read_csv('outputs/influencer_fraud_scores.csv')

# View top 10 highest-risk influencers
top_risk = df.nlargest(10, 'fraud_risk_score')
print(top_risk[['handle', 'platform', 'fraud_risk_score', 'vetting_action']])

# Filter by vetting action
reject = df[df['vetting_action'] == 'Reject / Escalate']
review = df[df['vetting_action'] == 'Manual Review — High Concern']
print(f"\nReject/Escalate: {len(reject)} accounts")
print(f"Manual Review: {len(review)} accounts")
```

## Results

### Summary Statistics

The model was evaluated on a dataset of 5,000 influencers across Instagram and TikTok. Key findings:

- **High-risk accounts** (fraud_risk_score ≥ 40): Flagged for `Reject / Escalate` — these accounts exhibit strong anomaly signals from both Isolation Forest and LOF.
- **Medium-risk accounts** (fraud_risk_score 25–40): Flagged for `Manual Review — High Concern` — these require human review to confirm fraud.
- **Low-risk accounts** (fraud_risk_score < 25): No action required.

### Platform Breakdown
The dataset includes influencers from both **Instagram** and **TikTok**, with the model performing comparably across both platforms.

### Model Performance
The precision-recall sweep (see `outputs/precision_recall_sweep.png`) demonstrates that the combined Isolation Forest + LOF approach achieves strong precision at high recall thresholds, making it suitable for fraud detection where false negatives are costly.

### Score Distribution
The score distribution (see `outputs/score_distribution.png`) shows a bimodal pattern — a large cluster of low-risk accounts and a smaller cluster of high-risk accounts, indicating clear separation between legitimate and fraudulent influencers.


```python
import pandas as pd

df = pd.read_csv('outputs/influencer_fraud_scores.csv')

# Summary statistics
print("=== Dataset Summary ===")
print(f"Total influencers: {len(df)}")
print(f"Platforms: {df['platform'].value_counts().to_dict()}")
print(f"\n=== Vetting Action Distribution ===")
print(df['vetting_action'].value_counts())
print(f"\n=== Fraud Risk Score Statistics ===")
print(df['fraud_risk_score'].describe())
print(f"\n=== Manually Labeled ===")
print(df['manually_labeled'].value_counts())
```

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pandas` | ≥1.3 | Data manipulation and analysis |
| `numpy` | ≥1.21 | Numerical computing |
| `scikit-learn` | ≥1.0 | Isolation Forest, LOF, metrics |
| `matplotlib` | ≥3.4 | Plotting and visualization |
| `seaborn` | ≥0.11 | Statistical data visualization |
| `jupyter` | ≥1.0 | Notebook environment |

Install all dependencies:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## License

This project is provided for research and educational purposes. The influencer data used is synthetic/simulated and does not contain real personal information.

---

### Quick Links
- **Main Notebook**: [`Index.ipynb`](Index.ipynb)
- **Output Data**: [`outputs/influencer_fraud_scores.csv`](outputs/influencer_fraud_scores.csv)
- **Precision-Recall Curve**: [`outputs/precision_recall_sweep.png`](outputs/precision_recall_sweep.png)
- **Score Distribution**: [`outputs/score_distribution.png`](outputs/score_distribution.png)
