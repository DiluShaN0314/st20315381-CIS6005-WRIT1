from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and scaler
best_model = joblib.load("models/lgbm_best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load team list and stats from tournament data
df_tournament = pd.read_csv("data/generated/tournament_data.csv")  # Ensure this CSV is generated from model.py
# Define team ID ranges
team_ids_set = set(pd.concat([df_tournament['WTeamID'], df_tournament['LTeamID']]).unique())
additional_team_ids = set(range(3101, 3481))
teams = sorted(team_ids_set | additional_team_ids)

# Load submission.csv
submission_df = pd.read_csv("data/generated/submission.csv")

# Split the 'ID' column into 'Year', 'Team_1', and 'Team_2'
submission_df[['Year', 'Team_1', 'Team_2']] = submission_df['ID'].str.split('_', expand=True)

# Convert 'Team_1' and 'Team_2' columns to integers if they represent team IDs
submission_df['Team_1'] = submission_df['Team_1'].astype(int)
submission_df['Team_2'] = submission_df['Team_2'].astype(int)

# Now your DataFrame will have 'Year', 'Team_1', and 'Team_2' columns.
print(submission_df.head())

# teams = submission_df['Team_1'].unique()
# Team ID to Name Mapping (You can replace this with real team names)
team_mapping = {team_id: f"Team {team_id}" for team_id in teams}

# Compute dataset-wide average scores to use as defaults
avg_wscore = df_tournament["WScore"].mean()
avg_lscore = df_tournament["LScore"].mean()

def get_team_stat(team_id, column, default=None):
    """Get team stat from df_tournament, replacing NaN with dataset mean if missing."""
    value = df_tournament.loc[df_tournament['WTeamID'] == team_id, column].mean()
    return value if pd.notna(value) else (default if default is not None else 0)  # Use dataset mean if available


@app.route('/')
def home():
    return render_template('index.html', teams=team_mapping)

@app.route('/predict', methods=['POST'])
def predict():
    team1 = int(request.form["team_1"])
    team2 = int(request.form["team_2"])

    # Ensure lower TeamID is first for consistency
    lower_id, higher_id = min(team1, team2), max(team1, team2)


    # Extract features for matchup
    seed_diff = get_team_stat(lower_id, "WSeed", default=0) - get_team_stat(higher_id, "WSeed", default=0)
    win_pct_rolling_7 = get_team_stat(lower_id, "Win_Pct_Rolling_7", default=0) - get_team_stat(higher_id, "Win_Pct_Rolling_7", default=0)

    wscore_rolling_7 = get_team_stat(lower_id, "WScore_Rolling_7", default=avg_wscore)
    lscore_rolling_7 = get_team_stat(higher_id, "LScore_Rolling_7", default=avg_lscore)

    wscore = get_team_stat(lower_id, "WScore", default=avg_wscore)
    lscore = get_team_stat(higher_id, "LScore", default=avg_lscore)

    day_num = df_tournament["DayNum"].mean()

    # Prepare input feature array
    input_features = [seed_diff, wscore_rolling_7, lscore_rolling_7, win_pct_rolling_7, wscore, lscore, day_num]

    # Scale input features
    scaled_input = scaler.transform([input_features])

    # Make predictions
    probability = best_model.predict_proba(scaled_input)[:, 1][0]
    predicted_winner = lower_id if probability > 0.5 else higher_id
    predicted_loser = higher_id if probability > 0.5 else lower_id

    # Get the winning and losing scores
    winning_score = wscore if predicted_winner == lower_id else lscore
    losing_score = lscore if predicted_winner == lower_id else wscore

     # Handle NaN scores (fallback to 0)
    winning_score = 0 if np.isnan(winning_score) else winning_score
    losing_score = 0 if np.isnan(losing_score) else losing_score

    # SHAP feature importance plot
    img = io.BytesIO()
    plt.figure(figsize=(10, 5))
    feature_importances = best_model.feature_importances_
    features = ["SeedDiff", "WScore_Rolling_7", "LScore_Rolling_7", "Win_Pct_Rolling_7", "WScore", "LScore", "DayNum"]
    plt.barh(features, feature_importances, color='royalblue')
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("LGBM Feature Importance")
    plt.savefig(img, format='png')
    img.seek(0)
    feature_importance_img = base64.b64encode(img.getvalue()).decode('utf-8')

    return jsonify({
        'winning_team': team_mapping[predicted_winner],
        'losing_team': team_mapping[predicted_loser],
        'winning_probability': round(probability, 2),
        'winning_score': round(winning_score),
        'losing_score': round(losing_score),
        'feature_importance_img': feature_importance_img
    })

if __name__ == '__main__':
    app.run(debug=True)
