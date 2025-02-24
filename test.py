import pandas as pd
from itertools import combinations

# Load teams
df_tournament = pd.read_csv("data/generated/tournament_data.csv")
teams = sorted(df_tournament["WTeamID"].unique())  # Ensure unique teams

# Generate all possible matchups (team1, team2) combinations
all_matchups = pd.DataFrame(list(combinations(teams, 2)), columns=["Team1", "Team2"])
all_matchups["ID"] = all_matchups.apply(lambda row: f"{row.Team1}_{row.Team2}", axis=1)

# Merge with generated submission to find missing matchups
generated_submission = pd.read_csv("data/generated/submission.csv")
missing_matchups = all_matchups[~all_matchups["ID"].isin(generated_submission["ID"])]

# Save missing matchups for reference
missing_matchups.to_csv("data/generated/missing_matchups.csv", index=False)
print(f"🔍 Missing Matchups: {len(missing_matchups)} saved to missing_matchups.csv!")

# Load the trained model
import joblib
import numpy as np

best_model = joblib.load("models/lgbm_best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Function to generate features for missing matchups
def generate_features(team1, team2, df):
    seed_diff = df[df["WTeamID"] == team1]["WSeed"].mean() - df[df["WTeamID"] == team2]["WSeed"].mean()
    win_pct_rolling_7 = df[df["WTeamID"] == team1]["Win_Pct_Rolling_7"].mean() - df[df["WTeamID"] == team2]["Win_Pct_Rolling_7"].mean()
    wscore_rolling_7 = df[df["WTeamID"] == team1]["WScore_Rolling_7"].mean()
    lscore_rolling_7 = df[df["WTeamID"] == team2]["LScore_Rolling_7"].mean()
    wscore = df[df["WTeamID"] == team1]["WScore"].mean()
    lscore = df[df["WTeamID"] == team2]["LScore"].mean()
    day_num = df["DayNum"].mean()

    return [seed_diff, wscore_rolling_7, lscore_rolling_7, win_pct_rolling_7, wscore, lscore, day_num]

# Predict missing matchups
predictions = []
for _, row in missing_matchups.iterrows():
    team1, team2 = row["Team1"], row["Team2"]
    features = generate_features(team1, team2, df_tournament)
    scaled_features = scaler.transform([features])

    probability = best_model.predict_proba(scaled_features)[:, 1][0]
    predictions.append([row["ID"], probability])

# Save predictions
missing_predictions_df = pd.DataFrame(predictions, columns=["ID", "Pred"])
missing_predictions_df.to_csv("data/generated/missing_predictions.csv", index=False)
print(f"✅ Predicted {len(missing_predictions_df)} missing games!")
