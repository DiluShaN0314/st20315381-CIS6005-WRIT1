import itertools
import pandas as pd
import joblib
import os

# Load preprocessed data
df_tournament = pd.read_csv("tournament_data.csv")
df_regular = pd.read_csv("regular_season_data.csv")
df_tournament_balanced = pd.read_csv("balanced_tournament_data.csv")

# Load your trained model and scaler
best_model = joblib.load("lgbm_best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load Team IDs from the dataset (Men’s and Women’s)
team_ids = sorted(pd.concat([df_tournament['WTeamID'], df_tournament['LTeamID']]).unique())

# Generate all possible matchups
matchups = list(itertools.combinations(team_ids, 2))  # Get all possible matchups

# Convert to DataFrame
matchup_df = pd.DataFrame(matchups, columns=['Team1', 'Team2'])
matchup_df["Season"] = 2025  # Assign current season

# Create matchup features
def create_matchup_features(row):
    team1, team2 = row["Team1"], row["Team2"]
    lower_id, higher_id = min(team1, team2), max(team1, team2)
    row["ID"] = f"2025_{lower_id}_{higher_id}"

    # Extracting missing features
    row["SeedDiff"] = df_tournament.loc[df_tournament['WTeamID'] == lower_id, "WSeed"].mean() - \
                      df_tournament.loc[df_tournament['WTeamID'] == higher_id, "WSeed"].mean()

    row["Win_Pct_Rolling_7"] = df_tournament.loc[df_tournament['WTeamID'] == lower_id, "Win_Pct_Rolling_7"].mean() - \
                                df_tournament.loc[df_tournament['WTeamID'] == higher_id, "Win_Pct_Rolling_7"].mean()

    row["WScore_Rolling_7"] = df_tournament.loc[df_tournament['WTeamID'] == lower_id, "WScore_Rolling_7"].mean()
    row["LScore_Rolling_7"] = df_tournament.loc[df_tournament['WTeamID'] == higher_id, "LScore_Rolling_7"].mean()
    row["WScore"] = df_tournament.loc[df_tournament['WTeamID'] == lower_id, "WScore"].mean()
    row["LScore"] = df_tournament.loc[df_tournament['WTeamID'] == higher_id, "LScore"].mean()
    row["DayNum"] = df_tournament["DayNum"].mean()  # Approximate average day number

    return row

# Apply feature engineering
matchup_df = matchup_df.apply(create_matchup_features, axis=1)

# Ensure all required features are included
features = ["SeedDiff", "WScore_Rolling_7", "LScore_Rolling_7", "Win_Pct_Rolling_7", "WScore", "LScore", "DayNum"]
X_new = matchup_df[features].fillna(0)


# Scale data and predict
X_new_scaled = scaler.transform(X_new)
matchup_df["Pred"] = best_model.predict_proba(X_new_scaled)[:, 1]

# Predict probabilities
matchup_df["Pred"] = best_model.predict_proba(X_new_scaled)[:, 1]

# Keep only required columns
submission = matchup_df[["ID", "Pred"]]

# Ensure the directory exists
output_dir = os.path.join("..", "data", "generated")
os.makedirs(output_dir, exist_ok=True)

# Path to save the CSV file
csv_file_path = os.path.join(output_dir, "submission.csv")
# Save submission file
submission.to_csv(csv_file_path, index=False)

print("✅ Submission file 'submission.csv' created successfully!")
