import os
import pandas as pd
import numpy as np
import glob
import chardet  # Import chardet
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from scipy.stats import randint
from imblearn.over_sampling import SMOTE  # Import SMOTE

dataframes = {}

csv_files = glob.glob('../data/march-machine-learning-mania-2025/*.csv')  # Specific path

for file in csv_files:
    try:
        filename = os.path.splitext(os.path.basename(file))[0]
        key = filename.lower().replace("m", "").replace("ncaa", "").replace("detailedresults", "").replace("regularseason", "rs").replace("tourney", "t").replace("mseeds","seeds").replace("mteamspellings","teamspellings").replace("wteamspellings","wteamspellings")

        loaded = False

        # 1. Try specific encodings that often work with these characters
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']:  # Add ISO-8859-1
            try:
                dataframes[key] = pd.read_csv(file, encoding=encoding)
                print(f"Loaded {file} with encoding: {encoding}")
                loaded = True
                break
            except UnicodeDecodeError:
                print(f"Failed to load {file} with encoding: {encoding}")
                pass

            # 2. If the above fails, try chardet (but be cautious)
            if not loaded:
                try:
                    with open(file, 'rb') as f:
                        result = chardet.detect(f.read())
                        encoding = result['encoding']
                        confidence = result['confidence']

                    print(f"Chardet detected encoding: {encoding} (confidence: {confidence})")

                    if confidence > 0.8:  # Only use if confidence is high
                        try:
                            dataframes[key] = pd.read_csv(file, encoding=encoding)
                            print(f"Loaded {file} with detected encoding: {encoding}")
                            loaded = True
                        except UnicodeDecodeError: #Catch the error again
                            print(f"Failed to load {file} with encoding: {encoding}")
                            pass
                    else:
                        print("Chardet confidence too low. Trying backup encodings.")
                        for encoding in ['latin-1', 'cp1252','ISO-8859-1']: #Add ISO-8859-1
                            try:
                                dataframes[key] = pd.read_csv(file, encoding=encoding)
                                print(f"Loaded {file} with encoding: {encoding}")
                                loaded = True
                                break
                            except UnicodeDecodeError:
                                print(f"Failed to load {file} with encoding: {encoding}")
                                pass


                except Exception as e:
                    print(f"Error with chardet: {e}")

            if not loaded:
                print(f"Error: Could not load {file} after trying multiple strategies.")

    except FileNotFoundError:
        print(f"Error: File not found: {file}")
    except pd.errors.ParserError:
        print(f"Error parsing CSV: {file}. Check file format.")
    except Exception as e:
        print(f"Error loading {file}: {e}")

print("\nLoaded datasets:", dataframes.keys())

# Example of iterating through the dataframes
for name, df in dataframes.items():
    print(f"DataFrame: {name}")
    print(df.head()) # print the first few rows of each dataframe

# Data Cleaning and Preprocessing

def clean_data(df, name):
    print(f"\n--- Cleaning {name.upper()} Data ---")

    # 1. Handle Missing Values (Example: Regular Season Results)
    if "rs" in name:  # Apply only to regular season data (adjust as needed)
        # Identify columns with missing values
        missing_values = df.isnull().sum()
        print(f"Missing Values:\n{missing_values}")

    # 2. Correct Inconsistencies (Example: Team Names)
    if "teams" in name:
        # Check for duplicate team names or variations in names
        print(f"Unique Team Names: {df['TeamName'].nunique()}")  # Check for duplicates or similar names
        # Standardize team names (e.g., using a lookup table or fuzzy matching if needed)
        # Example (very basic - you might need more sophisticated methods):
        df['TeamName'] = df['TeamName'].str.strip()  # Remove leading/trailing spaces

    # 3. Data Type Conversion
    # Ensure columns have the correct data types
    if "rs" in name or "t" in name:
        # Convert game scores to integers
        for col in ['WScore', 'LScore']:
            if col in df.columns: # Check if the column exists
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int') # Convert to numbers, set non-numbers to NaN, fill NaN with 0, convert to int

        # Convert date columns to datetime objects (if present)
        date_columns = [col for col in df.columns if 'Date' in col] # Find the columns with 'Date' in their name
        for col in date_columns:
            if col in df.columns: # Check if the column exists
                df[col] = pd.to_datetime(df[col], errors='coerce')

    # 4. Feature Engineering (Optional, but often helpful)
    # Create new features from existing ones
    if "rs" in name:
        df['ScoreDifference'] = df['WScore'] - df['LScore']

    # ... (Add more cleaning steps as needed for other datasets) ...

    return df


# Apply cleaning to all dataframes
for name, df in dataframes.items():
    dataframes[name] = clean_data(df.copy(), name)  # Create a copy to avoid SettingWithCopyWarning

# Verification: Check cleaned dataframes
for name, df in dataframes.items():
    print(f"\n--- Cleaned {name.upper()} Data (First 5 rows) ---")
    # print(df.head())
    # print(df.info()) # Show the info after cleaning

def engineer_features(df, name):
    print(f"\n--- Engineering Features for {name.upper()} Data ---")

    if "rs" in name:  # Regular Season Features
        df = df.sort_values(['Season', 'DayNum'])

        for days in [3, 7, 14]:
            df[f'WScore_Rolling_{days}'] = df.groupby('WTeamID')['WScore'].rolling(window=days, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'LScore_Rolling_{days}'] = df.groupby('LTeamID')['LScore'].rolling(window=days, min_periods=1).mean().reset_index(level=0, drop=True)

            # Correctly calculate rolling win percentage
            df['WWin'] = 1  # Create a win column for the winning team
            df['LWin'] = 0  # Create a win column for the losing team

            df[f'Win_Pct_Rolling_{days}_W'] = df.groupby('WTeamID')['WWin'].rolling(window=days, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'Win_Pct_Rolling_{days}_L'] = df.groupby('LTeamID')['LWin'].rolling(window=days, min_periods=1).mean().reset_index(level=0, drop=True)

            df.drop(['WWin', 'LWin'], axis=1, inplace=True) #remove the created column


    elif "t" in name:  # Tournament Features
        if 'MSeed' in df.columns:
            df['Seed_Diff'] = df['WSeed'].str.extract('(\d+)').astype(int) - df['LSeed'].str.extract('(\d+)').astype(int)

    elif "teams" in name:  # Team Features
        pass  # Placeholder

    # ... (Other feature engineering steps) ...

    return df



# Apply feature engineering and verify (same as before)
for name, df in dataframes.items():
    dataframes[name] = engineer_features(df.copy(), name)

for name, df in dataframes.items():
    print(f"\n--- {name.upper()} Data with Engineered Features (First 5 rows) ---")
    # print(df.head())
    # print(df.info())


# 3. Merge Seed Data with Tournament Results
if 't' in dataframes and 'tseeds' in dataframes:  # Correct the key name here
    # 1. Extract Numeric Part of Seed
    dataframes['tseeds']['Seed'] = dataframes['tseeds']['Seed'].str.extract('(\d+)').astype(float) #Extract the numbers

    # 2. Convert to Int (Handle NaN)
    dataframes['tseeds']['Seed'] = dataframes['tseeds']['Seed'].fillna(0).astype(int) #Fill NaN and convert to int


    # 3. Merge the DataFrames. It is important to merge with the correct ID columns.
    dataframes['t'] = pd.merge(dataframes['t'], dataframes['tseeds'][['Season', 'TeamID', 'Seed']], left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
    dataframes['t'].rename(columns={'Seed': 'WSeed'}, inplace=True) #Rename the column to WSeed
    dataframes['t'] = pd.merge(dataframes['t'], dataframes['tseeds'][['Season', 'TeamID', 'Seed']], left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
    dataframes['t'].rename(columns={'Seed': 'LSeed'}, inplace=True) #Rename the column to LSeed

    #Drop the extra column
    dataframes['t'].drop('TeamID_x', axis=1, inplace=True)
    dataframes['t'].drop('TeamID_y', axis=1, inplace=True)

    print("Tournament DataFrame after merge:\n", dataframes['t'].head()) #Check the merge
else:
    print("Error: 't' or 'seeds' DataFrame not found. Cannot merge.")
    sys.exit() #Exit if merge fails

# 4. Create 'Win' column for tournament data (Targeted Approach)

if 't' in dataframes:
    df_tournament = dataframes['t']  # Access the tournament DataFrame directly

    if 'WTeamID' in df_tournament.columns and 'LTeamID' in df_tournament.columns:
        df_tournament['Win'] = 0  # Initialize Win column to 0 (loss)
        df_tournament["Win"] = df_tournament["WScore"] > df_tournament["LScore"]  # True if WScore is greater than LScore

        # Convert the boolean values to integers (1 for Win, 0 for Loss)
        # df["Win"] = df["Win"].astype(int)

        #Create a Win column based on the WTeamID and LTeamID
        for index, row in df_tournament.iterrows():
            winning_team = row['WTeamID']
            losing_team = row['LTeamID']

            #Set Win to 1 where WTeamID matches TeamID in the original dataframe
            df_tournament.loc[index, 'Win'] = 1

    else:
        print("Warning: 'WTeamID' or 'LTeamID' column not found in tournament data. Cannot create 'Win' column.")
        sys.exit()

else:
    print("Error: 't' DataFrame not found. Cannot create 'Win' column.")
    sys.exit()



def create_features_tournament(df):
    print("\n--- Creating Tournament Features ---")
    print("Columns in Tournament DataFrame:", df.columns)  # Print the columns!

    # 1. Win/Loss (Already present in detailed results)
    # No need to create Win/Loss columns here as tourney results already have W/L

    # 2. Seed Difference
    w_seed_col = None
    l_seed_col = None

    for col in df.columns:
        if 'seed' in col.lower() and 'w' in col.lower(): #Check if 'seed' and 'w' are in the name
            w_seed_col = col
        elif 'seed' in col.lower() and 'l' in col.lower(): #Check if 'seed' and 'l' are in the name
            l_seed_col = col

    if w_seed_col is not None and l_seed_col is not None:
         try:
            # print(df[w_seed_col].head(), df[l_seed_col].head())
            df['SeedDiff'] = df['WSeed'] - df['LSeed']  # No .str[1:] needed
         except (AttributeError, ValueError):
            print("Warning: Issue with seed format/conversion. Check your data.")
            df['SeedDiff'] = 0
    else:
        print("Warning: 'w_seed' or 'l_seed' column not found. SeedDiff cannot be calculated.")
        df['SeedDiff'] = 0

    return df

def create_features_regular(df):
    print("\n--- Creating Regular Season Features ---")
    print("Columns in Regular Season DataFrame:", df.columns)  # Print the columns!

    # 1. Win/Loss Column (Create if not already present)
    if 'WWin' not in df.columns:  # Check if 'WWin' exists (it does in detailed results)
        df['WWin'] = 1
        df['LWin'] = 0

    # 2. Score Difference
    df['ScoreDiff'] = df['WScore'] - df['LScore']

    # 3. Game Outcome Features (Flipping for Loss Records)
    df['Win'] = 1  # Add a win column (1 for win, 0 for loss)
    df_loss = df.copy()
    df_loss['Win'] = 0

    # Swap winning and losing team stats (more efficient)
    for col in ['WTeamID', 'LTeamID', 'WScore', 'LScore', 'WAid', 'LAid']:
        if col in df_loss.columns:
            df_loss.rename(columns={col: col.replace('W', 'T') if 'W' in col else col.replace('L', 'W')}, inplace=True)

    df = pd.concat([df, df_loss], ignore_index=True)

    # 4. Rolling Averages (Calculate *after* flipping for losses)
    df = df.sort_values(['Season', 'DayNum'])  # Sort for rolling calculations
    for days in [3, 7, 14]:
        df[f'WScore_Rolling_{days}'] = df.groupby('WTeamID')['WScore'].rolling(window=days, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'LScore_Rolling_{days}'] = df.groupby('LTeamID')['LScore'].rolling(window=days, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'Win_Pct_Rolling_{days}'] = df.groupby('WTeamID')['Win'].rolling(window=days, min_periods=1).mean().reset_index(level=0, drop=True)

    return df

# Apply Feature Engineering
df_regular = create_features_regular(dataframes['rs'].copy())
df_tournament = create_features_tournament(dataframes['t'].copy())


def perform_eda(df, name):
    print(f"\n--- EDA for {name.upper()} Data ---")

    # 1. Statistical Summaries
    print(f"\nStatistical Summary for {name.upper()}:\n{df.describe()}")

    if "t" in name:
        # Use WSeed and LSeed directly (no need for MSeed or conversion)
        if 'WSeed' in df.columns and 'LSeed' in df.columns:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            sns.boxplot(x='WSeed', y='WScore', data=df)
            plt.title('Winning Score vs. Winning Seed')

            plt.subplot(1, 2, 2)
            sns.boxplot(x='LSeed', y='LScore', data=df)
            plt.title('Losing Score vs. Losing Seed')
            plt.show()

        if 'SeedDiff' in df.columns:  # Check if 'Seed_Diff' was created
            plt.figure(figsize=(8, 5))
            sns.histplot(df['SeedDiff'], kde=True)
            plt.title('Seed Difference Distribution')
            plt.show()
        else:
            print("Warning: 'SeedDiff' column not found.  Check your data.") # Print a warning message

    # 2. Histograms
    if "rs" in name:
        # Score Distributions
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(df['WScore'], kde=True, color='skyblue', label='Winning Score')
        sns.histplot(df['LScore'], kde=True, color='lightcoral', label='Losing Score')
        plt.title('Score Distribution (Regular Season)')
        plt.legend()

        plt.subplot(1, 2, 2)
        sns.histplot(df['ScoreDiff'], kde=True, color='mediumseagreen')
        plt.title('Score Difference Distribution (Regular Season)')
        plt.show()

        # Rolling Win Percentage Distribution
        for days in [3, 7, 14]:
            plt.figure(figsize=(10, 5))
            sns.histplot(df[f'Win_Pct_Rolling_{days}'], kde=True, color='orange')
            plt.title(f'{days}-day Rolling Win Percentage Distribution (Regular Season)')
            plt.show()

    elif "t" in name:
        # Seed Difference Distribution
        if 'SeedDiff' in df.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(df['SeedDiff'], kde=True, color='mediumslateblue')
            plt.title('Seed Difference Distribution (Tournament)')
            plt.show()

    # 3. Box Plots
    if "t" in name and 'WSeed' in df.columns and 'LSeed' in df.columns:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.boxplot(x='WSeed', y='WScore', data=df, color='gold')
        plt.title('Winning Score vs. Winning Seed (Tournament)')

        plt.subplot(1, 2, 2)
        sns.boxplot(x='LSeed', y='LScore', data=df, color='forestgreen')
        plt.title('Losing Score vs. Losing Seed (Tournament)')
        plt.show()

    # 4. Correlation Heatmaps
    if "rs" in name:
        corr_matrix = df[['WScore', 'LScore', 'ScoreDiff', 'Win_Pct_Rolling_3', 'Win_Pct_Rolling_7', 'Win_Pct_Rolling_14']].corr()  # Use calculated rolling averages
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")  # fmt adds 2 decimal places
        plt.title('Correlation Heatmap (Regular Season)')
        plt.show()

    # 5. Scatter Plots
    if "rs" in name:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='DayNum', y='WScore', data=df, label='Winning Score', color='royalblue', alpha=0.7)
        sns.scatterplot(x='DayNum', y='LScore', data=df, label='Losing Score', color='crimson', alpha=0.7)
        plt.title('Score vs. Day Number (Regular Season)')
        plt.xlabel('Day Number')
        plt.ylabel('Score')
        plt.legend()
        plt.show()

        #Rolling win percentage vs Win Percentage
        plt.figure(figsize=(12,6))
        sns.scatterplot(x='Win_Pct_Rolling_7', y='Win', data=df, alpha = 0.5)
        plt.title('Rolling Win Percentage vs Win (Regular Season)')
        plt.show()

    return None


perform_eda(df_regular.copy(), 'rs')  # Regular season EDA
perform_eda(df_tournament.copy(), 't')  # Tournament EDA


# 1. Get Tournament Team IDs (same as before)
tournament_teams = pd.concat([df_tournament['WTeamID'], df_tournament['LTeamID']]).unique()

# 2. Filter Regular Season Data for Tournament Teams (same as before)
rs_tournament_teams = df_regular[df_regular['WTeamID'].isin(tournament_teams) | df_regular['LTeamID'].isin(tournament_teams)]

# 3. Calculate Rolling Averages for Tournament Teams (same as before)
rs_tournament_teams = rs_tournament_teams.sort_values(['Season', 'DayNum'])
for days in [3, 7]:
    rs_tournament_teams[f'WScore_Rolling_{days}'] = rs_tournament_teams.groupby('WTeamID')['WScore'].rolling(window=days, min_periods=1).mean().reset_index(level=0, drop=True)
    rs_tournament_teams[f'LScore_Rolling_{days}'] = rs_tournament_teams.groupby('LTeamID')['LScore'].rolling(window=days, min_periods=1).mean().reset_index(level=0, drop=True)
    rs_tournament_teams[f'Win_Pct_Rolling_{days}'] = rs_tournament_teams.groupby('WTeamID')['Win'].rolling(window=days, min_periods=1).mean().reset_index(level=0, drop=True)

# 4. Merge Rolling Averages into Tournament Data (Corrected and More Robust)
# Create a temporary key for merging
df_tournament['merge_key'] = df_tournament['Season'].astype(str) + '_' + df_tournament['WTeamID'].astype(str) + '_' + df_tournament['LTeamID'].astype(str)
rs_tournament_teams['merge_key'] = rs_tournament_teams['Season'].astype(str) + '_' + rs_tournament_teams['WTeamID'].astype(str) + '_' + rs_tournament_teams['LTeamID'].astype(str)

df_tournament = pd.merge(df_tournament, rs_tournament_teams[['merge_key', 'WScore_Rolling_7', 'LScore_Rolling_7', 'Win_Pct_Rolling_7']], on='merge_key', how='left')

# Drop the temporary key
df_tournament.drop('merge_key', axis=1, inplace=True)
rs_tournament_teams.drop('merge_key', axis=1, inplace=True)

# Example Feature Selection (Adjust as needed)
features = ['SeedDiff', 'WScore_Rolling_7', 'LScore_Rolling_7', 'Win_Pct_Rolling_7', 'WScore', 'LScore', 'DayNum']
X = df_tournament[features].fillna(0)
y = df_tournament['Win']

# Ensure unique column names in the dataset
df_tournament = df_tournament.loc[:, ~df_tournament.columns.duplicated()]

# Create the win column for winning teams
df_tournament["win"] = 1

# Create a flipped dataset where losing teams become winners
df_tournament_flipped = df_tournament.copy()
df_tournament_flipped["win"] = 0  # Mark these as losses

# Swap win/loss columns
flip_columns = {
    "WTeamID": "LTeamID", "LTeamID": "WTeamID",
    "WScore": "LScore", "LScore": "WScore",
    "WSeed": "LSeed", "LSeed": "WSeed",
    "WFGM": "LFGM", "WFGA": "LFGA",
    "WFTA": "LFTA", "WLoc": "WLoc"  # Keep 'WLoc' as is
}

# Rename columns in the flipped dataset
df_tournament_flipped = df_tournament_flipped.rename(columns=flip_columns)

# Ensure all columns exist before merging
for col in df_tournament.columns:
    if col not in df_tournament_flipped.columns:
        df_tournament_flipped[col] = df_tournament[col]

# Reset index before merging
df_tournament.reset_index(drop=True, inplace=True)
df_tournament_flipped.reset_index(drop=True, inplace=True)

# Ensure unique column names before merging
df_tournament_flipped = df_tournament_flipped.loc[:, ~df_tournament_flipped.columns.duplicated()]

# Merge both datasets
df_tournament_balanced = pd.concat([df_tournament, df_tournament_flipped], ignore_index=True)

# Verify the distribution again
print("Balanced y value counts:")
print(df_tournament_balanced["win"].value_counts())

# Extract X and y
X = df_tournament_balanced[["SeedDiff", "WScore_Rolling_7", "LScore_Rolling_7", "Win_Pct_Rolling_7", "WScore", "LScore", "DayNum"]].fillna(0)
y = df_tournament_balanced["win"].astype(int)

# Confirm shape consistency
print("Final Shape of X:", X.shape)  # Should match y
print("Final Shape of y:", y.shape)  # Should match X

# Check unique values in y
print("Unique values in y after conversion:", y.unique())  # Should be [0,1]

# Ensure no NaNs in X or y
print("Missing values in X:\n", X.isnull().sum())
print("Missing values in y:", y.isnull().sum())


# 6. Model Training and Evaluation with Hyperparameter Tuning

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) #Stratify the split
print("y_train counts:", y_train.value_counts()) #Check that train and validation sets have both classes
print("y_val counts:", y_val.value_counts())

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Define a function to train and evaluate a model
def train_evaluate_model(model, param_grid, X_train, y_train, X_val, y_val, tuning_method="GridSearchCV"):
    if tuning_method == "GridSearchCV":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1, error_score='raise')
    elif tuning_method == "RandomizedSearchCV":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1, random_state=42)
    else:
        raise ValueError("Invalid tuning_method. Choose 'GridSearchCV' or 'RandomizedSearchCV'.")

    # Oversampling ONLY within cross-validation
    best_auc = 0
    best_model = None

    for train_index, val_index in cv.split(X_train, y_train):
        train_mask = np.zeros(len(X_train), dtype=bool)
        val_mask = np.zeros(len(X_train), dtype=bool)

        train_mask[train_index] = True
        val_mask[val_index] = True

        X_train_fold, X_val_fold = X_train[train_mask], X_train[val_mask] #Use boolean indexing
        y_train_fold, y_val_fold = y_train[train_mask], y_train[val_mask]

        if len(y_train.unique()) > 1:
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            print("Skipping SMOTE: Only one class in y_train")
            X_train_resampled, y_train_resampled = X_train, y_train
        # smote = SMOTE(random_state=42)
        # X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)

        scaler = StandardScaler()
        X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
        X_val_fold_scaled = scaler.transform(X_val_fold)

        grid_search.fit(X_train_resampled_scaled, y_train_resampled)
        if grid_search.best_score_ > best_auc:
            best_auc = grid_search.best_score_
            best_model = grid_search.best_estimator_

    val_probs = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    val_brier = brier_score_loss(y_val, val_probs)

    print(f"{type(model).__name__} (Best):")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Validation AUC: {val_auc:.4f}")
    print(f"Validation Brier Score: {val_brier:.4f}")

    return best_model, val_auc, val_brier

# 7. Define Hyperparameter Grids (Example - Adjust these!)
lr_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']} #Example
rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]} #Example
lgbm_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [-1, 5, 10], 'learning_rate': [0.01, 0.1, 0.5]} #Example

# 8. Train and Evaluate Models with Tuning
lr_model, lr_auc, lr_brier = train_evaluate_model(LogisticRegression(random_state=42, solver='liblinear'), lr_param_grid, X_train, y_train, X_val, y_val, tuning_method="GridSearchCV")
rf_model, rf_auc, rf_brier = train_evaluate_model(RandomForestClassifier(random_state=42), rf_param_grid, X_train, y_train, X_val, y_val, tuning_method="RandomizedSearchCV") #Example with RandomizedSearchCV
lgbm_model, lgbm_auc, lgbm_brier = train_evaluate_model(lgb.LGBMClassifier(random_state=42), lgbm_param_grid, X_train, y_train, X_val, y_val, tuning_method="GridSearchCV")

# 9. Model Comparison and Selection
best_model = None
best_brier = float('inf')  # Initialize with a large value

for model, auc, brier in [(lr_model, lr_auc, lr_brier), (rf_model, rf_auc, rf_brier), (lgbm_model, lgbm_auc, lgbm_brier)]:
    print(f"\n{type(model).__name__} Performance:")
    print(f"AUC: {auc:.4f}")
    print(f"Brier Score: {brier:.4f}")

    if brier < best_brier:
        best_brier = brier
        best_model = model

print(f"\nBest Model: {type(best_model).__name__}")

import joblib
joblib.dump(best_model, "lgbm_best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✔ Model and Scaler saved successfully!")

new_X_scaled = scaler.transform(X)  # Apply the same preprocessing
predictions = best_model.predict(new_X_scaled)
probabilities = best_model.predict_proba(new_X_scaled)[:, 1]

print("Predictions:", predictions)
print("Win Probabilities:", probabilities)

import matplotlib.pyplot as plt

feature_importances = best_model.feature_importances_
features = ["SeedDiff", "WScore_Rolling_7", "LScore_Rolling_7", "Win_Pct_Rolling_7", "WScore", "LScore", "DayNum"]

plt.figure(figsize=(10, 5))
plt.barh(features, feature_importances, color='royalblue')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("LGBM Feature Importance")
plt.show()

# Ensure the directory exists
output_dir = os.path.join("..", "data", "generated")
os.makedirs(output_dir, exist_ok=True)

# Path to save the CSV file
tournament_file_path = os.path.join(output_dir, "tournament_data.csv")
regular_file_path = os.path.join(output_dir, "regular_season_data.csv")
tournament_balanced_file_path = os.path.join(output_dir, "balanced_tournament_data.csv")

# Save files
df_tournament.to_csv(tournament_file_path, index=False)
df_regular.to_csv(regular_file_path, index=False)
df_tournament_balanced.to_csv(tournament_balanced_file_path, index=False)

print("✔ Preprocessed data saved successfully!")

import shap
print(X.keys())
X["Seed_Adjusted_Score"] = X["WScore_Rolling_7"] / (X["SeedDiff"] + 1)
explainer = shap.Explainer(best_model)
shap_values = explainer(new_X_scaled)
shap.summary_plot(shap_values, X)

shap.dependence_plot("SeedDiff", shap_values, X)


