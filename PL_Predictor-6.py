import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import customtkinter as ctk

results = pd.read_csv("results.csv", index_col=None)
results.columns = results.columns.str.strip()
results["result_code"] = results["result"].astype("category").cat.codes
results["opponent_code"] = results["away_team"].astype("category").cat.codes
results["target"] = (results["result"] == "A").astype("int")
results["season"] = results["season"].str.replace("–", "-")
results['season_start'] = results['season'].str.split('-').str[0].astype(int)
results['season_end'] = results['season'].str.split('-').str[1].astype(int)

rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)
train = results[results["season"] < '2016-2017']
test = results[results["season"] > '2016-2017']
predictors = ["opponent_code", "season_start", "season_end"]
rf.fit(train[predictors], train["target"])
preds = rf.predict(test[predictors])
acc = accuracy_score(test["target"], preds)

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("season")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ["home_goals", "away_goals"]
new_cols = [f"{c}_rolling" for c in cols]
results_rolling = results.groupby("home_team").apply(lambda x: rolling_averages(x, cols, new_cols))
results_rolling.index = range(results_rolling.shape[0])

predictors = ["opponent_code", "season_start", "season_end"] + new_cols

def validate_season(season):
    try:
        start, end = map(int, season.split('-'))
        if start >= end or end != start + 1:
            return False
        if start < 1888 or end > 2100:
            return False
        return True
    except ValueError:
        return False

def get_latest_rolling_average(team, is_home):
    team_data = results_rolling[results_rolling['home_team' if is_home else 'away_team'] == team]
    if team_data.empty:
        return 0
    return team_data.iloc[-1][f'{"home" if is_home else "away"}_goals_rolling']

def predict_match():
    home_team = home_team_var.get()
    away_team = away_team_var.get()
    season = season_entry.get()

    if not validate_season(season):
        result_label.configure(text="Invalid season format. Use YYYY-YYYY.")
        probability_label.configure(text="")
        return

    matchup_data = pd.DataFrame({
        'home_team': [home_team],
        'away_team': [away_team],
        'season': [season],
        'home_goals_rolling': [get_latest_rolling_average(home_team, True)],
        'away_goals_rolling': [get_latest_rolling_average(away_team, True)],
    })

    matchup_data['season_start'] = matchup_data['season'].str.split('-').str[0].astype(int)
    matchup_data['season_end'] = matchup_data['season'].str.split('-').str[1].astype(int)
    matchup_data['opponent_code'] = results['away_team'].astype('category').cat.codes[results['away_team'] == away_team].iloc[0]

    prediction_proba = rf.predict_proba(matchup_data[predictors])
    away_win_probability = prediction_proba[0][1]

    print(f"Predicted probabilities: {prediction_proba}")
    result = "Away team wins" if away_win_probability > 0.5 else "Home team wins"
    result_label.configure(text=f"Prediction: {result}")
    probability_label.configure(text=f"Probability of away team winning: {away_win_probability:.2f}")

root = ctk.CTk()
root.title("Match Predictor")
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

home_team_label = ctk.CTkLabel(root, text="Home Team:")
home_team_label.grid(row=0, column=0, padx=5, pady=5)
home_teams = sorted(results['home_team'].unique())
home_team_var = ctk.StringVar()
home_team_dropdown = ctk.CTkComboBox(root, variable=home_team_var, values=home_teams)
home_team_dropdown.grid(row=0, column=1, padx=5, pady=5)

away_team_label = ctk.CTkLabel(root, text="Away Team:")
away_team_label.grid(row=1, column=0, padx=5, pady=5)
away_teams = sorted(results['away_team'].unique())
away_team_var = ctk.StringVar()
away_team_dropdown = ctk.CTkComboBox(root, variable=away_team_var, values=away_teams)
away_team_dropdown.grid(row=1, column=1, padx=5, pady=5)

season_label = ctk.CTkLabel(root, text="Season (e.g., 2023-2024):")
season_label.grid(row=2, column=0, padx=5, pady=5)
season_entry = ctk.CTkEntry(root)
season_entry.grid(row=2, column=1, padx=5, pady=5)

predict_button = ctk.CTkButton(root, text="Predict", command=predict_match)
predict_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

result_label = ctk.CTkLabel(root, text="")
result_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

probability_label = ctk.CTkLabel(root, text="")
probability_label.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()
