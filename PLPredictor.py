#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import tkinter as tk
from tkinter import ttk

results = pd.read_csv("results.csv", index_col=None)

results.head()

results.shape

results["season"].value_counts()

results["result"].value_counts()

results["away_team"].value_counts()

results["home_team"].value_counts()

print(results.columns)

print(results.head())

results.columns = results.columns.str.strip()

print(results.head())

results["season"].value_counts()

results.dtypes

results["result_code"] = results["result"].astype("category").cat.codes

results

results["opponent_code"] = results["away_team"].astype("category").cat.codes

results

results["target"] = (results["result"] == "H").astype("int")

results["target"] = (results["result"] == "A").astype("int")

results["season"] = results["season"].str.replace("–", "-")

results["season_start"] = results["season"].str.split("-").str[0].astype(int)

results["season_end"] = results["season"].str.split("-").str[1].astype(int)

results["year_code"] = results["season_start"]

results['season_start'] = results['season'].str.split('-').str[0].astype(int)

results['season_end'] = results['season'].str.split('-').str[1].astype(int)

print(results.head())

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)

train = results[results["season"] < '2016-2017']

test = results[results["season"] > '2016-2017']

predictors = ["opponent_code", "season_start", "season_end"]

rf.fit(train[predictors], train["target"])

preds = rf.predict(test[predictors])

from sklearn.metrics import accuracy_score

acc = accuracy_score(test["target"], preds)

acc

combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))

print(test["target"].value_counts())

from collections import Counter

print(Counter(preds))

importances = rf.feature_importances_

print(importances)

print(test["target"].value_counts())

print(train["target"].value_counts())

pd.crosstab(index=combined["actual"], columns=combined["prediction"])

from sklearn.metrics import precision_score

precision_score(test["target"], preds)

grouped_matches = results.groupby("home_team")

group = grouped_matches.get_group("Liverpool")

group

def rolling_averages(group, cols, new_cols):
    group=group.sort_values("season")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ["home_goals", "away_goals"]

new_cols = [f"{c}_rolling" for c in cols]

new_cols

rolling_averages(group, cols, new_cols)

results_rolling = results.groupby("home_team").apply(lambda x: rolling_averages(x, cols, new_cols))

results_rolling.index = range(results_rolling.shape[0])

results_rolling

def make_predictions(data, predictors):
    train = data[data["season"] < '2016-2017']
    test=data[data["season"] > '2016-2017']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted = preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision

combined, precision = make_predictions(results_rolling, predictors + new_cols)

precision

combined

combined = combined.merge(results_rolling[["home_team", "away_team", "result", "season"]], left_index=True, right_index=True)

combined

class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Tottenham Hotspur": "Spurs",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}

mapping = MissingDict(**map_values)

mapping["West Ham United"]

mapping["Brighton and Hove Albion"]

combined["new_home_team"] = combined["home_team"].map(mapping)
combined["new_away_team"] = combined["away_team"].map(mapping)

combined["new_home_team"] = combined["home_team"].map(mapping)

combined["new_home_team"] = combined["home_team"].map(mapping)

combined

merged = combined.merge(combined, left_on=["season", "new_home_team"], right_on=["season","away_team"])

merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]["actual_x"].value_counts()

7007/7841

7007/7841

results.columns

print(combined.columns)

print(results.columns)

print(merged.columns)

predictors = ["opponent_code", "season_start", "season_end"] + new_cols

def predict_match():
    home_team = home_team_var.get()
    away_team = away_team_var.get()
    season = season_entry.get()
    
    # Create a DataFrame with the input data
    matchup_data = pd.DataFrame({
        'home_team': [home_team],
        'away_team': [away_team],
        'season': [season],
        'home_goals_rolling': [0], # Replace with actual rolling averages if available
        'away_goals_rolling': [0] # Replace with actual rolling averages if available
    })
    
    # Preprocess the input data
    matchup_data['season_start'] = matchup_data['season'].str.split('-').str[0].astype(int)
    matchup_data['season_end'] = matchup_data['season'].str.split('-').str[1].astype(int)
    
    # Ensure opponent_code is calculated correctly
    matchup_data['opponent_code'] = results['away_team'].astype('category').cat.codes[results['away_team'] == away_team].iloc[0]
    
    # Make prediction
    prediction_proba = rf.predict_proba(matchup_data[predictors])
    away_win_probability = prediction_proba[0][1]
    print(f"Predicted probabilities: {prediction_proba}") # Debugging line
    result = "Away team wins" if away_win_probability > 0.5 else "Home team wins"
    result_label.config(text=f"Prediction: {result}")
    probability_label.config(text=f"Probability of away team winning: {away_win_probability:.2f}")

# Create the main window
root = tk.Tk()
root.title("Match Predictor")

# Create and place widgets
home_team_label = ttk.Label(root, text="Home Team:")
home_team_label.grid(row=0, column=0, padx=5, pady=5)

home_teams = sorted(results['home_team'].unique())
home_team_var = tk.StringVar()
home_team_dropdown = ttk.Combobox(root, textvariable=home_team_var, values=home_teams)
home_team_dropdown.grid(row=0, column=1, padx=5, pady=5)

away_team_label = ttk.Label(root, text="Away Team:")
away_team_label.grid(row=1, column=0, padx=5, pady=5)

away_teams = sorted(results['away_team'].unique())
away_team_var = tk.StringVar()
away_team_dropdown = ttk.Combobox(root, textvariable=away_team_var, values=away_teams)
away_team_dropdown.grid(row=1, column=1, padx=5, pady=5)

season_label = ttk.Label(root, text="Season (e.g., 2023-2024):")
season_label.grid(row=2, column=0, padx=5, pady=5)

season_entry = ttk.Entry(root)
season_entry.grid(row=2, column=1, padx=5, pady=5)

predict_button = ttk.Button(root, text="Predict", command=predict_match)
predict_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

result_label = ttk.Label(root, text="")
result_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

probability_label = ttk.Label(root, text="")
probability_label.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

# Start the GUI event loop
root.mainloop()

print(results["target"].value_counts())

acc
