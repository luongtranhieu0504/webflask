from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd
import pickle
import numpy as np
from datetime import datetime


# Khởi tạo Flask app
app = Flask(__name__)

# API key và URL từ Football-Data.org
API_KEY = "f47ce38563c94168b98e3860ea053e56"
BASE_URL = "https://api.football-data.org/v4"
model_file = "logistic_model.pkl"


# Load dữ liệu thống kê đội bóng
stats = pd.read_csv("stats_squads_standard.csv")

# Load model
class LogisticRegressionCustom:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Prevent overflow
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
            self.weights = model_data['weights']
            self.bias = model_data['bias']

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.softmax(linear_model)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# Load the model
model = LogisticRegressionCustom()
model.load_model(model_file)

# Hàm lấy chỉ số đội bóng
def get_team_stats(team_name, stats_df):
    team_data = stats_df[stats_df['Squad'] == team_name]
    if team_data.empty:
        raise ValueError(f"Không tìm thấy dữ liệu cho đội: {team_name}")
    return team_data['xG'].mean(), team_data['xAG'].mean(), team_data['Poss'].mean()

def predict_match(model, team, opponent, stats_df):
    try:
        team_xg, team_xga, team_poss = get_team_stats(team, stats_df)
        opp_xg, opp_xga, opp_poss = get_team_stats(opponent, stats_df)

        diff_xg = team_xg - opp_xga
        diff_xga = team_xga - opp_xg
        diff_poss = team_poss - opp_poss

        input_features = np.array([[diff_xg, diff_xga, diff_poss]])
        probs = model.predict_proba(input_features)[0]

        return {
            "team_win": f"Xác suất thắng của {team}: {probs[0] * 100:.2f}%",
            "draw": f"Xác suất hòa: {probs[1] * 100:.2f}%",
            "opponent_win": f"Xác suất thắng của {opponent}: {probs[2] * 100:.2f}%"
        }

    except ValueError as e:
        return {"error": str(e)}


# Route chính để render trang HTML
@app.route("/")
def index():
    return render_template("index.html")

# Route API trả về dự đoán trận đấu
@app.route("/predict", methods=["GET"])
def predict_upcoming_matches():
    try:
        league_id = request.args.get("league_id", default="PL")
        url = f"{BASE_URL}/competitions/{league_id}/matches"
        print(f"Fetching data from URL: {url}")
        headers = {"X-Auth-Token": API_KEY}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return jsonify({"error": "Unable to fetch data"}), 500
        
        today = datetime.utcnow().date()  # Lấy ngày hôm nay UTC
        end_date = datetime(2024, 12, 31).date() 
        matches_data = response.json()
        upcoming_matches = []

        for match in matches_data.get("matches", []):
            match_date_str = match.get("utcDate")  # Ngày giờ trận đấu từ API (dạng ISO string)
            if match_date_str:
                # Chuyển đổi chuỗi ngày thành đối tượng datetime
                match_date = datetime.fromisoformat(match_date_str[:-1]).date()

        # Lọc các trận đấu từ hôm nay đến ngày 31/12
            if today <= match_date <= end_date:
                home_team = match["homeTeam"]["name"]
                away_team = match["awayTeam"]["name"]
                prediction = predict_match(model, home_team, away_team, stats)
                
                upcoming_matches.append({
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_team_logo": match['homeTeam']['crest'],  # URL logo đội nhà
                    "away_team_logo": match['awayTeam']['crest'],  # URL logo đội khách
                    "utc_date": match['utcDate'],  # Ngày giờ trận đấu
                    "prediction": prediction
                })
        return jsonify({"upcoming_matches": upcoming_matches})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
