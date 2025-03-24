import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Path to the dataset
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'nba_games.csv')

# Load the data
df = pd.read_csv(DATA_PATH)

# Drop rows with missing values
df.dropna(inplace=True)

# Drop columns that are currently not needed
df.drop(['GAME_DATE_EST', 'GAME_ID', 'GAME_STATUS_TEXT', 'SEASON'], axis=1, inplace=True)

# Create point difference feature
df['Point Difference'] = df['PTS_home'] - df['PTS_away']

# Scale the data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[['PTS_home', 'FG_PCT_home', 'FT_PCT_home',
                                           'FG3_PCT_home', 'AST_home', 'REB_home',
                                           'PTS_away', 'FG_PCT_away', 'FT_PCT_away',
                                           'FG3_PCT_away', 'AST_away', 'REB_away', 'Point Difference']])
df[['PTS_home', 'FG_PCT_home', 'FT_PCT_home',
                                           'FG3_PCT_home', 'AST_home', 'REB_home',
                                           'PTS_away', 'FG_PCT_away', 'FT_PCT_away',
                                           'FG3_PCT_away', 'AST_away', 'REB_away', 'Point Difference']] = scaled_features

# Split into training and testing sets
X = df[['TEAM_ID_home', 'TEAM_ID_away', 'PTS_home', 'FG_PCT_home',
        'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home',
        'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away',
        'AST_away', 'REB_away']]
y = df['HOME_TEAM_WINS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Save data after processing
processed_path = os.path.join(ROOT_DIR, 'data', 'processed_nba_games.csv')
df.to_csv(processed_path, index=False)

print(f"Processed data saved to {processed_path}")


