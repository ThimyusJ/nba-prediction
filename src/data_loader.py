import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import processed_path


def load_data():
    df = pd.read_csv(processed_path)

    # Define features and targets
    X = df[['TEAM_ID_home', 'TEAM_ID_away', 'PTS_home', 'FG_PCT_home',
            'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home',
            'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away',
            'AST_away', 'REB_away', 'Point Difference']]
    y = df['HOME_TEAM_WINS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test
