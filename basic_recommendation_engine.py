import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset from file
data = pd.read_csv("ott_user_data.csv")

# Features and target
X = data[['watch_time', 'genre_match', 'avg_user_rating', 'show_popularity']]
y = data['user_rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)

# Predict for a new user
new_user = [[100, 0.85, 4.1, 80]]
predicted_rating = model.predict(new_user)

print("Predicted User Rating:", predicted_rating[0])
