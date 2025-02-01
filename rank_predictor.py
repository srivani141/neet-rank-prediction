import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. Data Exploration and Analysis

# Load datasets (update with your actual file paths)
current_quiz_data = pd.read_json("current_quiz_data.json")
historical_quiz_data = pd.read_json("historical_quiz_data.json")

# Inspect data (for debugging)
print("Current Quiz Data")
print(current_quiz_data.head())
print(current_quiz_data.info())

print("Historical Quiz Data")
print(historical_quiz_data.head())
print(historical_quiz_data.info())

# 2. Analyzing Performance by Topic and Difficulty

# Performance by topic (assuming there's a 'topic' column and 'score' column)
performance_by_topic = historical_quiz_data.groupby('topic')['score'].mean()

# Plot performance by topic
plt.figure(figsize=(10, 6))
sns.barplot(x=performance_by_topic.index, y=performance_by_topic.values)
plt.xlabel('Topic')
plt.ylabel('Average Score')
plt.title('Average Quiz Score by Topic')
plt.xticks(rotation=90)
plt.show()

# Performance by difficulty (assuming there's a 'difficulty' column and 'score' column)
performance_by_difficulty = historical_quiz_data.groupby('difficulty')['score'].mean()

# Plot performance by difficulty
plt.figure(figsize=(10, 6))
sns.barplot(x=performance_by_difficulty.index, y=performance_by_difficulty.values)
plt.xlabel('Difficulty Level')
plt.ylabel('Average Score')
plt.title('Average Quiz Score by Difficulty')
plt.show()

# 3. Rank Prediction Model

# Prepare the data (assuming 'score' as feature and 'neet_rank' as target)
# Calculate average score across last 5 quizzes for each student
X = historical_quiz_data.groupby('student_id')['score'].mean().values.reshape(-1, 1)

# Target: NEET rank (assuming the 'neet_rank' column exists)
y = historical_quiz_data.groupby('student_id')['neet_rank'].mean().values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Example prediction for a new student based on current quiz data
new_student_score = current_quiz_data['score'].mean()  # You can change this to a custom aggregation
predicted_rank = model.predict([[new_student_score]])
print(f"Predicted NEET Rank: {predicted_rank[0]}")

# 4. Bonus: Predict College Based on Predicted Rank

# Example mapping of rank ranges to colleges (you can modify this as needed)
college_rank_mapping = {
    'College A': (1, 500),
    'College B': (501, 1000),
    'College C': (1001, 1500),
    # Add more colleges as necessary
}

# Function to predict college based on rank
def predict_college(rank):
    for college, (min_rank, max_rank) in college_rank_mapping.items():
        if min_rank <= rank <= max_rank:
            return college
    return 'No suitable college found'

# Predict college based on predicted NEET rank
predicted_college = predict_college(predicted_rank[0])
print(f"Predicted College: {predicted_college}")

