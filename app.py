import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# STUDENT PERFORMANCE PREDICTION SYSTEM
# ============================================================

print("=" * 60)
print("        STUDENT PERFORMANCE PREDICTION SYSTEM")
print("=" * 60)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

file_path = "data/student_data.csv"

if not os.path.exists(file_path):
    print("\nERROR: Dataset file not found!")
    print("Make sure student_data.csv is inside the data folder.")
    input("\nPress Enter to exit...")
    exit()

data = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Number of students:", len(data))

# ------------------------------------------------------------
# 2. DISPLAY DATASET
# ------------------------------------------------------------

print("\nStudent Performance Dataset:")
print(data.to_string(index=False))

# ------------------------------------------------------------
# 3. SELECT INPUT FEATURES AND TARGET
# ------------------------------------------------------------

features = [
    "study_hours",
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
    "sleep_hours",
    "screen_time"
]

target = "final_score"

# Check whether required columns exist
missing_columns = [column for column in features + [target]
                   if column not in data.columns]

if missing_columns:
    print("\nERROR: Missing columns:")
    print(missing_columns)
    input("\nPress Enter to exit...")
    exit()

X = data[features].values.astype(float)
y = data[target].values.astype(float)

# ------------------------------------------------------------
# 4. TRAIN-TEST SPLIT
# ------------------------------------------------------------

# Use a fixed random seed so the result is reproducible
np.random.seed(42)

indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# 80% training and 20% testing
split_index = int(0.8 * len(X))

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("\n" + "-" * 60)
print("DATA SPLIT")
print("-" * 60)
print("Training data:", len(X_train))
print("Testing data :", len(X_test))

# ------------------------------------------------------------
# 5. NORMALIZE FEATURES
# ------------------------------------------------------------

# Normalization helps the linear regression calculation
mean_X = X_train.mean(axis=0)
std_X = X_train.std(axis=0)

# Prevent division by zero
std_X[std_X == 0] = 1

X_train_scaled = (X_train - mean_X) / std_X
X_test_scaled = (X_test - mean_X) / std_X

# Add a column of 1s for the intercept/bias
X_train_bias = np.column_stack(
    (np.ones(len(X_train_scaled)), X_train_scaled)
)

X_test_bias = np.column_stack(
    (np.ones(len(X_test_scaled)), X_test_scaled)
)

# ------------------------------------------------------------
# 6. LINEAR REGRESSION
# ------------------------------------------------------------

# Calculate coefficients using the Normal Equation
# beta = (X^T X)^-1 X^T y

try:
    beta = np.linalg.solve(
        X_train_bias.T @ X_train_bias,
        X_train_bias.T @ y_train
    )
except np.linalg.LinAlgError:
    beta = np.linalg.pinv(X_train_bias.T @ X_train_bias) @ (
        X_train_bias.T @ y_train
    )

# ------------------------------------------------------------
# 7. PREDICTION
# ------------------------------------------------------------

predictions = X_test_bias @ beta

# Keep predictions between 0 and 100
predictions = np.clip(predictions, 0, 100)

# ------------------------------------------------------------
# 8. CALCULATE MODEL PERFORMANCE
# ------------------------------------------------------------

# Mean Absolute Error
mae = np.mean(np.abs(y_test - predictions))

# Mean Squared Error
mse = np.mean((y_test - predictions) ** 2)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# R2 Score
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual = np.sum((y_test - predictions) ** 2)

if ss_total != 0:
    r2 = 1 - (ss_residual / ss_total)
else:
    r2 = 0

# ------------------------------------------------------------
# 9. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("              MODEL PERFORMANCE")
print("=" * 60)

print(f"Mean Absolute Error : {mae:.2f}")
print(f"Mean Squared Error  : {mse:.2f}")
print(f"Root Mean Squared Error : {rmse:.2f}")
print(f"R2 Score            : {r2:.2f}")

# ------------------------------------------------------------
# 10. ACTUAL VS PREDICTED SCORES
# ------------------------------------------------------------

results = pd.DataFrame({
    "Actual Score": y_test,
    "Predicted Score": np.round(predictions, 2)
})

print("\n" + "=" * 60)
print("           ACTUAL VS PREDICTED SCORES")
print("=" * 60)

print(results.to_string(index=False))

# ------------------------------------------------------------
# 11. STUDY HOURS VS FINAL SCORE GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    data["study_hours"],
    data["final_score"],
    alpha=0.8
)

plt.xlabel("Study Hours per Day")
plt.ylabel("Final Score")
plt.title("Study Hours vs Final Score")
plt.grid(True)

plt.tight_layout()

# Save graph
plt.savefig("study_hours_vs_final_score.png")

plt.show()

# ------------------------------------------------------------
# 12. ACTUAL VS PREDICTED GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    predictions,
    alpha=0.8
)

# Perfect prediction line
minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("Actual vs Predicted Final Scores")
plt.grid(True)

plt.tight_layout()

# Save graph
plt.savefig("actual_vs_predicted.png")

plt.show()

# ------------------------------------------------------------
# 13. CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("        CORRELATION WITH FINAL SCORE")
print("=" * 60)

correlations = data[features + [target]].corr()[target]

for feature in features:
    print(f"{feature:20s}: {correlations[feature]:.3f}")

# ------------------------------------------------------------
# 14. STUDENT PERFORMANCE PREDICTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("       PREDICT FINAL SCORE FOR A NEW STUDENT")
print("=" * 60)

choice = input("\nDo you want to predict a new student's score? (yes/no): ")

if choice.lower() in ["yes", "y"]:

    try:
        study_hours = float(
            input("Enter study hours per day: ")
        )

        attendance = float(
            input("Enter attendance percentage: ")
        )

        internal_marks = float(
            input("Enter internal marks: ")
        )

        assignment_marks = float(
            input("Enter assignment marks: ")
        )

        previous_score = float(
            input("Enter previous score: ")
        )

        sleep_hours = float(
            input("Enter sleep hours: ")
        )

        screen_time = float(
            input("Enter screen time per day: ")
        )

        new_student = np.array([[
            study_hours,
            attendance,
            internal_marks,
            assignment_marks,
            previous_score,
            sleep_hours,
            screen_time
        ]])

        # Scale new student's data using training statistics
        new_student_scaled = (
            new_student - mean_X
        ) / std_X

        new_student_bias = np.column_stack(
            (np.ones(1), new_student_scaled)
        )

        predicted_score = (
            new_student_bias @ beta
        )[0]

        predicted_score = float(
            np.clip(predicted_score, 0, 100)
        )

        print("\n" + "-" * 60)
        print("PREDICTION RESULT")
        print("-" * 60)

        print(
            f"Predicted Final Score: {predicted_score:.2f}"
        )

        # Performance category
        if predicted_score >= 90:
            category = "Excellent"
        elif predicted_score >= 80:
            category = "Very Good"
        elif predicted_score >= 70:
            category = "Good"
        elif predicted_score >= 60:
            category = "Average"
        elif predicted_score >= 50:
            category = "Needs Improvement"
        else:
            category = "At Risk"

        print("Performance Level:", category)

        # Suggestions
        print("\nRecommendation:")

        if study_hours < 3:
            print("- Increase daily study hours.")

        if attendance < 75:
            print("- Improve class attendance.")

        if internal_marks < 40:
            print("- Focus more on internal examinations.")

        if assignment_marks < 15:
            print("- Complete assignments regularly.")

        if sleep_hours < 6:
            print("- Maintain sufficient sleep.")

        if screen_time > 6:
            print("- Reduce unnecessary screen time.")

        if (
            study_hours >= 3
            and attendance >= 75
            and internal_marks >= 40
            and assignment_marks >= 15
            and sleep_hours >= 6
            and screen_time <= 6
        ):
            print("- Keep maintaining your current study habits.")

    except ValueError:
        print("\nPlease enter numbers only.")

# ------------------------------------------------------------
# 15. FINISH
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("       STUDENT PERFORMANCE PREDICTION SYSTEM")
print("                  COMPLETED")
print("=" * 60)

input("\nPress Enter to close...")