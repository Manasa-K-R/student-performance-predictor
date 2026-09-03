import streamlit as st
import pandas as pd
import numpy as np

# ============================================================
# STUDENT PERFORMANCE PREDICTOR
# PROFESSIONAL DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================

file_path = "data/student_data.csv"

try:
    data = pd.read_csv(file_path)
except Exception:
    st.error("Unable to load student_data.csv")
    st.stop()

# ============================================================
# PAGE HEADER
# ============================================================

st.title("🎓 Student Performance Predictor")

st.write(
    "An ML-based application that predicts student performance "
    "using academic factors and study habits."
)

st.divider()

# ============================================================
# SIDEBAR - STUDENT INPUT
# ============================================================

st.sidebar.title("🧑‍🎓 Student Information")
st.sidebar.write("Enter the student's details below.")

study_hours = st.sidebar.slider(
    "📚 Study Hours per Day",
    0.0,
    12.0,
    5.0,
    0.5
)

attendance = st.sidebar.slider(
    "🏫 Attendance (%)",
    0,
    100,
    85
)

internal_marks = st.sidebar.slider(
    "📝 Internal Marks",
    0,
    50,
    40
)

assignment_marks = st.sidebar.slider(
    "📖 Assignment Marks",
    0,
    20,
    18
)

previous_score = st.sidebar.slider(
    "📊 Previous Score (%)",
    0,
    100,
    75
)

sleep_hours = st.sidebar.slider(
    "😴 Sleep Hours per Day",
    0.0,
    12.0,
    7.0,
    0.5
)

screen_time = st.sidebar.slider(
    "📱 Screen Time per Day",
    0.0,
    12.0,
    3.0,
    0.5
)

# ============================================================
# MACHINE LEARNING PREPARATION
# ============================================================

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

X = data[features].values.astype(float)
y = data[target].values.astype(float)

# ============================================================
# TRAINING DATA
# ============================================================

np.random.seed(42)

indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

split_index = int(0.8 * len(X))

X_train = X[:split_index]
y_train = y[:split_index]

# ============================================================
# NORMALIZATION
# ============================================================

mean_X = X_train.mean(axis=0)
std_X = X_train.std(axis=0)

std_X[std_X == 0] = 1

X_train_scaled = (
    X_train - mean_X
) / std_X

X_train_bias = np.column_stack(
    (
        np.ones(len(X_train_scaled)),
        X_train_scaled
    )
)

# ============================================================
# LINEAR REGRESSION
# ============================================================

try:
    beta = np.linalg.solve(
        X_train_bias.T @ X_train_bias,
        X_train_bias.T @ y_train
    )
except np.linalg.LinAlgError:
    beta = np.linalg.pinv(
        X_train_bias.T @ X_train_bias
    ) @ (
        X_train_bias.T @ y_train
    )

# ============================================================
# PREDICT NEW STUDENT
# ============================================================

new_student = np.array([[
    study_hours,
    attendance,
    internal_marks,
    assignment_marks,
    previous_score,
    sleep_hours,
    screen_time
]])

new_student_scaled = (
    new_student - mean_X
) / std_X

new_student_bias = np.column_stack(
    (
        np.ones(1),
        new_student_scaled
    )
)

predicted_score = (
    new_student_bias @ beta
)[0]

predicted_score = float(
    np.clip(predicted_score, 0, 100)
)

# ============================================================
# PERFORMANCE LEVEL
# ============================================================

if predicted_score >= 90:
    performance = "Excellent 🌟"
elif predicted_score >= 80:
    performance = "Very Good 🏆"
elif predicted_score >= 70:
    performance = "Good 👍"
elif predicted_score >= 60:
    performance = "Average 🙂"
elif predicted_score >= 50:
    performance = "Needs Improvement ⚠️"
else:
    performance = "At Risk 🚨"

# ============================================================
# TOP METRICS
# ============================================================

st.subheader("📊 Student Performance Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Predicted Score",
        f"{predicted_score:.2f}%"
    )

with col2:
    st.metric(
        "Average Score",
        f"{data['final_score'].mean():.2f}%"
    )

with col3:
    st.metric(
        "Highest Score",
        f"{data['final_score'].max():.0f}%"
    )

with col4:
    st.metric(
        "Lowest Score",
        f"{data['final_score'].min():.0f}%"
    )

# ============================================================
# PERFORMANCE RESULT
# ============================================================

st.divider()

col1, col2 = st.columns(2)

with col1:

    st.subheader("🎯 Prediction Result")

    st.metric(
        "Expected Final Score",
        f"{predicted_score:.2f}%"
    )

    st.write("Performance Level:")
    st.success(performance)

    st.progress(
        int(predicted_score)
    )

with col2:

    st.subheader("💡 Personalized Recommendations")

    recommendations = []

    if study_hours < 3:
        recommendations.append(
            "Increase your daily study hours."
        )

    if attendance < 75:
        recommendations.append(
            "Improve your class attendance."
        )

    if internal_marks < 40:
        recommendations.append(
            "Focus more on internal examinations."
        )

    if assignment_marks < 15:
        recommendations.append(
            "Complete assignments regularly."
        )

    if sleep_hours < 6:
        recommendations.append(
            "Try to maintain at least 6 hours of sleep."
        )

    if screen_time > 6:
        recommendations.append(
            "Reduce unnecessary screen time."
        )

    if not recommendations:
        recommendations.append(
            "Excellent habits! Keep maintaining your current routine."
        )

    for recommendation in recommendations:
        st.info("✓ " + recommendation)

# ============================================================
# DATASET ANALYSIS
# ============================================================

st.divider()

st.subheader("📈 Academic Performance Analysis")

tab1, tab2, tab3 = st.tabs([
    "📚 Study Analysis",
    "🏫 Attendance Analysis",
    "📊 Score Distribution"
])

# ============================================================
# TAB 1 - STUDY HOURS
# ============================================================

with tab1:

    st.write(
        "Relationship between study hours and final academic score."
    )

    study_chart = data[
        ["study_hours", "final_score"]
    ].set_index("study_hours")

    st.scatter_chart(
        study_chart
    )

    average_by_study = (
        data.groupby("study_hours")["final_score"]
        .mean()
    )

    st.write("Average score by study hours:")
    st.bar_chart(
        average_by_study
    )

# ============================================================
# TAB 2 - ATTENDANCE
# ============================================================

with tab2:

    st.write(
        "Relationship between attendance and final academic score."
    )

    attendance_chart = data[
        ["attendance", "final_score"]
    ].set_index("attendance")

    st.scatter_chart(
        attendance_chart
    )

# ============================================================
# TAB 3 - SCORE DISTRIBUTION
# ============================================================

with tab3:

    st.write(
        "Distribution of final scores among students."
    )

    st.bar_chart(
        data["final_score"].value_counts()
        .sort_index()
    )

# ============================================================
# DATASET TABLE
# ============================================================

st.divider()

st.subheader("📋 Student Dataset")

st.dataframe(
    data,
    use_container_width=True
)

# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("ℹ️ Project Information")

info_col1, info_col2 = st.columns(2)

with info_col1:

    st.write("**Machine Learning Approach**")
    st.write("Linear Regression using NumPy")

    st.write("**Input Features**")

    for feature in features:
        st.write("• " + feature)

with info_col2:

    st.write("**Dataset Statistics**")

    st.write(
        f"Total Students: {len(data)}"
    )

    st.write(
        f"Average Final Score: "
        f"{data['final_score'].mean():.2f}%"
    )

    st.write(
        f"Highest Final Score: "
        f"{data['final_score'].max():.0f}%"
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Student Performance Predictor | "
    "Python + Pandas + NumPy + Streamlit | "
    "AIML Project"
)