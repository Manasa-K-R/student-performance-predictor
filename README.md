# 🎓 Student Performance Predictor

## 📌 Project Overview

The **Student Performance Predictor** is a Python-based machine learning project designed to estimate a student's expected final score based on academic and lifestyle factors.

The system uses student performance data to identify patterns and provide an estimated final score for a new student.

It also provides performance analysis, data visualizations, personalized recommendations, and SQLite database functionality for storing student prediction records.

## ✨ Features

### 📊 Student Score Prediction

The system predicts the expected final score based on:

- Study Hours
- Attendance
- Internal Marks
- Assignment Marks
- Previous Score
- Sleep Hours
- Screen Time

### 📈 Performance Analysis

The application provides visual analysis of the student dataset, including:

- Academic Performance Analysis
- Attendance Analysis
- Score Distribution

### 🎯 Performance Classification

Based on the predicted score, the system classifies student performance into different levels.

### 💡 Personalized Recommendations

The system provides suggestions based on the student's academic and lifestyle inputs.

Examples:

- Increase study hours
- Improve attendance
- Reduce unnecessary screen time
- Maintain proper sleep
- Improve academic performance

### 🗄️ SQLite Database

The project includes a separate SQLite database component for:

- Adding student prediction records
- Saving student information
- Viewing saved student records

## 🛠️ Technologies Used

- Python
- Pandas
- Numpy
- Matplotlib
- Streamlit
- SQLite


## 🧠 How the System Works

The system follows these basic steps:

```text
Student Dataset
       ↓
Data Loading and Processing
       ↓
Prediction Model
       ↓
User Enters New Student Details
       ↓
Expected Final Score
       ↓
Performance Classification
       ↓
Personalized recommendations