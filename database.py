import sqlite3

DATABASE_NAME = "student_predictions.db"


def create_database():
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            study_hours REAL,
            attendance REAL,
            internal_marks REAL,
            assignment_marks REAL,
            previous_score REAL,
            sleep_hours REAL,
            screen_time REAL,
            predicted_score REAL,
            performance TEXT
        )
    """)

    connection.commit()
    connection.close()


def add_student():
    print("\n" + "=" * 60)
    print("       STUDENT PERFORMANCE DATABASE")
    print("=" * 60)

    student_name = input("\nEnter student name: ")

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
        input("Enter sleep hours per day: ")
    )

    screen_time = float(
        input("Enter screen time per day: ")
    )

    predicted_score = float(
        input("Enter predicted final score: ")
    )

    if predicted_score >= 90:
        performance = "Excellent"
    elif predicted_score >= 80:
        performance = "Very Good"
    elif predicted_score >= 70:
        performance = "Good"
    elif predicted_score >= 60:
        performance = "Average"
    elif predicted_score >= 50:
        performance = "Needs Improvement"
    else:
        performance = "At Risk"

    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            student_name,
            study_hours,
            attendance,
            internal_marks,
            assignment_marks,
            previous_score,
            sleep_hours,
            screen_time,
            predicted_score,
            performance
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        student_name,
        study_hours,
        attendance,
        internal_marks,
        assignment_marks,
        previous_score,
        sleep_hours,
        screen_time,
        predicted_score,
        performance
    ))

    connection.commit()
    connection.close()

    print("\n" + "-" * 60)
    print("✅ STUDENT RECORD SAVED SUCCESSFULLY!")
    print("-" * 60)

    print("Student Name:", student_name)
    print(f"Predicted Score: {predicted_score:.2f}%")
    print("Performance:", performance)


def view_students():
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            student_name,
            study_hours,
            attendance,
            internal_marks,
            assignment_marks,
            previous_score,
            sleep_hours,
            screen_time,
            predicted_score,
            performance
        FROM predictions
        ORDER BY id DESC
    """)

    students = cursor.fetchall()
    connection.close()

    print("\n" + "=" * 100)
    print("                    SAVED STUDENT RECORDS")
    print("=" * 100)

    if not students:
        print("\nNo student records found.")
        return

    for student in students:
        print("\nID:", student[0])
        print("Student Name:", student[1])
        print("Study Hours:", student[2])
        print("Attendance:", student[3], "%")
        print("Internal Marks:", student[4])
        print("Assignment Marks:", student[5])
        print("Previous Score:", student[6])
        print("Sleep Hours:", student[7])
        print("Screen Time:", student[8])
        print("Predicted Score:", student[9], "%")
        print("Performance:", student[10])
        print("-" * 60)


def main():
    create_database()

    while True:

        print("\n" + "=" * 60)
        print("       STUDENT PERFORMANCE PREDICTOR DATABASE")
        print("=" * 60)

        print("\n1. Add Student Prediction")
        print("2. View Saved Students")
        print("3. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":

            try:
                add_student()

            except ValueError:
                print("\n❌ Please enter valid numerical values.")

        elif choice == "2":

            view_students()

        elif choice == "3":

            print("\nThank you for using Student Performance Predictor! 🎓")
            break

        else:

            print("\n❌ Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()