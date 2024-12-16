from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

app = Flask(__name__)

# Load and prepare the dataset
file_path = 'Salary Prediction Data.csv'  # Ensure this file is in the project directory
data = pd.read_csv(file_path)

# Drop rows with missing target values (Salary)
data = data.dropna(subset=["Salary"])

# Split features and target
X = data.drop(columns=["Salary"])
y = data["Salary"]

# Identify numeric and categorical columns
numeric_features = ["Age", "Years of Experience"]
categorical_features = ["Gender", "Education Level", "Job Title"]

# Preprocessing for numeric data
numeric_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Combine preprocessors in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Define the model
model = RandomForestRegressor(random_state=42)

# Create the pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
pipeline.fit(X_train, y_train)

# Optionally, evaluate the model
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model MAE on test set: {mae:.2f}")

# Extract unique options for categorical features
unique_genders = sorted(data["Gender"].dropna().unique())
unique_education = sorted(data["Education Level"].dropna().unique())
unique_job_titles = sorted(data["Job Title"].dropna().unique())

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        age = request.form.get('age')
        gender = request.form.get('gender')
        education = request.form.get('education')
        job_title = request.form.get('job_title')
        experience = request.form.get('experience')
        
        # Input validation
        try:
            age = int(age)
            experience = float(experience)
        except ValueError:
            return render_template('index.html', 
                                   genders=unique_genders,
                                   educations=unique_education,
                                   job_titles=unique_job_titles,
                                   error="Please enter valid numeric values for Age and Years of Experience.")
        
        # Create DataFrame for prediction
        example_input = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Education Level": education,
            "Job Title": job_title,
            "Years of Experience": experience
        }])
        
        # Make prediction
        predicted_salary = pipeline.predict(example_input)[0]
        
        return render_template('result.html', salary=predicted_salary)
    
    return render_template('index.html', 
                           genders=unique_genders,
                           educations=unique_education,
                           job_titles=unique_job_titles)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

