print("App is starting...")

from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

df = pd.read_csv("insurance_data_linear.csv")
df = pd.get_dummies(df, drop_first=True)

X = df.drop("charges", axis=1)
y = df["charges"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        sex = request.form["sex"]
        smoker = request.form["smoker"]
        region = request.form["region"]

        sex_male = 1 if sex == "male" else 0
        smoker_yes = 1 if smoker == "yes" else 0
        region_northwest = 1 if region == "northwest" else 0
        region_southeast = 1 if region == "southeast" else 0
        region_southwest = 1 if region == "southwest" else 0

        input_data = [[age, bmi, children, sex_male, smoker_yes,
                       region_northwest, region_southeast, region_southwest]]

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        result = round(float(prediction), 2)

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)