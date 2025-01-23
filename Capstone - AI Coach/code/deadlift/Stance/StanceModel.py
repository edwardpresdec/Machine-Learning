import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

MODEL_PATH = "deadliftStance.pkl"

# Step 1: Load the Data
try:
    df = pd.read_csv('deadliftStance.csv')
except Exception as e:
    print(e)

# Step 2: Prepare the Data
X = df.iloc[:, 1:]  # Features (all columns except the first)
y = df.iloc[:, 0]   # Target variable (first column)

# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

pipeline = Pipeline([
    ('scaler', StandardScaler()),              # Step 1: Standardize the features
    ('classifier', RandomForestClassifier())  # Step 2: Random Forest Classifier
])

model = pipeline.fit(X_train,y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test.values, y_pred)
precision = precision_score(y_test.values,y_pred,average="weighted")
recall = recall_score(y_test.values,y_pred,average="weighted")

print("Accuracy:", accuracy)
print("\nPrecision: ",precision)
print("\nRecall: ",recall)

with open(MODEL_PATH, 'wb') as file:
    pickle.dump(model, file)


