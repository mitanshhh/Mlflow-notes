# Import Scikit-learn flavor
import mlflow.sklearn

# Set the experiment to "Sklearn Model"
mlflow.set_experiment("Sklearn Model")

# Set Auto logging for Scikit-learn flavor 
mlflow.sklearn.autolog()

lr = LinearRegression()
lr.fit(X_train, y_train)

# Get a prediction from test data
print(lr.predict(X_test.iloc[[5]]))
