# Create Python Class
class CustomPredict(mlflow.pyfunc.PythonModel):
    # Set method for loading model
    def load_context(self, context):
        self.model = mlflow.sklearn.load_model("./lr_model/")
    # Set method for custom inference     
    def predict(self, context, model_input):
        predictions = self.model.predict(model_input)
        decoded_predictions = []  
        for prediction in predictions:
            if prediction == 0:
                decoded_predictions.append("female")
            else:
                decoded_predictions.append("male")
        return decoded_predictions