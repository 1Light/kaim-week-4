import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class SalesPredictionModel:
    def __init__(self, data_path, loss_function='mse'):
        """
        Initializes the SalesPredictionModel with the dataset and loss function.
        
        :param data_path: The path to the dataset.
        :param loss_function: The loss function to use for model evaluation. Options are 'mse' or 'mae'.
        """
        self.data_path = data_path
        self.df = None
        self.model = None
        self.loss_function = loss_function
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """ Loads the dataset. """
        try:
            logger.info(f"Loading data from {self.data_path}...")
            self.df = pd.read_csv(self.data_path)
            logger.info("Data loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading data: {e}")

    def preprocess_data(self):
        """ Preprocess the data: handle missing values, encode categorical features, etc. """
        logger.info("Preprocessing data...")
        self.df.fillna(0, inplace=True)  # Simple imputation
        X = self.df.drop(columns=['Sales', 'Date'])  # Features
        y = self.df['Sales']  # Target
        
        # One-hot encoding for categorical variables
        X = pd.get_dummies(X, drop_first=True)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info("Data preprocessing completed.")

    def train_model(self):
        """ Train the regression model using Random Forest Regressor. """
        logger.info("Training the model...")
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        logger.info("Model trained successfully.")

    def evaluate_model(self):
        """ Evaluate the model using the selected loss function. """
        logger.info(f"Evaluating the model using {self.loss_function}...")

        # Predictions
        y_pred = self.model.predict(self.X_test)

        if self.loss_function == 'mse':
            loss = mean_squared_error(self.y_test, y_pred)
            logger.info(f"Mean Squared Error: {loss}")
        elif self.loss_function == 'mae':
            loss = mean_absolute_error(self.y_test, y_pred)
            logger.info(f"Mean Absolute Error: {loss}")
        else:
            logger.error("Invalid loss function. Use 'mse' or 'mae'.")

    def save_model(self):
        """ Save the trained model with a timestamp. """
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f")  # Format as "10-08-2020-16-32-31-00"
        model_filename = f"sales_prediction_model_{timestamp}.pkl"
        try:
            logger.info(f"Saving model to {model_filename}...")
            joblib.dump(self.model, model_filename)
            logger.info("Model saved successfully!")
        except Exception as e:
            logger.error(f"Error saving model: {e}")


# Run the model pipeline
if __name__ == "__main__":
    data_path = os.path.abspath("../cleaned_data/primary_data.csv")  # Update with your path
    model = SalesPredictionModel(data_path, loss_function='mse')  # You can choose 'mae' here as well

    # Execute the model steps
    model.load_data()
    model.preprocess_data()
    model.train_model()
    model.evaluate_model()
    model.save_model()