import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataCleaner:
    def __init__(self, base_dir, input_filenames=["train.csv", "store.csv"], output_filename="cleaned_ml.csv"):
        """
        Initializes the DataCleaner object with paths for input and output files.
        
        :param base_dir: The base directory for relative paths.
        :param input_filenames: List of input CSV files to load.
        :param output_filename: The name of the output CSV file after cleaning.
        """
        self.base_dir = os.path.abspath(base_dir)
        self.data_folder = os.path.join(self.base_dir, "../data")
        self.cleaned_data_folder = os.path.join(self.base_dir, "../cleaned_data")  # Define cleaned data folder
        self.input_files = [os.path.join(self.data_folder, filename) for filename in input_filenames]
        self.output_file = os.path.join(self.cleaned_data_folder, output_filename)
        self.train_data = None
        self.store_data = None

    def load_data(self):
        """
        Loads the data from input CSV files into pandas DataFrames.
        
        :return: List of pandas DataFrames loaded from the files.
        """
        try:
            print(f"Loading data from {self.input_files[0]} and {self.input_files[1]}...")
            self.train_data = pd.read_csv(self.input_files[0], low_memory=False)
            self.store_data = pd.read_csv(self.input_files[1], low_memory=False)
            print("Data loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An error occurred while loading data: {e}")

    def assess_data_quality(self):
        """
        Performs a data quality assessment, including checking for missing values and empty columns.
        """
        if self.train_data is not None:
            print("\nTrain Data Quality Assessment: Checking for missing values...")
            self.check_missing_values(self.train_data, "Train Data")
        if self.store_data is not None:
            print("\nStore Data Quality Assessment: Checking for missing values...")
            self.check_missing_values(self.store_data, "Store Data")

    def check_missing_values(self, data, dataset_name):
        """
        Checks and prints the missing values for each column in a dataset.
        
        :param data: The dataset (pandas DataFrame) to check.
        :param dataset_name: The name of the dataset for identification in the output.
        """
        print(f"\nMissing values assessment for {dataset_name}:")
        missing_data = data.isnull().sum()
        missing_percentage = (missing_data / len(data)) * 100
        missing_summary = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percentage})
        missing_summary = missing_summary[missing_summary['Missing Values'] > 0]
        
        if not missing_summary.empty:
            print(f"\nColumns with Missing Data in {dataset_name}:")
            print(missing_summary)
        else:
            print(f"\nNo missing values detected in {dataset_name}.")

    def handle_missing_values(self):
        """
        Handles missing values in both the Train and Store datasets.
        """
        # Handle missing values in the Store dataset
        if self.store_data is not None:
            self.store_data = self.handle_store_missing_values(self.store_data)

    def handle_store_missing_values(self, store_data):
        """
        Handle missing values in the Store dataset according to specified rules.
        
        :param store_data: The Store dataset (pandas DataFrame).
        :return: Store dataset with missing values handled.
        """
        # Fill 'CompetitionDistance' with the median
        store_data['CompetitionDistance'] = store_data['CompetitionDistance'].fillna(store_data['CompetitionDistance'].median())

        # Fill 'PromoInterval' with the mode
        store_data['PromoInterval'] = store_data['PromoInterval'].fillna(store_data['PromoInterval'].mode()[0])

        # Fill 'CompetitionOpenSinceMonth' with forward fill
        store_data['CompetitionOpenSinceMonth'] = store_data['CompetitionOpenSinceMonth'].ffill()

        # Fill 'CompetitionOpenSinceYear' with backward fill
        store_data['CompetitionOpenSinceYear'] = store_data['CompetitionOpenSinceYear'].bfill()

        # Fill 'Promo2SinceWeek' with the mean value
        store_data['Promo2SinceWeek'] = store_data['Promo2SinceWeek'].fillna(store_data['Promo2SinceWeek'].mean())

        # Fill 'Promo2SinceYear' with the mean value
        store_data['Promo2SinceYear'] = store_data['Promo2SinceYear'].fillna(store_data['Promo2SinceYear'].mean())

        return store_data

    def clean_data(self):
        """
        Cleans the data by converting data types, filling missing values, and standardizing values.
        """
        if self.train_data is not None and self.store_data is not None:
            # Merge train_data and store_data on the 'Store' column (adjust as needed)
            self.merged_data = pd.merge(self.train_data, self.store_data, on="Store", how="left")
            
            # Handle specific columns for the merged dataset
            if 'Store' in self.merged_data.columns:
                self.merged_data['Store'] = pd.to_numeric(self.merged_data['Store'], errors='coerce')

            if 'DayOfWeek' in self.merged_data.columns:
                self.merged_data['DayOfWeek'] = pd.to_numeric(self.merged_data['DayOfWeek'], errors='coerce')

            if 'Date' in self.merged_data.columns:
                self.merged_data['Date'] = pd.to_datetime(self.merged_data['Date'], errors='coerce')

            if 'Sales' in self.merged_data.columns:
                self.merged_data['Sales'] = pd.to_numeric(self.merged_data['Sales'], errors='coerce')

            if 'Customers' in self.merged_data.columns:
                self.merged_data['Customers'] = pd.to_numeric(self.merged_data['Customers'], errors='coerce')

            # Convert binary columns (Open, Promo, SchoolHoliday)
            for col in ['Open', 'Promo', 'SchoolHoliday']:
                if col in self.merged_data.columns:
                    self.merged_data[col] = self.merged_data[col].astype(bool)

            # Adjust StateHoliday
            if 'StateHoliday' in self.merged_data.columns:
                self.merged_data['StateHoliday'] = self.merged_data['StateHoliday'].replace({'a': 0, 0: 0, 1: 1}).astype(bool)

            # Handle store dataset specific columns
            if 'StoreType' in self.merged_data.columns:
                self.merged_data['StoreType'] = self.merged_data['StoreType'].str.strip().str.lower()

            if 'Assortment' in self.merged_data.columns:
                self.merged_data['Assortment'] = self.merged_data['Assortment'].str.strip().str.lower()

            # Correct scaling part: columns to be scaled
            columns_to_scale = ['Sales', 'Customers', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']

            # Check if all columns to scale are in the dataset
            missing_columns = [col for col in columns_to_scale if col not in self.merged_data.columns]
            if missing_columns:
                print(f"Warning: The following columns are missing from the dataset and will not be scaled: {missing_columns}")

            # Apply scaling to the columns that exist
            existing_columns_to_scale = [col for col in columns_to_scale if col in self.merged_data.columns]

            if existing_columns_to_scale:
                scaler = StandardScaler()
                self.merged_data[existing_columns_to_scale] = scaler.fit_transform(self.merged_data[existing_columns_to_scale])
                print(f"Scaled columns: {existing_columns_to_scale}")

            # If needed, split the merged data back into separate datasets
            self.train_data = self.merged_data.drop(columns=self.store_data.columns, errors='ignore')  # Remove store columns from train data
            self.store_data = self.merged_data[self.store_data.columns]  # Retain store columns for store data

    def save_cleaned_data(self):
        """
        Saves the cleaned data to the output CSV file, including the merged data as 'primary_data.csv'.
        """
        if self.train_data is not None and self.store_data is not None:
            # Create the cleaned_data folder if it doesn't exist
            if not os.path.exists(self.cleaned_data_folder):
                os.makedirs(self.cleaned_data_folder)

            # Save the cleaned data as separate CSV files in the cleaned_data folder
            self.train_data.to_csv(os.path.join(self.cleaned_data_folder, "cleaned_train.csv"), index=False)
            self.store_data.to_csv(os.path.join(self.cleaned_data_folder, "cleaned_store.csv"), index=False)

            # Save the merged data (primary data) as 'primary_data.csv'
            self.merged_data.to_csv(os.path.join(self.cleaned_data_folder, "primary_data.csv"), index=False)
            
            print(f"Cleaned data saved to {self.cleaned_data_folder}!")

    def process(self):
        """
        Orchestrates the entire process of loading, assessing, cleaning, and saving the data.
        """
        self.load_data()
        self.assess_data_quality()
        self.handle_missing_values()
        self.clean_data()
        self.save_cleaned_data()

# Example usage of the DataCleaner class
if __name__ == "__main__":
    # Initialize the DataCleaner object
    cleaner = DataCleaner(base_dir=os.path.dirname(__file__))

    # Perform the data processing
    cleaner.process()