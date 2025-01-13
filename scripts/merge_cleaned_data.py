import os
import pandas as pd

class DataMerger:
    def __init__(self, base_dir, input_filenames=["cleaned_train.csv", "cleaned_store.csv"], output_filename="merged_data.csv"):
        """
        Initializes the DataMerger object with paths for input and output files.
        
        :param base_dir: The base directory for relative paths.
        :param input_filenames: List of input CSV files to load.
        :param output_filename: The name of the output CSV file after merging.
        """
        self.base_dir = os.path.abspath(base_dir)
        self.cleaned_data_folder = os.path.join(self.base_dir, "../cleaned_data")  # Define cleaned data folder
        self.input_files = [os.path.join(self.cleaned_data_folder, filename) for filename in input_filenames]
        self.output_file = os.path.join(self.cleaned_data_folder, output_filename)
        self.train_data = None
        self.store_data = None

        # Ensure the cleaned_data directory exists
        if not os.path.exists(self.cleaned_data_folder):
            os.makedirs(self.cleaned_data_folder)

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

    def merge_data(self):
        """
        Merges the loaded dataframes on the 'Store' column.
        
        :return: Merged pandas DataFrame.
        """
        if self.train_data is not None and self.store_data is not None:
            print("Merging data...")
            merged_data = pd.merge(self.train_data, self.store_data, on="Store", how="left")
            print("Data merged successfully!")
            return merged_data
        else:
            print("Error: Data not loaded properly.")
            return None

    def save_data(self, merged_data):
        """
        Saves the merged data to a CSV file.
        
        :param merged_data: The merged DataFrame to be saved.
        """
        if merged_data is not None:
            try:
                # Ensure the directory exists before saving the file
                os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
                merged_data.to_csv(self.output_file, index=False)
                print(f"Merged data saved to {self.output_file}")
            except Exception as e:
                print(f"An error occurred while saving the data: {e}")
        else:
            print("No data to save.")

if __name__ == "__main__":
    # Specify the base directory where the script is located
    base_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize the DataMerger class
    merger = DataMerger(base_directory)
    
    # Load the data
    merger.load_data()
    
    # Merge the data
    merged_data = merger.merge_data()
    
    # Save the merged data
    merger.save_data(merged_data)