import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class EDA:
    def __init__(self, base_dir, data_filename="merged_data.csv"):
        """
        Initializes the EDA object with paths for the merged data.
        
        :param base_dir: The base directory where the script is located.
        :param data_filename: The name of the merged data file.
        """
        self.base_dir = os.path.abspath(base_dir)
        self.cleaned_data_folder = os.path.join(self.base_dir, "../cleaned_data")  # Define cleaned data folder
        self.results_folder = os.path.join(self.base_dir, "../results")  # Define results folder
        self.data_file = os.path.join(self.cleaned_data_folder, data_filename)  # Path to merged_data.csv
        self.df = None
        
        # Create results folder if it doesn't exist
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
        
    def load_data(self):
        """ Load the cleaned and merged data using the defined path. """
        try:
            logger.info(f"Loading data from {self.data_file}...")
            self.df = pd.read_csv(self.data_file)
            logger.info("Data loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def check_missing_values(self):
        """ Check for missing values in the dataset. """
        logger.info("Checking for missing values in the dataset...")
        missing = self.df.isnull().sum()
        logger.info("Missing values check:")
        logger.info(missing[missing > 0])
        return missing
    
    def plot_distribution(self, column):
        """ Plot the distribution of a column and save the plot as PNG. """
        try:
            logger.info(f"Plotting distribution for column: {column}...")
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[column], kde=True, bins=30)
            plt.title(f"Distribution of {column}")
            plot_filename = os.path.join(self.results_folder, f"{column}_distribution.png")
            plt.savefig(plot_filename)
            plt.close()
            logger.info(f"Saved plot: {plot_filename}")
        except Exception as e:
            logger.error(f"Error plotting distribution for {column}: {e}")
    
    def compare_sales_before_during_after_holidays(self):
        """ Compare sales behavior before, during, and after holidays and save the plot. """
        logger.info("Comparing sales before, during, and after holidays...")
        if 'Sales' in self.df.columns and 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df['holiday_period'] = self.df['Date'].apply(self.assign_holiday_period)
            
            holiday_sales = self.df.groupby('holiday_period')['Sales'].mean().reset_index()
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='holiday_period', y='Sales', data=holiday_sales)
            plt.title("Average Sales Before, During, and After Holidays")
            plot_filename = os.path.join(self.results_folder, "holiday_sales_comparison.png")
            plt.savefig(plot_filename)
            plt.close()
            logger.info(f"Saved plot: {plot_filename}")
    
    def assign_holiday_period(self, date):
        """ Helper function to assign before, during, or after holidays. """
        holiday_dates = ["12-25", "01-01", "04-05"]  # Example of holidays (adjust as needed)
        holiday_str = date.strftime("%m-%d")
        
        if holiday_str in holiday_dates:
            return "During Holiday"
        elif date.month == 12 or date.month == 1:  # Example for before and after holiday
            return "Before Holiday"
        else:
            return "After Holiday"
    
    def correlation_analysis(self):
        """ Check for correlation between numeric variables and save the heatmap as PNG. """
        logger.info("Performing correlation analysis between numeric variables...")
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        correlation = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title("Correlation Matrix")
        plot_filename = os.path.join(self.results_folder, "correlation_matrix.png")
        plt.savefig(plot_filename)
        plt.close()
        logger.info(f"Saved plot: {plot_filename}")
    
    def plot_sales_vs_customers(self):
        """ Analyze the correlation between sales and number of customers and save the plot. """
        logger.info("Plotting relationship between Sales and Number of Customers...")
        if 'Sales' in self.df.columns and 'Customers' in self.df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='Customers', y='Sales', data=self.df)
            plt.title("Sales vs. Number of Customers")
            plt.xlabel("Number of Customers")
            plt.ylabel("Sales")
            plot_filename = os.path.join(self.results_folder, "sales_vs_customers.png")
            plt.savefig(plot_filename)
            plt.close()
            logger.info(f"Saved plot: {plot_filename}")
    
    def check_promo_impact_on_sales(self):
        """ Check how promos affect sales and save the plot. """
        logger.info("Analyzing promo impact on sales...")
        if 'Promo' in self.df.columns and 'Sales' in self.df.columns:
            promo_sales = self.df.groupby('Promo')['Sales'].mean().reset_index()
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Promo', y='Sales', data=promo_sales)
            plt.title("Average Sales with and without Promo")
            plot_filename = os.path.join(self.results_folder, "promo_impact_on_sales.png")
            plt.savefig(plot_filename)
            plt.close()
            logger.info(f"Saved plot: {plot_filename}")
    
    def explore_store_behavior(self):
        """ Explore customer behavior in stores and save the plot. """
        logger.info("Exploring customer behavior across stores...")
        if 'Store' in self.df.columns:
            store_sales = self.df.groupby('Store')['Sales'].mean().reset_index()
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Store', y='Sales', data=store_sales)
            plt.title("Average Sales by Store")
            plt.xticks(rotation=90)
            plot_filename = os.path.join(self.results_folder, "average_sales_by_store.png")
            plt.savefig(plot_filename)
            plt.close()
            logger.info(f"Saved plot: {plot_filename}")

    def analyze_competitor_impact(self):
        """ Analyze how competitor distance affects sales. """
        logger.info("Analyzing competitor distance effect on sales...")
        if 'CompetitionDistance' in self.df.columns:
            competitor_sales = self.df.groupby('CompetitionDistance')['Sales'].mean().reset_index()
            plt.figure(figsize=(10, 6))
            sns.lineplot(x='CompetitionDistance', y='Sales', data=competitor_sales)
            plt.title("Sales vs. Competitor Distance")
            plot_filename = os.path.join(self.results_folder, "competitor_distance_impact.png")
            plt.savefig(plot_filename)
            plt.close()
            logger.info(f"Saved plot: {plot_filename}")
    
    def perform_eda(self):
        """ Perform the entire exploratory data analysis and save plots. """
        logger.info("Starting Exploratory Data Analysis...")
        
        self.load_data()
        self.check_missing_values()
        self.plot_distribution('Sales')
        self.compare_sales_before_during_after_holidays()
        self.check_promo_impact_on_sales()
        self.analyze_competitor_impact()    
        self.correlation_analysis()
        self.plot_sales_vs_customers()
        self.explore_store_behavior()
        
        logger.info("Exploratory Data Analysis completed.")

# Run EDA
if __name__ == "__main__":
    # Specify the base directory where the script is located
    base_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize the EDA class
    eda = EDA(base_directory)
    
    # Perform EDA
    eda.perform_eda()