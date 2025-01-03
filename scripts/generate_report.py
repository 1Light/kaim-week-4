import os
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class ReportGenerator:
    def __init__(self, base_dir, data_filename="merged_data.csv"):
        """
        Initializes the ReportGenerator object with paths for the merged data.
        
        :param base_dir: The base directory where the script is located.
        :param data_filename: The name of the merged data file.
        """
        self.base_dir = os.path.abspath(base_dir)
        self.cleaned_data_folder = os.path.join(self.base_dir, "../cleaned_data")  # Define cleaned data folder
        self.data_file = os.path.join(self.cleaned_data_folder, data_filename)  # Path to merged_data.csv
        self.df = None
    
    def load_data(self):
        """ Load the cleaned and merged data using the defined path. """
        try:
            logger.info(f"Loading data from {self.data_file}...")
            self.df = pd.read_csv(self.data_file)
            logger.info("Data loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def analyze_promo_distribution(self):
        """ Check for distribution in both training and test sets for promos. """
        if 'Promo' in self.df.columns and 'Set' in self.df.columns:  # Assuming 'Set' distinguishes train/test
            promo_distribution = self.df.groupby('Set')['Promo'].value_counts(normalize=True)
            logger.info("Promo distribution in train and test sets:")
            logger.info(promo_distribution)
    
    def check_seasonal_purchase_behavior(self):
        """ Find seasonal purchase behaviors like Christmas, Easter, etc. """
        self.df['Month'] = pd.to_datetime(self.df['Date']).dt.month
        seasonal_sales = self.df.groupby('Month')['Sales'].mean()
        logger.info("Seasonal sales behavior:")
        for month, avg_sales in seasonal_sales.items():
            logger.info(f"Month {month}: Average Sales = {avg_sales:.2f}")
    
    def analyze_promo_effect(self):
        """ Analyze promo impact on existing and new customers. """
        if 'Promo' in self.df.columns and 'Customers' in self.df.columns:
            promo_impact = self.df.groupby('Promo')[['Sales', 'Customers']].mean()
            logger.info("Promo impact on sales and customers:")
            logger.info(promo_impact)
    
    def recommend_promo_deployment(self):
        """ Recommend stores for better promo deployment. """
        if 'Store' in self.df.columns and 'Promo' in self.df.columns:
            promo_effectiveness = self.df.groupby(['Store', 'Promo'])['Sales'].mean().unstack()
            logger.info("Promo effectiveness by store:")
            logger.info(promo_effectiveness)
    
    def analyze_store_opening_behavior(self):
        """ Analyze trends in customer behavior during store opening/closing times. """
        if 'Store_Opening_Time' in self.df.columns and 'Store_Closing_Time' in self.df.columns:
            # Assuming these are string columns like "09:00" and "20:00"
            self.df['Opening_Hour'] = pd.to_datetime(self.df['Store_Opening_Time'], format='%H:%M').dt.hour
            self.df['Closing_Hour'] = pd.to_datetime(self.df['Store_Closing_Time'], format='%H:%M').dt.hour
            hourly_sales = self.df.groupby('Opening_Hour')['Sales'].mean()
            logger.info("Sales behavior by store opening hours:")
            logger.info(hourly_sales)
    
    def analyze_store_weekend_sales(self):
        """ Analyze sales of stores open on all weekdays. """
        if 'Store' in self.df.columns and 'Weekday_Open' in self.df.columns:  # Assuming this column exists
            weekend_sales = self.df.groupby('Store')['Weekday_Open', 'Sales'].mean()
            logger.info("Weekend sales for stores open all weekdays:")
            logger.info(weekend_sales)
    
    def analyze_assortment_impact(self):
        """ Check how assortment type affects sales. """
        if 'Assortment' in self.df.columns:
            assortment_sales = self.df.groupby('Assortment')['Sales'].mean()
            logger.info("Sales by assortment type:")
            logger.info(assortment_sales)
    
    def analyze_competitor_distance_effect(self):
        """ Analyze how competitor distance affects sales. """
        if 'CompetitionDistance' in self.df.columns:
            competitor_effect = self.df.groupby('CompetitionDistance')['Sales'].mean()
            logger.info("Sales by competitor distance:")
            logger.info(competitor_effect)
    
    def analyze_competitor_openings(self):
        """ Analyze impact of opening/reopening competitors. """
        if 'CompetitionOpenSinceYear' in self.df.columns and 'CompetitionDistance' in self.df.columns:
            self.df['CompetitionOpenYear'] = self.df['CompetitionOpenSinceYear'].fillna(0).astype(int)
            competitor_effect = self.df.groupby('CompetitionOpenYear')['Sales'].mean()
            logger.info("Sales by year of competitor opening:")
            logger.info(competitor_effect)

    def generate_report(self):
        """ Perform the entire exploratory data analysis and log insights. """
        logger.info("Starting Exploratory Data Analysis...")
        
        self.load_data()
        self.check_seasonal_purchase_behavior()
        self.analyze_promo_distribution()
        self.analyze_promo_effect()
        self.recommend_promo_deployment()
        self.analyze_store_opening_behavior()
        self.analyze_store_weekend_sales()
        self.analyze_assortment_impact()
        self.analyze_competitor_distance_effect()
        self.analyze_competitor_openings()
        
        logger.info("Exploratory Data Analysis completed.")

# Run ReportGenerator
if __name__ == "__main__":
    base_directory = os.path.dirname(os.path.abspath(__file__))
    report_gen = ReportGenerator(base_directory)
    report_gen.generate_report()