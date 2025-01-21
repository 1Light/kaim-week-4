# Rossmann Pharmaceuticals Sales Forecasting

## Project Overview
It project aims to build an end-to-end machine learning solution for forecasting sales at Rossmann Pharmaceuticals stores across various cities. The finance team requires a tool to predict sales six weeks ahead, enabling better planning and resource allocation. The solution will analyze various factors such as promotions, competition, holidays, and seasonality to provide accurate forecasts.

## Business Need
The current sales forecasting relies on managers' experience and judgment. To enhance accuracy and consistency, a data-driven approach is essential. This project delivers a prediction system that helps analysts and managers make informed decisions.

## Objectives
- **Exploratory Data Analysis (EDA):**
  - Understand customer purchasing behavior.
  - Analyze the impact of promotions, holidays, and store attributes on sales.
- **Machine Learning Model:**
  - Build predictive models to forecast sales.
- **Deep Learning Approach:**
  - Explore neural network-based solutions for enhanced accuracy.
- **Web Interface:**
  - Serve predictions via a web application for easy access by analysts.

## Key Features
- **Store:** Unique identifier for each store.
- **Sales:** Turnover for a given day (target variable).
- **Customers:** Number of customers visiting the store.
- **Promo:** Indicates promotional activities.
- **StateHoliday & SchoolHoliday:** Information on holidays affecting sales.
- **CompetitionDistance:** Distance to the nearest competitor.

## Folder Structure
```
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   └── __init__.py
├── notebooks/
│   ├── __init__.py
│   └── README.md
├── tests/
│   └── __init__.py
└── scripts/
    └── __init__.py
```

## Getting Started
1. **Clone the repository:**
   ```bash
   git clone https://github.com/1Light/kaim-week-4.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd kaim-week-4
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Start exploring the data:**
   Run the notebooks in the `notebooks/` directory.

## Deliverables
- **Exploratory Data Analysis:** Insights from customer behavior and sales patterns.
- **Machine Learning Models:** Accurate sales forecasting system.
- **Web Application:** User-friendly interface for serving predictions.

## Tools and Technologies
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Flask  
- **Visualization:** Matplotlib, Seaborn  
- **Web Framework:** Flask (or alternative if required)  
- **Deployment:** Docker (optional for containerization)

## Contributors
- **Nasir A. Degu:** Machine Learning Engineer

## License
This project is licensed under the MIT License.

## Acknowledgments
- **Kaggle** for providing the dataset.  
- **Rossmann Pharmaceuticals** for the business challenge.