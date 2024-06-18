# SP500 Stocks Analysis Application

Welcome to the SP500 Stocks Analysis application, built using Streamlit. This app allows you to analyze and explore fundamental and technical data of S&P 500 companies dynamically fetched from various sources.

## Features:
- Dynamic Data Fetching: Retrieves company data including market cap, price, ratios, and more from Yahoo Finance and Wikipedia.
- Data Visualization: Visualizes stock performance metrics such as daily, monthly, and yearly variations.
- Filtering Options: Filters stocks based on ticker symbols, industries, market cap, EPS forward, and calculated score.
- Interactive Charts: Displays top 5 stocks based on performance metrics with interactive bar charts.
- Scatter Plot: Provides a scatter plot to compare Market Cap and Score across industries.

### Installation:
To run the application locally:

Clone the repository.
Install required libraries:

```bash
pip install streamlit yfinance pandas numpy requests beautifulsoup4 sqlalchemy plotly
```

Run the application:

```bash
streamlit run app.py
```
### Data Sources:
- Yahoo Finance: Used for retrieving historical stock prices and financial metrics.
- Wikipedia: Used to fetch the list of S&P 500 companies for ticker symbols.
- Database; stores fetched and calculated data in a local SQLite database (stock_data.db).

Explore and analyze S&P 500 stocks with the SP500 Stocks Analysis app!
