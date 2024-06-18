import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
import sqlite3
import os
import plotly.express as px

st.set_page_config(page_title="SP500 Shows", layout="wide") 
st.title("SP500 Stocks Analysis")

def get_symbols():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    tickers = [s.replace('\n', '') for s in tickers]
    return tickers

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def populate_with_info(data, stock_info):
    data['Symbol'].append(stock_info.get('symbol', np.nan))
    data['Name'].append(stock_info.get('longName', 'N/A'))
    data['Market Cap'].append(stock_info.get('marketCap', 0))
    data['Price'].append(stock_info.get('currentPrice', 0))
    data['PB'].append(stock_info.get('priceToBook', 0))
    data['EPS fwd'].append(stock_info.get('forwardEps', 0))
    data['PE fwd'].append(stock_info.get('forwardPE', 0))
    data['PEG'].append(stock_info.get('pegRatio', 0))
    data['D2E'].append(stock_info.get('debtToEquity', 0))
    data['ROE'].append(stock_info.get('returnOnEquity', 0))
    if 'freeCashflow' in stock_info and 'marketCap' in stock_info:
        fcfy = (stock_info['freeCashflow'] / stock_info['marketCap']) * 100
        data['FCFY'].append(round(fcfy, 2))
    else:
        data['FCFY'].append(0)
    data['CR'].append(stock_info.get('currentRatio', 0))
    data['QR'].append(stock_info.get('quickRatio',0))
    data['DY'].append(stock_info.get('dividendYield', 0.0) * 100)
    data['Beta'].append(stock_info.get('beta',0))
    data['Industry'].append(stock_info.get('industry', 'N/A'))

def save_to_database(data):
    if not os.path.exists('db'):
        os.makedirs('db')
    db = 'sqlite:///db/stock_data.db'
    engine = create_engine(db, echo=False)
    data.to_sql('fundamentals', con=engine, if_exists='replace', index=False)

def get_price_changes(symbols):
    data = {
        'Symbol': [],
        '1d%': [],
        '7d%': [],
        '30d%': [],
        '365d%': []
    }
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            if hist.empty:
                data['Symbol'].append(symbol)
                data['1d%'].append(0)
                data['7d%'].append(0)
                data['30d%'].append(0)
                data['365d%'].append(0)
                continue
            
            today = hist['Close'].iloc[-1]
            one_day_ago = hist['Close'].iloc[-2] if len(hist) > 1 else  np.nan
            seven_days_ago = hist['Close'].iloc[-6] if len(hist) > 6 else np.nan
            thirty_days_ago = hist['Close'].iloc[-22] if len(hist) > 22 else np.nan
            one_year_ago = hist['Close'].iloc[0] if len(hist) > 0 else np.nan

            data['Symbol'].append(symbol)
            data['1d%'].append(((today - one_day_ago) / one_day_ago * 100) if not np.isnan(one_day_ago) else 0)
            data['7d%'].append(((today - seven_days_ago) / seven_days_ago * 100) if not np.isnan(seven_days_ago) else 0)
            data['30d%'].append(((today - thirty_days_ago) / thirty_days_ago * 100) if not np.isnan(thirty_days_ago) else 0)
            data['365d%'].append(((today - one_year_ago) / one_year_ago * 100) if not np.isnan(one_year_ago) else 0)
            
        except Exception as e:
            
            print(f"Error retrieving price data for {symbol}: {e}")
            data['Symbol'].append(symbol)
            data['1d%'].append(0)
            data['7d%'].append(0)
            data['30d%'].append(0)
            data['365d%'].append(0)
    
    return pd.DataFrame(data)

def calculate_score(data):
    data['PE_fwd_norm'] = (1 / data['PE fwd'])*0.33  # PE fwd (Price to Earnings forward) Inverso porque PE bajo es mejor
    data['ROE_norm'] = (data['ROE'] / data['ROE'].max())*0.33 # ROE (Return on Equity)
    data['FCFY_norm'] = (data['FCFY'] / data['FCFY'].max())*0.33 # FCFY (Free Cash Flow Yield):
    data['Score'] = data['PE_fwd_norm'] + data['ROE_norm'] + data['FCFY_norm']
    data.drop(columns=['PE_fwd_norm', 'ROE_norm', 'FCFY_norm'], inplace=True)

    return data

def create_bar_chart(data, metric, title):
    top_5 = data.nlargest(5, metric)  # Obtener los 5 mejores
    top_5 = top_5.sort_values(by=metric, ascending=True)  # Invertir el orden de los datos
    fig = px.bar(top_5, y='Symbol', x=metric, orientation='h', title=title, text=metric, color_discrete_sequence=['#FF4B4B'])
    fig.update_traces(texttemplate='%{x:.2f}%', textposition='outside')  # Redondear a dos decimales y agregar %
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', height=300)  # Reducir la altura del gráfico
    return fig

data = {
    'Symbol': [],
    'Name': [],
    'Market Cap': [],
    'Industry': [],
    'EPS fwd': [],
    'PE fwd': [],
    'PEG': [],
    'PB': [],
    'ROE' : [],
    'FCFY' : [],
    'D2E' : [],
    'CR' : [],
    'QR' : [],
    'DY' : [],
    'Beta': [],
    'Price': []
    }

#symbols  = ['MSFT', 'AAPL', 'GOOG','AMZN']
symbols = get_symbols()[:10]

for symbol in symbols:
    try:
        ticker = yf.Ticker(symbol)     
        populate_with_info(data, ticker.info)
    except Exception as e:
        print(f"Error retrieving data for {symbol}: {e}")
        pass

df = pd.DataFrame(data)
df = calculate_score(df)

technical = get_price_changes(symbols)
fd = pd.merge(df, technical, on='Symbol', how='left')
industries = fd['Industry'].unique().tolist()

save_to_database(fd)

conn = sqlite3.connect('db/stock_data.db')
query = "SELECT * FROM fundamentals"
fundamentals = pd.read_sql_query(query, conn)
conn.close()

ticker_filter = st.sidebar.multiselect("Seleccionar Tickers", ['SP500'] + symbols, default=['SP500'])
industry_filter = st.sidebar.multiselect("Seleccionar Industrias", ['Industries'] + industries, default=['Industries'])
market_cap_filter = st.sidebar.slider("Seleccionar Market Cap", int(fundamentals['Market Cap'].min()), int(fundamentals['Market Cap'].max()), (int(fundamentals['Market Cap'].min()), int(fundamentals['Market Cap'].max())))
eps_fwd_filter = st.sidebar.slider("Seleccionar EPS fwd", float(fundamentals['EPS fwd'].min()), float(fundamentals['EPS fwd'].max()), (float(fundamentals['EPS fwd'].min()), float(fundamentals['EPS fwd'].max())))
score_filter = st.sidebar.slider("Seleccionar Score", float(fundamentals['Score'].min()), float(fundamentals['Score'].max()), (float(fundamentals['Score'].min()), float(fundamentals['Score'].max())))

filtered_data = fundamentals[
    (fundamentals['Symbol'].isin(ticker_filter) if 'SP500' not in ticker_filter else True) &
    (fundamentals['Industry'].isin(industry_filter) if 'Industries' not in industry_filter else True) &
    (fundamentals['Market Cap'] >= market_cap_filter[0]) &
    (fundamentals['Market Cap'] <= market_cap_filter[1]) &
    (fundamentals['EPS fwd'] >= eps_fwd_filter[0]) &
    (fundamentals['EPS fwd'] <= eps_fwd_filter[1]) &
    (fundamentals['Score'] >= score_filter[0]) &
    (fundamentals['Score'] <= score_filter[1]) 
]

charts = [
    create_bar_chart(fundamentals, '1d%', 'Top 5 Stocks Daily Variation'),
    create_bar_chart(fundamentals, '30d%', 'Top 5 Stocks by Monthly Variation'),
    create_bar_chart(fundamentals, '365d%', 'Top 5 Stocks by Yearly Variation'),
]

st.write("## 🔥 Trending")
cols = st.columns(3)
for i, chart in enumerate(charts):
    with cols[i % 3]:
        st.plotly_chart(chart, use_container_width=True)

st.dataframe(filtered_data)

########################################################

fig_scatter = px.scatter(
    filtered_data,
    x='Market Cap',
    y='Score',
    color="Industry",
    color_discrete_sequence=px.colors.qualitative.Set1,
    size='Market Cap',
    hover_name='Name',
    log_x=True,
    title='Market Cap vs Score by Industry'
)

fig_scatter.update_layout(xaxis_title='Market Cap (log scale)', yaxis_title='Score')

st.plotly_chart(fig_scatter)