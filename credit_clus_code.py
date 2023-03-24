#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load data
df = pd.read_csv('clustering_work.csv')

# Preprocess data
num_cols = ['currentRatio','quickRatio','cashRatio','daysOfSalesOutstanding','netProfitMargin','pretaxProfitMargin',
            'grossProfitMargin','operatingProfitMargin','returnOnAssets','returnOnCapitalEmployed','returnOnEquity',
            'assetTurnover','fixedAssetTurnover','debtEquityRatio','debtRatio','effectiveTaxRate',
            'freeCashFlowOperatingCashFlowRatio','freeCashFlowPerShare','cashPerShare','companyEquityMultiplier',
            'ebitPerRevenue','enterpriseValueMultiple','operatingCashFlowPerShare','operatingCashFlowSalesRatio',
            'payablesTurnover','PX1','PX2','PX3','PX4','PX5','PX6','PX7','PX8','PX9','PX10']
cat_cols = ['Rating','Name','Symbol','Rating Agency Name','Date','Sector']

# Scale numerical features
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df[num_cols])
scaled_df = pd.DataFrame(scaled_df, columns=num_cols)

# Concatenate numerical and categorical features
df = pd.concat([df[cat_cols], scaled_df], axis=1)

# Cluster companies using KMeans algorithm
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_df)

# Calculate silhouette score to evaluate clustering performance
silhouette_avg = silhouette_score(scaled_df, clusters)

# Add cluster labels to the original dataframe
df['Cluster'] = clusters

# Save clustered data to a new file
df.to_csv('clustered_credit_ratings.csv', index=False)


# In[2]:


# Import libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st

# Load data
df = pd.read_csv('clustered_credit_ratings.csv')

# Define function to predict credit rating
def predict_credit_rating(company_name, sector, date, rating_agency, current_ratio, quick_ratio,
                          cash_ratio, days_of_sales_outstanding, net_profit_margin, pretax_profit_margin,
                          gross_profit_margin, operating_profit_margin, return_on_assets, return_on_capital_employed,
                          return_on_equity, asset_turnover, fixed_asset_turnover, debt_equity_ratio, debt_ratio,
                          effective_tax_rate, free_cash_flow_operating_cash_flow_ratio, free_cash_flow_per_share,
                          cash_per_share, company_equity_multiplier, ebit_per_revenue, enterprise_value_multiple,
                          operating_cash_flow_per_share, operating_cash_flow_sales_ratio, payables_turnover,
                          px1, px2, px3, px4, px5, px6, px7, px8, px9, px10):
    # Load clustered data
    clustered_df = pd.read_csv('clustered_credit_ratings.csv')

    # Filter data by sector, date, and rating agency
    filtered_df = clustered_df[(clustered_df['Sector'] == sector) & (clustered_df['Date'] == date) & 
                               (clustered_df['Rating Agency Name'] == rating_agency)]

    # Create a new row for the input data
    input_data = pd.DataFrame({'Name': [company_name], 'Symbol': [''], 'Rating Agency Name': [rating_agency],
                               'Date': [date], 'Sector': [sector], 'Rating': [''], 'PX1': [px1], 'PX2': [px2],
                               'PX3': [px3], 'PX4': [px4], 'PX5': [px5], 'PX6': [px6], 'PX7': [px7], 'PX8': [px8],
                               'PX9': [px9], 'PX10': [px10], 
                               'Current Ratio': [current_ratio], 'Quick Ratio': [quick_ratio], 'Cash Ratio': [cash_ratio],
                               'Days of Sales Outstanding': [days_of_sales_outstanding],
                               'Net Profit Margin': [net_profit_margin], 'Pretax Profit Margin': [pretax_profit_margin],
                               'Gross Profit Margin': [gross_profit_margin],
                               'Operating Profit Margin': [operating_profit_margin],
                               'Return on Assets': [return_on_assets],
                               'Return on Capital Employed': [return_on_capital_employed],
                               'Return on Equity': [return_on_equity], 'Asset Turnover': [asset_turnover],
                                'Fixed Asset Turnover': [fixed_asset_turnover], 'Debt/Equity Ratio': [debt_equity_ratio], 'Debt Ratio': [debt_ratio],
                               'Effective Tax Rate': [effective_tax_rate],
                               'Free Cash Flow/Operating Cash Flow Ratio': [free_cash_flow_operating_cash_flow_ratio],
                               'Free Cash Flow per Share': [free_cash_flow_per_share], 'Cash per Share': [cash_per_share],
                               'Company Equity Multiplier': [company_equity_multiplier],
                               'EBIT per Revenue': [ebit_per_revenue],
                               'Enterprise Value Multiple': [enterprise_value_multiple],
                               'Operating Cash Flow per Share': [operating_cash_flow_per_share],
                               'Operating Cash Flow/Sales Ratio': [operating_cash_flow_sales_ratio],
                               'Payables Turnover': [payables_turnover]
                               })

    # Concatenate the filtered data and the input data
    concat_df = pd.concat([filtered_df, input_data], ignore_index=True)

    # Scale the data
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(concat_df.iloc[:, 6:])

    # Compute silhouette score
    silhouette_scores = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(scaled_df)
        silhouette_scores.append(silhouette_score(scaled_df, kmeans.labels_))

    # Select the number of clusters with the highest silhouette score
    n_clusters = np.argmax(silhouette_scores) + 2

    # Cluster the data
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(scaled_df)
    concat_df['Cluster'] = kmeans.labels_

    # Filter the data by cluster
    same_cluster_df = concat_df[concat_df['Name'] == company_name]

    # Compute average credit rating for same cluster
    average_rating = np.mean(same_cluster_df['Rating'])

    # Return predicted credit rating
    return average_rating


# In[ ]:




