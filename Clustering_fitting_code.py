# -*- coding: utf-8 -*-
"""
Created on Sat May 11 00:44:04 2024

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

# Number 1
# Load the data from CSV
data1 = pd.read_csv("gdp-per-capita-worldbank_2.csv")
clustering_data = data1[['GDP_per_capita', 'CO2_emissions_per_capita', 'Renewable_energy_percentage', 'Forest_coverage_percentage', 'Temperature_increase']]
scaler = StandardScaler()
normalized_data = scaler.fit_transform(clustering_data)
kmeans = KMeans(n_clusters=3, random_state=42)
data1['Cluster'] = kmeans.fit_predict(normalized_data)
plt.figure(figsize=(10, 6))
plt.scatter(data1['GDP_per_capita'], data1['CO2_emissions_per_capita'], c=data1['Cluster'], cmap='viridis', label='Data Points')
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100, label='Cluster Centers')
plt.xlabel('GDP per Capita')
plt.ylabel('CO2 Emissions per Capita')
plt.title('Clustering Analysis')
plt.legend()
plt.show()

# Number 2
data2 = pd.read_csv("gdp-per-capita-worldbank.csv")
years = data2['Year']
gdp_per_capita = data2['GDP per capita, PPP (constant 2017 international $)']
def model_function(x, a, b, c):
    return a * x**2 + b * x + c
popt, pcov = curve_fit(model_function, years, gdp_per_capita)
perr = np.sqrt(np.diag(pcov))
plt.scatter(years, gdp_per_capita, label='Data')
plt.plot(years, model_function(years, *popt), 'r-', label='Best Fit')
plt.fill_between(years, model_function(years, *popt) - 1.96 * perr[0],
                 model_function(years, *popt) + 1.96 * perr[0], color='gray', alpha=0.3)
plt.xlabel('Year')
plt.ylabel('GDP per capita (constant 2017 international $)')
plt.title('Fitting Analysis')
plt.legend()
plt.show()

# Number 3
data3 = pd.read_csv('Book1.csv')
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data3.drop('Country', axis=1))
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clusters, cmap='viridis', marker='o', edgecolors='k')
plt.title('Clustering of Countries')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()
for cluster_id in range(num_clusters):
    cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
    cluster_countries = data3.loc[cluster_indices, 'Country']
    print(f'Countries in Cluster {cluster_id + 1}: {", ".join(cluster_countries)}')
