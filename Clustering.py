# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:34:20 2024

@author: USER
"""
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
from sklearn.compose import ColumnTransformer


def try_parse_date(date_str):
    try:
        # Try format "%m/%d/%Y" (assuming most dates are in this format)
        return pd.to_datetime(date_str, format="%m/%d/%Y")
    except ValueError:
        try:
            # Try format "%Y-%m-%d"
            return pd.to_datetime(date_str, format="%Y-%m-%d")
        except ValueError:
            # Try format "%d %m %Y" (assuming encountering non-standard format)
            return pd.to_datetime(date_str, format="%m %d %Y")




# Load data from URL using requests (ensure data availability)

url = "https://github.com/binyag/Cities_Weather_Clustering/raw/main/data.csv"
data = pd.read_csv(url)
# Create a copy of the DataFrame for data manipulation (avoid modifying original data)
df = data.copy()
# Clean 'Address' column by removing leading/trailing single quotes
df['Address'] = df['Address'].str.strip("'")
df.info()
df["Conditions"].value_counts()
# Data Cleaning:
# Fill missing values in "Snow Depth" with 0
df["Snow Depth"].fillna(0, inplace=True)
# Fill missing values in "Visibility" with the median value of the column
df["Visibility"].fillna(df["Visibility"].median(), inplace=True)
# Fill missing values in "Cloud Cover" with 0
df["Cloud Cover"].fillna(0, inplace=True)

# Fill missing values in "Wind Direction" with the median value per address group

df['Wind Direction'].fillna(df.groupby('Address')['Wind Direction'].transform('median'), inplace=True)
# Feature Engineering:

# Convert "Date time" column to datetime format (assuming valid format)
df["Date time"] = df["Date time"].apply(try_parse_date)
# Drop the "Conditions" column as it might not be relevant for further analysis
df.drop(columns=['Conditions'], inplace=True)

# Define features for normalization (numerical columns)

features_N = [ 'Minimum Temperature', 'Maximum Temperature','Temperature','Dew Point', 'Relative Humidity','Wind Speed','Precipitation', 'Precipitation Cover','Snow Depth', 'Visibility', 'Cloud Cover', 'Sea Level Pressure']

# Create a scaler for normalizing numerical features
scaler = sk.preprocessing.StandardScaler()
# Create a column transformer to scale only the specified features
column_transformer = ColumnTransformer([('num', scaler, features_N)], remainder='drop')
# Create a pipeline to combine preprocessing steps
pipeline = Pipeline([('preprocess', column_transformer),])
# Fit the pipeline to the data (learn normalization parameters)
pipeline.fit(df)

norm_data = pipeline.fit_transform(df)

n_pca_data_a = pd.concat([pd.DataFrame(norm_data).iloc[:,:1], df["Date time"], df["Address"]], axis=1)
# Create a pivot table from the combined data, with "Address" as rows and "Date time" as columns
pivot_data_not_pca = pd.pivot(n_pca_data_a, index = 'Address',columns = "Date time" )

# Perform Principal Component Analysis (PCA) for dimensionality reduction

# Define a PCA object with 10 principal components and a random state for reproducibility
pca = PCA(n_components=10 ,random_state=26)  
# Fit the PCA model to the normalized data to learn the components
pca.fit(norm_data)
# Transform the normalized data using the fitted PCA model
pca_data = pca.transform(norm_data)
# Print the principal components (weightings for transformed features)
print(pca.components_)
# Print the total explained variance ratio (proportion of variance explained by the selected components)
print(np.sum(pca.explained_variance_ratio_))

sorted_pca_data = pca_data[df.index.values]
pca_data_df = pd.DataFrame(pca_data)  # Create DataFrame from pca_data

pca_data_a = pd.concat([pca_data_df, df["Date time"], df["Address"]], axis=1)

pivot_data_pca = pd.pivot(pca_data_a, index = 'Address',columns = "Date time" )
A = pd.DataFrame(pivot_data_pca.index)



# Define the range of n_clusters values to evaluate
n_clusters_range = np.arange(2, 15)
kmeans = KMeans(n_clusters=1, random_state=26, init='k-means++')
kmeans.fit(pivot_data_not_pca.dropna(axis=1, how='any'))

inertia_o_n_pca = kmeans.inertia_
alpha_k = 0.04
# Dictionaries to store results
silhouette_scores = {}
inertia_scores = {}

# Perform k-means clustering with k-means++ initialization for each n_clusters
for n_clusters in n_clusters_range:
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=26, init='k-means++')
    kmeans.fit(pivot_data_not_pca.dropna(axis=1, how='any'))

  # Calculate Silhouette score
    silhouette_scores[n_clusters] = silhouette_score(pivot_data_not_pca.dropna(axis=1, how='any'), kmeans.labels_)

  # Calculate inertia (within-cluster sum of squares)
    inertia_scores[n_clusters] = kmeans.inertia_ / inertia_o_n_pca + alpha_k * n_clusters



# Plot Silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, silhouette_scores.values(), label='Silhouette Score')
plt.xlabel('n_clusters')
plt.ylabel('Silhouette Score')
plt.title("Silhouette Score without PCA")

plt.legend()
plt.show()

# Plot Scaled Inertia
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, inertia_scores.values(), label='Scaled Inertia')
plt.xlabel('n_clusters')
plt.ylabel('Scaled Inertia')
plt.title("Scaled Inertia without PCA")

plt.legend()
plt.show()












kmeans = KMeans(n_clusters=1, random_state=26, init='k-means++')
kmeans.fit(pivot_data_pca.dropna(axis=1, how='any'))

inertia_o_pca = kmeans.inertia_
# Dictionaries to store results
silhouette_scores = {}
inertia_scores = {}

# Perform k-means clustering with k-means++ initialization for each n_clusters
for n_clusters in n_clusters_range:
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=26, init='k-means++')
    kmeans.fit(pivot_data_pca.dropna(axis=1, how='any'))

  # Calculate Silhouette score
    silhouette_scores[n_clusters] = silhouette_score(pivot_data_pca.dropna(axis=1, how='any'), kmeans.labels_)

  # Calculate inertia (within-cluster sum of squares)
    inertia_scores[n_clusters] = kmeans.inertia_ / inertia_o_pca + alpha_k * n_clusters



# Plot Silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, silhouette_scores.values(), label='Silhouette Score')
plt.xlabel('n_clusters')
plt.ylabel('Silhouette Score')
plt.title("Silhouette Score with PCA")

plt.legend()
plt.show()

# Plot Scaled Inertia
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, inertia_scores.values(), label='Scaled Inertia')
plt.xlabel('n_clusters')
plt.ylabel('Scaled Inertia')
plt.title("Scaled Inertia with PCA")

plt.legend()
plt.show()



kmeans = KMeans(n_clusters=5, random_state=26, init='k-means++')
kmeans.fit(pivot_data_not_pca.dropna(axis=1, how='any'))
A ["clus_not_pca"] = kmeans.labels_

kmeans = KMeans(n_clusters=5, random_state=26, init='k-means++')
kmeans.fit(pivot_data_pca.dropna(axis=1, how='any'))
A ["clus_pca"] = kmeans.labels_



# Step 1: Group DataFrame df by 'Address' column and calculate mean temperature and humidity
city_means = df.groupby('Address').agg({'Temperature': 'mean',
                                        'Relative Humidity': 'mean',
                                        "Wind Speed": "mean" ,
                                        "Visibility":"mean" }).reset_index()

# Step 2: Merge cluster labels from DataFrame A with calculated mean temperature and humidity
merged_df = city_means.merge(A, left_on='Address', right_on='Address')

# Step 3: Plot mean temperature and humidity for each cluster
plt.figure(figsize=(12, 8))

# Iterate over unique cluster labels
for cluster_label in merged_df['clus_not_pca'].unique():
    cluster_data = merged_df[merged_df['clus_not_pca'] == cluster_label]
    plt.scatter(cluster_data['Temperature'], cluster_data['Relative Humidity'], label=f'Cluster {cluster_label}')

plt.xlabel('Mean Temperature')
plt.ylabel('Mean Relative Humidity')
plt.title('Clustering of Cities based on Mean Temperature and Humidity')
plt.legend()
plt.grid(True)
plt.show()






# Assuming clus_pca and clus_not_pca are the cluster labels obtained from PCA and non-PCA data
ari = adjusted_rand_score(A['clus_pca'], A['clus_not_pca'])

print(f"Adjusted Rand Index (ARI) between clusterings with PCA and without PCA: {ari}")


cross_tab = pd.crosstab(A['clus_pca'], A['clus_not_pca'])
print("Cross-tabulation table between clusterings with PCA and without PCA:")
print(cross_tab)


palette = sns.color_palette('Set2', n_colors=len(merged_df['clus_not_pca'].unique()))  # Create color palette

plt.figure(figsize=(15, 9))

# Plot clusters obtained without PCA (using a loop for individual color assignment)
for i, cluster_label in enumerate(merged_df['clus_not_pca'].unique()):
    cluster_data = merged_df[merged_df['clus_not_pca'] == cluster_label]
    color = palette[i % len(palette)]  # Access color from palette cyclically

    plt.scatter(cluster_data['Temperature'], cluster_data['Relative Humidity'],
                label=f'Non-PCA Cluster {cluster_label}',
                marker='s',
                edgecolors=color,
                linewidth=2.5, 
                facecolors='none',
                s=150)

# Plot clusters obtained with PCA (optional, adjust marker style)
for cluster_label in merged_df['clus_pca'].unique():
    cluster_data = merged_df[merged_df['clus_pca'] == cluster_label]
    plt.scatter(cluster_data['Temperature'], cluster_data['Relative Humidity'],
                label=f'PCA Cluster {cluster_label}',
                marker='o', # Use 'o' for circles (or choose another style)
                s=50)  # Adjust marker size (optional)

plt.xlabel('Mean Temperature')
plt.ylabel('Mean Relative Humidity')
plt.title('Comparison of Clustering Results with and without PCA')
plt.legend()
plt.grid(True)
plt.show()










plt.figure(figsize=(15, 9))

# Plot clusters obtained without PCA (using a loop for individual color assignment)
for i, cluster_label in enumerate(merged_df['clus_not_pca'].unique()):
    cluster_data = merged_df[merged_df['clus_not_pca'] == cluster_label]
    color = palette[i % len(palette)]  # Access color from palette cyclically

    plt.scatter(cluster_data["Visibility"], cluster_data["Wind Speed"],
                label=f'Non-PCA Cluster {cluster_label}',
                marker='s',
                edgecolors=color,
                linewidth=2.5, 
                facecolors='none',
                s=150)

# Plot clusters obtained with PCA (optional, adjust marker style)
for cluster_label in merged_df['clus_pca'].unique():
    cluster_data = merged_df[merged_df['clus_pca'] == cluster_label]
    plt.scatter(cluster_data["Visibility"], cluster_data["Wind Speed"],
                label=f'PCA Cluster {cluster_label}',
                marker='o', # Use 'o' for circles (or choose another style)
                s=50)  # Adjust marker size (optional)

plt.xlabel('Mean Visibility')
plt.ylabel('Mean Wind Speed')
plt.title('Comparison of Clustering Results with and without PCA')
plt.legend()
plt.grid(True)
plt.show()
