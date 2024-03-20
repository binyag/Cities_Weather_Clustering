# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:34:20 2024

@author: USER
"""
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.cluster import KMeans ,DBSCAN
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  OneHotEncoder
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax  # Consider normalization
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


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
            return pd.to_datetime(date_str, format="%d %m %Y")




url = "https://github.com/binyag/Cities_Weather_Clustering/raw/main/data.csv"
# טען את הנתונים שלך
data = pd.read_csv(url)
df = data.copy()
df['Address'] = df['Address'].str.strip("'")
df.info()
df["Conditions"].value_counts()
# מילוי ערכים חסרים
df["Snow Depth"].fillna(0, inplace=True)
df["Visibility"].fillna(df["Visibility"].median(), inplace=True)
df["Cloud Cover"].fillna(0, inplace=True)

df['Wind Direction'].fillna(df.groupby('Address')['Wind Direction'].transform('median'), inplace=True)

# החלת הפונקציה על עמודה "תאריך"
df["Date time"] = df["Date time"].apply(try_parse_date)
#df = df.set_index('Date time')


df["Rain"] = np.where(df["Conditions"].str.contains("Rain"), 1, 0)
df["Partially cloudy"] = np.where(df["Conditions"].str.contains("Partially cloudy"), 1, 0)
df["Overcast"] = np.where(df["Conditions"].str.contains("Overcast"), 1, 0)
df.drop(columns="Conditions", inplace=True)

# בחר את התכונות הרלוונטיות
features_N = [ 'Minimum Temperature', 'Maximum Temperature','Temperature','Dew Point', 'Relative Humidity','Wind Direction','Wind Speed','Precipitation', 'Precipitation Cover','Snow Depth', 'Visibility', 'Cloud Cover', 'Sea Level Pressure']


scaler = sk.preprocessing.StandardScaler()

column_transformer = sk.compose.ColumnTransformer([
    ('num', scaler, features_N),
    ('cat', 'passthrough', ['Overcast',"Partially cloudy","Rain"])  # Add 'Overcast' with 'passthrough' transformer
], remainder='drop')
pipeline = Pipeline([
    ('preprocess', column_transformer),
])
pipeline.fit(df)

norm_data = pipeline.fit_transform(df)






# 1. צור PCA

pca = PCA(n_components=12 ,random_state=18)  # 2 רכיבים ראשיים

# 2. התאם את ה-PCA לנתונים

pca.fit(norm_data)

# 3. המר את הנתונים

pca_data = pca.transform(norm_data)

# 4. הדפס רכיבים ראשיים

print(pca.components_)

# 5. הדפס וריאנס מוסבר

print(pca.explained_variance_ratio_)
sorted_pca_data = pca_data[df.index.values]
pca_data_df = pd.DataFrame(pca_data)  # Create DataFrame from pca_data

pca_data_a = pd.concat([pca_data_df, df["Date time"], df["Address"]], axis=1)



















"""


# הגדר את מספר הקבוצות
n_clusters = 5

# צור מודל KMeans
kmeans = KMeans(n_clusters=n_clusters)

# התאם את המודל לנתונים
kmeans.fit(norm_data)

# צור תווית אשכול לכל עיר
df["cluster"] = kmeans.labels_

# הדפס את תוויות האשכול
print(df["cluster"])

# מצא ערים באותו אשכול
for cluster_id in range(n_clusters):
    cluster_cities = df[df["cluster"] == cluster_id]["Address"].tolist()
    print(f"Cluster {cluster_id + 1}: {cluster_cities}")







# Select all numeric columns except 'Conditions' (categorical)
# Consider feature selection or engineering based on domain knowledge
numeric_columns = df.select_dtypes(include=[np.number]).columns[:-1]
X = p.values

# Preprocess the data (optional but recommended)
scaler = TimeSeriesScalerMinMax()  # Consider min-max or standardization
X = scaler.fit_transform(X)

# Define the KMEANS model parameters
n_clusters = 5  # Adjust based on your data and desired number of clusters
metric = "dtw"  # Dynamic Time Warping for time series similarity
max_iter = 50  # Maximum iterations for convergence
random_state = 42  # Set a random state for reproducibility

# Create and fit the KMEANS model
model = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, max_iter=max_iter, random_state=random_state)
model.fit(X)

# Get the cluster labels for each time series
cluster_labels = model.labels_

# Optional: Add cluster labels as a new column in the DataFrame
A['cluster'] = gmm.predict(p)

# Analyze and interpret the clusters
# Use domain knowledge and data exploration techniques
# (e.g., visualizing clusters, calculating cluster centroids)

print("Cluster labels:", cluster_labels)  # Example output
"""


p = pd.pivot(pca_data_a, index = 'Address',columns = "Date time" )
A = pd.DataFrame(p.index)



gmm = GaussianMixture(n_components=4)

# אימון המודל על הדאטה
gmm.fit(p)

# הדפסת ממוצעי הרכיבים
print("ממוצעים:", gmm.means_)

# הדפסת סטיית התקן של הרכיבים
print("סטיית תקן:", gmm.covariances_)

# הדפסת הסתברויות ההשתייכות לכל נקודה
for i in range(len(A)):
    print(A[i], gmm.predict(p)[i])
A = A.assign(cluster_GMM4=pd.Series(gmm.predict(p)))

type(pd.Series(gmm.predict(p)))
