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
import matplotlib.pyplot as plt


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


data.columns


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
x_w = df['Wind Direction'].apply(lambda x: np.cos(x * np.pi / 180))
y_w = df['Wind Direction'].apply(lambda x: np.sin(x * np.pi / 180))
df['vector_Wind_Direction'] = list(zip(x_w, y_w))
# החלת הפונקציה על עמודה "תאריך"
df["Date time"] = df["Date time"].apply(try_parse_date)
#df = df.set_index('Date time')


#df["Rain"] = np.where(df["Conditions"].str.contains("Rain"), 1, 0)
#df["Partially cloudy"] = np.where(df["Conditions"].str.contains("Partially cloudy"), 1, 0)
#df["Overcast"] = np.where(df["Conditions"].str.contains("Overcast"), 1, 0)
df.drop(columns="Conditions", inplace=True)


# בחר את התכונות הרלוונטיות
features_N = [ 'Minimum Temperature', 'Maximum Temperature','Temperature','Dew Point', 'Relative Humidity','Wind Speed','Precipitation', 'Precipitation Cover','Snow Depth', 'Visibility', 'Cloud Cover', 'Sea Level Pressure']


scaler = sk.preprocessing.StandardScaler()

column_transformer = sk.compose.ColumnTransformer([
    ('num', scaler, features_N)#,
    #('cat', 'passthrough', ['Overcast',"Partially cloudy","Rain"])  # Add 'Overcast' with 'passthrough' transformer
], remainder='drop')
pipeline = Pipeline([
    ('preprocess', column_transformer),
])
pipeline.fit(df)

norm_data = pipeline.fit_transform(df)


n_pca_data_a = pd.concat([pd.DataFrame(norm_data).iloc[:,:1], df["Date time"], df["Address"]], axis=1)

p_n_pca = pd.pivot(n_pca_data_a, index = 'Address',columns = "Date time" )


# הגדרת צבעים אוטומטיים
colors = plt.cm.tab20(np.arange(len(df['Address'].unique())))

# לולאה לכל עמודה
for col in df.columns:
    # סינון נתונים לא זמינים
    df_filtered = df

    # יצירת גרף
    plt.figure(figsize=(10, 6))
    for i, address in enumerate(df_filtered['Address'].unique()):
        df_address = df_filtered.loc[df_filtered['Address'] == address]
        plt.plot(df_address['Date time'], df_address[col], color=colors[i], label=address)
    
    # הגדרות גרף
    plt.title(col)
    plt.xlabel('תאריך')
    plt.ylabel(col)
    plt.show()



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
gmm.fit(p.dropna(axis=1, how='any'))

# הדפסת ממוצעי הרכיבים
print("ממוצעים:", gmm.means_)

# הדפסת סטיית התקן של הרכיבים
print("סטיית תקן:", gmm.covariances_)

# הדפסת הסתברויות ההשתייכות לכל נקודה
for i in range(len(A)):
    print(A[i], gmm.predict(p)[i])
A = A.assign(cluster_GMM4=pd.Series(gmm.predict(p)))

type(pd.Series(gmm.predict(p)))
df_with_missing_rows = p[p.isnull().any(axis=1)]
cols_with_nan = p.columns[p.isnull().sum(axis=0) > 0]
