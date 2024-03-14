# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:34:20 2024

@author: USER
"""

import pandas as pd
import sklearn as sk
from sklearn.cluster import KMeans

# טען את הנתונים שלך
df = pd.read_csv("weather_all_data.csv")

# בחר את התכונות הרלוונטיות
features = ["temperature", "humidity", "pressure"]

# נורמל את הנתונים
scaler = sk.preprocessing.StandardScaler()
scaler.fit(df[features])
df[features] = scaler.transform(df[features])

# הגדר את מספר הקבוצות
n_clusters = 5

# צור מודל KMeans
kmeans = KMeans(n_clusters=n_clusters)

# התאם את המודל לנתונים
kmeans.fit(df[features])

# צור תווית אשכול לכל עיר
df["cluster"] = kmeans.labels_

# הדפס את תוויות האשכול
print(df["cluster"])

# מצא ערים באותו אשכול
for cluster_id in range(n_clusters):
    cluster_cities = df[df["cluster"] == cluster_id]["city"].tolist()
    print(f"Cluster {cluster_id + 1}: {cluster_cities}")
