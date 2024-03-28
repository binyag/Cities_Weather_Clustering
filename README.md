# Cities_Weather_Clustering
 
This project aims to cluster cities worldwide based on their weather patterns using machine learning techniques. The motivation behind this study is to understand and group cities with similar weather characteristics, which can have various practical applications, such as assisting individuals in making informed decisions regarding relocation or agricultural planning.

## Dataset

The dataset used in this project is obtained from the Visual Crossing Weather API, which provides historical weather data for various locations worldwide. The dataset includes features such as temperature, dew point, relative humidity, wind speed, precipitation, snow depth, visibility, cloud cover, and sea level pressure for approximately 100 cities spanning over 250 days.

## Methodology

The primary methodology employed in this project is K-means clustering, a popular unsupervised learning algorithm used for grouping data points based on similarity. The following steps are involved:

1. **Data Preprocessing**: Missing values in the dataset are handled appropriately, wind direction is converted to vector representation, and date parsing and normalization are performed.
2. **Feature Engineering**: Principal Component Analysis (PCA) is applied to the preprocessed data to extract principal components representing the most significant variance, reducing the dimensionality of the dataset.
3. **Clustering**: K-means clustering is performed on both the original data and the PCA-transformed data to group cities based on their weather patterns.
4. **Evaluation**: The optimal number of clusters is determined using techniques like the elbow method and silhouette analysis. The clustering results obtained with and without PCA are compared using the Adjusted Rand Index (ARI) and visual representations.

## Results

The analysis reveals that clustering cities based on weather patterns can provide valuable insights for various applications. The optimal number of clusters was found to be 5 for both the PCA-transformed and the original data, indicating the robustness of the clustering approach.

The comparison between the clustering results with and without PCA showed a moderate level of agreement, with an ARI value of 0.4729, suggesting that while there are some discrepancies, there exists a substantial overlap between the two clustering approaches.

Visual representations of the clustering results, such as scatter plots of mean temperature and relative humidity, offer intuitive insights into the grouping of cities based on weather similarities.

## Future Work

Future work could involve exploring additional weather parameters, incorporating geographical locations, or considering temporal aspects such as seasonal variations or long-term trends. Furthermore, feature engineering and selection could be explored to potentially enhance clustering performance.

## Usage

To replicate the analysis and results, follow these steps:

1. Clone the repository: `git clone https://github.com/binyag/Cities_Weather_Clustering.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the scripts in the following order:
  - `API_CODE.py`: Retrieves historical weather data from the Visual Crossing Weather API and saves it as a CSV file.
  - `Clustering.py`: Performs data preprocessing, feature engineering, clustering, and evaluation.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

