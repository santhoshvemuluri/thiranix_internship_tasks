import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Generate Mock Customer Dataset
# ==========================================
np.random.seed(42)
# Creating 200 mock customers with Age, Income, and Spending Score
data = {
    'CustomerID': range(1, 201),
    'Age': np.random.randint(18, 70, 200),
    'AnnualIncome_k$': np.random.randint(15, 150, 200),
    'SpendingScore_1_100': np.random.randint(1, 100, 200)
}
df = pd.DataFrame(data)

print("Dataset Head:")
print(df.head(), "\n")

# ==========================================
# 2. Data Preprocessing
# ==========================================
# Selecting Income and Spending Score for 2D clustering and visualization
X = df[['AnnualIncome_k$', 'SpendingScore_1_100']]

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. Determine Optimal Clusters (Elbow Method)
# ==========================================
wcss = [] # Within-Cluster Sum of Square
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method graph (Optional - comment out if you only want the final plot)
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
# plt.title('Elbow Method For Optimal k')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS')
# plt.show()

# ==========================================
# 4. Apply K-Means Clustering
# ==========================================
# Based on standard datasets like this, 5 is usually a good number of clusters
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add the cluster labels back to the original dataframe
df['Cluster'] = cluster_labels

# ==========================================
# 5. Analyze Purchase Patterns & Preferences
# ==========================================
print("Cluster Analysis (Average Values per Segment):")
cluster_summary = df.groupby('Cluster')[['Age', 'AnnualIncome_k$', 'SpendingScore_1_100']].mean()
print(cluster_summary, "\n")

# ==========================================
# 6. Visualize Segments
# ==========================================
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='AnnualIncome_k$', 
    y='SpendingScore_1_100', 
    hue='Cluster', 
    data=df, 
    palette='viridis', 
    s=100, 
    alpha=0.8
)

# Plotting the centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')

plt.title('Customer Segments based on Income and Spending Behavior')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Save the plot as an image file and show it
plt.savefig('customer_segments.png')
print("Visualization saved as 'customer_segments.png'.")
plt.show()
