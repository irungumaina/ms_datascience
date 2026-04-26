import time
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Filter for Kenya
kenya_df = df[df['country'] == 'Kenya'].copy()

# Features for clustering
features = ['household_size', 'age_of_respondent', 'location_type', 'cellphone_access', 'gender_of_respondent']

# Preprocessing
kenya_df['location_type'] = kenya_df['location_type'].map({'Rural': 0, 'Urban': 1})
kenya_df['cellphone_access'] = kenya_df['cellphone_access'].map({'No': 0, 'Yes': 1})
kenya_df['gender_of_respondent'] = kenya_df['gender_of_respondent'].map({'Female': 0, 'Male': 1})

X = kenya_df[features].dropna()

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sample for speed if dataset is large (it's 6k rows for Kenya usually, checking)
print(f"Kenya sample size: {len(X_scaled)}")

# K-Means
start_time = time.time()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_time = time.time() - start_time
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)

# DBSCAN
start_time = time.time()
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
dbscan_time = time.time() - start_time
# Silhouette score only works if more than one cluster is found (excluding noise)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
if n_clusters_dbscan > 1:
    dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
else:
    dbscan_silhouette = "N/A (too few clusters)"

print(f"K-Means Time: {kmeans_time:.4f}s, Silhouette: {kmeans_silhouette:.4f}")
print(f"DBSCAN Time: {dbscan_time:.4f}s, Silhouette: {dbscan_silhouette}, Clusters: {n_clusters_dbscan}")