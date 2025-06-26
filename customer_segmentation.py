import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set global styles
plt.style.use('ggplot')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)

class RFMAnalyzer:
    """Class to handle RFM analysis and customer segmentation"""
    def __init__(self, data_path):
        self.df = None
        self.rfm = None
        self.scaler = StandardScaler()
        self.load_data(data_path)
    
    def load_data(self, data_path):
        """Load and preprocess the dataset"""
        self.df = pd.read_excel(data_path)
        print(f"Original dataset shape: {self.df.shape}")
        
        # Data cleaning
        self.df = self.df.dropna(subset=['CustomerID'])
        self.df = self.df[(self.df['Quantity'] > 0) & (self.df['UnitPrice'] > 0)]
        self.df['CustomerID'] = self.df['CustomerID'].astype(int)
        self.df['TotalSpend'] = self.df['Quantity'] * self.df['UnitPrice']
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        print(f"Cleaned dataset shape: {self.df.shape}")
    
    def calculate_rfm(self):
        """Calculate RFM metrics"""
        reference_date = self.df['InvoiceDate'].max() + timedelta(days=1)
        
        self.rfm = self.df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',  # Frequency
            'TotalSpend': 'sum'  # Monetary
        }).reset_index()
        
        self.rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        return self.rfm
    
    def preprocess_data(self):
        """Preprocess data for clustering"""
        # Log transform to handle skewness
        self.rfm['Monetary'] = np.log1p(self.rfm['Monetary'])
        self.rfm['Frequency'] = np.log1p(self.rfm['Frequency'])
        
        # Standardize features
        self.rfm_scaled = self.scaler.fit_transform(
            self.rfm[['Recency', 'Frequency', 'Monetary']])
        return self.rfm_scaled
    
    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using multiple metrics"""
        metrics = {'K': [], 'Inertia': [], 'Silhouette': [], 
                  'Calinski_Harabasz': [], 'Davies_Bouldin': []}
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(self.rfm_scaled)
            
            metrics['K'].append(k)
            metrics['Inertia'].append(kmeans.inertia_)
            metrics['Silhouette'].append(silhouette_score(self.rfm_scaled, cluster_labels))
            metrics['Calinski_Harabasz'].append(calinski_harabasz_score(self.rfm_scaled, cluster_labels))
            metrics['Davies_Bouldin'].append(davies_bouldin_score(self.rfm_scaled, cluster_labels))
        
        return pd.DataFrame(metrics)
    
    def apply_clustering(self, n_clusters=4, method='kmeans'):
        """Apply clustering algorithm"""
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == 'dbscan':
            model = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError("Unsupported clustering method")
            
        self.rfm['Cluster'] = model.fit_predict(self.rfm_scaled)
        return self.rfm
    
    def plot_cluster_metrics(self, metrics_df):
        """Plot clustering evaluation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Elbow method
        axes[0,0].plot(metrics_df['K'], metrics_df['Inertia'], 'bo-')
        axes[0,0].set_title('Elbow Method')
        axes[0,0].set_xlabel('Number of clusters')
        axes[0,0].set_ylabel('Inertia')
        
        # Silhouette Score
        axes[0,1].plot(metrics_df['K'], metrics_df['Silhouette'], 'ro-')
        axes[0,1].set_title('Silhouette Score')
        axes[0,1].set_xlabel('Number of clusters')
        
        # Calinski-Harabasz Index
        axes[1,0].plot(metrics_df['K'], metrics_df['Calinski_Harabasz'], 'go-')
        axes[1,0].set_title('Calinski-Harabasz Index')
        axes[1,0].set_xlabel('Number of clusters')
        
        # Davies-Bouldin Index
        axes[1,1].plot(metrics_df['K'], metrics_df['Davies_Bouldin'], 'mo-')
        axes[1,1].set_title('Davies-Bouldin Index')
        axes[1,1].set_xlabel('Number of clusters')
        
        plt.tight_layout()
        plt.savefig('clustering_metrics.png')
        plt.close()
    
    def plot_clusters_3d(self):
        """Create 3D visualization of clusters"""
        fig = px.scatter_3d(
            self.rfm, 
            x='Recency', 
            y='Frequency', 
            z='Monetary',
            color='Cluster',
            title='3D Customer Segments',
            labels={'Recency': 'Recency (days)',
                   'Frequency': 'Frequency (log)',
                   'Monetary': 'Monetary (log)'},
            opacity=0.7,
            size_max=10
        )
        fig.write_html('3d_clusters.html')
    
    def analyze_clusters(self):
        """Analyze and summarize clusters"""
        # Calculate cluster statistics
        cluster_summary = self.rfm.groupby('Cluster').agg({
            'Recency': ['count', 'mean', 'std'],
            'Frequency': ['mean', 'std'],
            'Monetary': ['mean', 'std']
        }).round(2)
        
        # Save to CSV
        cluster_summary.to_csv('detailed_cluster_summary.csv')
        self.rfm.to_csv('customer_segments.csv', index=False)
        
        return cluster_summary

def main():
    # Initialize analyzer
    analyzer = RFMAnalyzer('Online Retail.xlsx')
    
    # Calculate RFM metrics
    rfm = analyzer.calculate_rfm()
    
    # Preprocess data
    rfm_scaled = analyzer.preprocess_data()
    
    # Find optimal number of clusters
    print("Finding optimal number of clusters...")
    metrics_df = analyzer.find_optimal_clusters()
    analyzer.plot_cluster_metrics(metrics_df)
    
    # Apply clustering
    n_clusters = 5  # Can be adjusted based on metrics
    print(f"\nApplying K-means clustering with {n_clusters} clusters...")
    analyzer.apply_clustering(n_clusters=n_clusters, method='kmeans')
    
    # Generate visualizations
    print("Generating visualizations...")
    analyzer.plot_clusters_3d()
    
    # Analyze and save results
    print("Analyzing clusters...")
    cluster_summary = analyzer.analyze_clusters()
    
    print("\nCluster Summary:")
    print(cluster_summary)
    print("\nAnalysis complete! Check the generated files for results.")

if __name__ == "__main__":
    main()