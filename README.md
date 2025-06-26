# Customer Segmentation using RFM Analysis

This project implements customer segmentation using RFM (Recency, Frequency, Monetary) analysis with machine learning clustering techniques.

## Features

- Data loading and preprocessing
- RFM metric calculation
- Multiple clustering algorithms (K-means, Hierarchical, DBSCAN)
- Cluster evaluation using multiple metrics
- Interactive 3D visualizations
- Detailed cluster analysis

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd customer-segmentation
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your retail data file (Excel format) in the project directory
2. Update the filename in `customer_segmentation.py` if different from 'Online Retail.xlsx'
3. Run the script:
   ```bash
   python customer_segmentation.py
   ```

## Output

The script generates several output files:
- `clustering_metrics.png`: Evaluation metrics for different cluster counts
- `3d_clusters.html`: Interactive 3D visualization of customer segments
- `detailed_cluster_summary.csv`: Comprehensive statistics for each cluster
- `customer_segments.csv`: Complete dataset with cluster assignments

## License

MIT
