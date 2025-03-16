The purpose of analyzing this dataset is to cluster bank customers into several groups in order to understand their behavior and apply appropriate business strategies to each cluster.

# Credit Card Dataset for Clustering

This project focuses on analyzing a dataset of bank customers to group them into clusters based on their behavior, using K-Means clustering. The goal is to understand customer behavior more effectively and apply targeted business strategies to each cluster.
## Problem Statement

As a Data Scientist, my task is to develop a customer segmentation model using K-Means clustering. The segmentation is based on the customers' credit card usage data over the last six months. This model will enable the bank to categorize customers into distinct groups based on their credit card usage patterns, allowing for more efficient management of customers by implementing suitable business strategies for each customer segment.
## Authors

This project was developed by [Dendi Apriyandi](https://www.linkedin.com/in/dendiapriyandi), an entry level data scientist and data engineer.
## Dataset

[![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

The dataset contains anonymized data about the credit card usage of bank customers over the last six months, with various attributes used for the clustering process. Key features include:

- Customer ID
- Credit Limit
- Balance
- Transactions
- Payment Behavior
- Other related features
## Methodology

The clustering is performed using the K-Means algorithm. Here are the key steps in the analysis:

- Data Cleaning: Handling missing or inconsistent data.
- Feature Scaling: Normalizing the features to ensure they contribute equally to the clustering algorithm.
- K-Means Clustering: Implementing the K-Means algorithm to segment the customers.
- Cluster Evaluation: Using metrics like inertia and silhouette score to evaluate the quality of the clusters.
- Insights & Visualization: Interpreting the clusters and visualizing the customer segmentation.
## Results

The customer segmentation model groups customers into distinct clusters based on their behavior, helping the bank to:

- Identify high-value customers who use credit cards frequently and make high-value transactions.
- Recognize customers with irregular payment behavior or high outstanding balances, allowing for targeted interventions.
- Tailor marketing and business strategies based on each segment's characteristics.
## Tools and Technologies

This project uses the following tools and technologies:

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib/Seaborn for visualizations
## Conclusion

This project showcases how clustering techniques like K-Means can be used to segment bank customers based on their credit card usage behavior, allowing businesses to design personalized and efficient strategies for customer management.
