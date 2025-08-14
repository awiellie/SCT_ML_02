# 🛍️ Customer Segmentation using K-Means

This project is part of Task 02 of the SkillCraft Technology internship program. It implements a K-Means clustering model to group mall customers based on:
- Annual income (Annual Income (k$))
- Spending behavior (Spending Score (1–100))

The pipeline scales features, evaluates k = 2–10 with Elbow and Silhouette, and outputs cluster labels, a cluster profile (means & counts), and visualization plots (elbow, silhouette, scatter with centroids).

## 📌 Technologies Used

- Python
- Pandas
- Scikit-learn
- Numpy
- Matplotlib

## 📦 Installation
To run this project locally, follow the steps below:
1. **Clone the repository**:
```
git clone https://github.com/Agent-A345/SCT_ML_02.git
```
2. **Install Dependencies**
```
pip install pandas numpy scikit-learn matplotlib
```
3. **Run the program**
```
python task2.py
```

## 🧠 How It Works

1. Load the dataset (Mall_Customers.csv).
2. Select features: Annual Income (k$) and Spending Score (1–100) → drop NaNs and standardize.
3. Search k (2–10) using Elbow (inertia) and Silhouette, then pick the best k (≈ 5 in this run).
4. Fit K-Means with the chosen k and assign a Cluster label to every customer.
5. Export outputs: mall_customers_with_clusters.csv, cluster_profile.csv, and plots — elbow_plot.png, silhouette_scores.png, clusters_scatter.png.
6. Interpret segments using the cluster profile (means & counts) for marketing actions.

## 🙌 Acknowledgements
Thanks to SkillCraft Technology for the opportunity to work on this internship project.

## License
This project is licensed under the MIT License.
