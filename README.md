# Mall Customers Clustering with K-Means

## Customer Segmentation Analysis using Machine Learning

K-Means clustering implementation for mall customer segmentation using customer demographics, income, and spending behavior.

This project groups customers into meaningful segments to help understand different spending patterns and support data‑driven marketing strategies.

---

## Project Overview

The goal is to:

- Explore mall customer data (demographics, income, spending score).
- Apply **K-Means clustering** to segment customers into groups.
- Visualize clusters to interpret customer behavior.
- Derive business insights for targeted marketing and customer relationship management.

---

## Dataset

- **Number of records:** 200 customers  
- **Typical columns (features):**
  - `CustomerID` – unique identifier
  - `Gender`
  - `Age`
  - `Annual Income (k$)`
  - `Spending Score (1–100)`

> Note: Column names may vary slightly depending on the dataset version, but the core information remains the same.

---

## Features Used for Clustering

The clustering focuses on:

- **Demographics**
  - Gender
  - Age
- **Socioeconomic**
  - Annual Income
- **Behavioral**
  - Spending Score (assigned by the mall based on purchasing behavior)

Different combinations of these features may be used to build and compare clustering results, for example:

- `Annual Income` vs. `Spending Score`
- `Age` vs. `Spending Score`
- Multi-dimensional clustering with all numeric features.

---

## Methods and Workflow

1. **Data Loading**
   - Load the dataset into a Jupyter Notebook using `pandas`.

2. **Exploratory Data Analysis (EDA)**
   - Inspect missing values and basic statistics.
   - Plot distributions of age, income, and spending score.
   - Visualize relationships (e.g., income vs. spending score).

3. **Preprocessing**
   - Select relevant numeric features.
   - Optional scaling/normalization (e.g., `StandardScaler` or `MinMaxScaler`).

4. **Choosing the Number of Clusters (K)**
   - Use the **Elbow method** (within-cluster sum of squares).
   - Optionally, use the **Silhouette score** to validate cluster quality (see explanation below).

5. **Modeling**
   - Train a **K-Means** model with the chosen `n_clusters`.
   - Assign each customer to a cluster.

6. **Visualization**
   - 2D scatter plots showing clusters (e.g., income vs. spending score, colored by cluster).
   - Cluster centroids to show typical behavior of each group.

7. **Interpretation**
   - Characterize each cluster (e.g., *"High income, high spending"*, *"Low income, low spending"*).
   - Discuss potential marketing or business strategies for each segment.

---

## Understanding the WCSS Results for Different Values of K

The values below show how the Within-Cluster Sum of Squares (WCSS) changes as we increase the number of clusters (K):

- **K=2: WCSS=588.80**
- **K=3: WCSS=476.79**
- **K=4: WCSS=388.72**
- **K=5: WCSS=331.31**
- **K=6: WCSS=276.41**
- **K=7: WCSS=236.20**
- **K=8: WCSS=199.75**
- **K=9: WCSS=174.24**
- **K=10: WCSS=152.03**

**What does this mean?**

- **WCSS** measures the compactness of the clusters: the lower the WCSS, the closer the data points are to their cluster centers.
- As K increases, WCSS decreases. Having more clusters usually means each point is closer to the center of its cluster.
- However, after a certain K value, the decrease in WCSS becomes smaller (the curve flattens). This is known as the 'elbow' in the Elbow Method.
- The 'elbow' point is the optimal K, balancing tight clusters with model simplicity. For this dataset, it is typically around K=5.

> *Choosing the right value of K using the Elbow Method helps us avoid over-segmentation while capturing the main patterns in customer behavior.*

---

## Understanding the Silhouette Analysis Plot

The **silhouette analysis plot** (`silhouette_analysis_plot.png`) is a visualization that helps evaluate the quality of clustering and determine the optimal number of clusters (K).

### What is a Silhouette Score?

The **silhouette score** measures how similar a data point is to its own cluster compared to other clusters. For each data point, it calculates:

- **a**: The average distance between the point and all other points in the same cluster (intra-cluster distance)
- **b**: The average distance between the point and all points in the nearest neighboring cluster (inter-cluster distance)

The silhouette score for a single point is calculated as:

```
silhouette = (b - a) / max(a, b)
```

### Interpreting Silhouette Scores

| Score Range | Interpretation |
|-------------|----------------|
| **+1** | The point is well-matched to its own cluster and poorly matched to neighboring clusters (ideal) |
| **0** | The point is on or very close to the decision boundary between two clusters |
| **-1** | The point may have been assigned to the wrong cluster |

### How to Read the Silhouette Analysis Plot

The plot shows:

1. **X-axis**: Number of clusters (K) tested
2. **Y-axis**: Average silhouette score for that K value
3. **Trend line**: Shows how cluster quality changes as K increases

**Key insights from the plot:**

- **Higher silhouette scores indicate better-defined clusters** - points are well-separated from neighboring clusters
- **The optimal K** is typically where the silhouette score is highest (or shows a significant peak)
- **Scores above 0.5** generally indicate reasonable cluster structure
- **Scores below 0.25** suggest weak or artificial clustering

### Example from This Project

In the `silhouette_analysis_plot.png` generated by this project:
- The plot shows silhouette scores for K values from 2 to 10
- A vertical dashed line indicates the optimal K value based on the highest silhouette score
- Each point on the curve represents the average silhouette score for all data points at that K value

### Silhouette Score vs. Elbow Method

| Method | What it Measures | Best For |
|--------|------------------|----------|
| **Elbow Method** | Within-cluster variance (WCSS) | Finding where adding more clusters gives diminishing returns |
| **Silhouette Analysis** | Cluster cohesion and separation | Validating cluster quality and separation |

**Best practice**: Use both methods together. The elbow method suggests candidate K values, while silhouette analysis validates which K produces the best-separated clusters.

---

## Technologies Used

- **Language:** Python (via Jupyter Notebook)
- **Libraries:**
  - `pandas`, `numpy` – data handling
  - `matplotlib`, `seaborn` – visualization
  - `scikit-learn` – K-Means clustering and metrics
  - `jupyter` – interactive exploration

---

## Requirements

This project has been tested with the following versions:

- pandas: **2.3.3**
- numpy: **2.3.4**
- scikit-learn: **1.7.2**
- matplotlib: **3.10.6**
- seaborn: **0.13.2**
- jupyter notebook: **7.4.5**
- jupyter core: **5.8.1**

You can install these using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## How to Run This Project

1. **Clone the repository**

   ```bash
   git clone https://github.com/JTaguiamdev/Mall-Customers-Clustering-K-means.git
   cd Mall-Customers-Clustering-K-means
   ```

2. **Create and activate a virtual environment (optional but recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   If a `requirements.txt` file is present:

   ```bash
   pip install -r requirements.txt
   ```

   Otherwise, install the main libraries manually:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

4. **Launch Jupyter Notebook**

   ```bash
   jupyter notebook
   ```

5. **Open the notebook**

   - Open the main notebook file (e.g., `Mall-Customers-Clustering-K-means.ipynb`).
   - Run the cells in order to:
     - Load data
     - Perform EDA
     - Run K-Means clustering
     - Visualize and interpret clusters

---

## Interpreting the Clusters

Typical cluster types that often appear in this dataset:

- **Cluster A – High Income, High Spending**
  - Wealthy, loyal customers.
  - Candidates for premium services, loyalty programs, and exclusive offers.

- **Cluster B – High Income, Low Spending**
  - High potential value, currently under-engaged.
  - Target with personalized promotions to increase spending.

- **Cluster C – Low/Moderate Income, High Spending**
  - Value-conscious but engaged shoppers.
  - Maintain engagement with discounts, bundles, and rewards.

- **Cluster D – Low Income, Low Spending**
  - Price-sensitive customers.
  - Consider budget-friendly products, promotions, and awareness campaigns.

> The exact cluster profiles in your notebook may differ depending on preprocessing, feature selection, and chosen K value.

---

## Possible Extensions

- Try different clustering algorithms:
  - **Hierarchical Clustering**
  - **DBSCAN**
- Apply feature scaling and compare cluster stability.
- Add more features if available (e.g., visit frequency, total annual spend).
- Build dashboards using tools like **Plotly Dash** or **Streamlit** for interactive exploration.

---

## Project Structure

A typical structure for this repository could include:

- `Mall-Customers-Clustering-K-means.ipynb` – main analysis notebook  
- `Mall_Customers.csv` (or similar) – dataset file  
- `README.md` – project documentation

---

## License

Specify a license here if you plan to share or reuse this work (e.g., MIT License). If no license is specified, the project defaults to “all rights reserved”.

---

## Author

**GitHub:** [JTaguiamdev](https://github.com/JTaguiamdev)


Feel free to open an issue or fork the repository if you’d like to suggest improvements or build on this project.
