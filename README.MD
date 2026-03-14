# 🛍️ Mall Customer Segmentation using K-Means Clustering

> An unsupervised machine learning project that segments mall customers based on demographic and spending behavior to enable smarter, data-driven marketing strategies.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Key Insights](#key-insights)

---

## Overview

Customer segmentation helps businesses understand who their customers are and how they behave. This project applies **K-Means Clustering** to group mall customers into distinct segments based on their **Age, Gender, Annual Income, and Spending Score**.

The complete workflow covers:
- Data preprocessing & feature scaling
- Optimal cluster selection via the Elbow Method
- K-Means clustering & segment labeling
- Visualization and export of results

---

## Problem Statement

Retail businesses collect large amounts of customer data but often struggle to extract actionable insights. Without proper segmentation:

- Marketing strategies become inefficient
- High-value customers are difficult to identify
- Personalization of services is nearly impossible

**Goal:** Identify natural customer groups based on spending behavior and income using machine learning.

---

## Dataset

The project uses the **Mall Customers Dataset** — demographic and behavioral data of 200 mall visitors.

| Feature | Description |
|---|---|
| `CustomerID` | Unique identifier for each customer |
| `Gender` | Male or Female |
| `Age` | Age of the customer |
| `Annual Income (k$)` | Annual income in thousands of dollars |
| `Spending Score (1–100)` | Score based on spending behavior and patterns |

- **Records:** 200 customers
- **Features:** 5 attributes

---

## Methodology

The project follows a structured data science pipeline:

```
Load Data → Preprocess → Scale Features → Find Optimal K → Cluster → Label → Visualize → Export
```

| Step | Description |
|---|---|
| **1. Data Loading** | Load dataset using Pandas |
| **2. Preprocessing** | Encode categorical variables (Gender → numeric) |
| **3. Feature Scaling** | Standardize features using `StandardScaler` |
| **4. Elbow Method** | Determine optimal number of clusters |
| **5. K-Means Clustering** | Group customers by similar characteristics |
| **6. Cluster Labeling** | Assign meaningful names to each cluster |
| **7. Visualization** | Plot clusters using scatter plots |
| **8. Export** | Save results as CSV and PNG |

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computations |
| Scikit-Learn | K-Means algorithm and feature scaling |
| Matplotlib | Cluster visualization |
| Seaborn | Statistical plotting |

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/customer-segmentation.git
cd customer-segmentation
```

### 2. Create and activate a virtual environment
```bash
# Create
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

Run the main script:

```bash
python main.py
```

The program will automatically:
1. Load and preprocess the dataset
2. Apply K-Means clustering
3. Assign cluster labels
4. Save the segmented dataset and cluster visualization

**Output files generated:**

```
data/customer_segments_final.csv
outputs/plots/final_clusters.png
```

---

## Results

### Cluster Distribution

| Cluster | Label | Customer Count |
|---|---|---|
| 0 | Young Big Spenders | 51 |
| 1 | Older Low Spenders | 49 |
| 2 | Middle Age Average | 42 |
| 3 | Middle Age Big Spenders | 38 |
| 4 | Young Medium Spenders | 20 |

The scatter plot (`final_clusters.png`) visualizes all five segments across **Annual Income** and **Spending Score** axes, with each color representing a distinct customer group.

---

## Key Insights

- **High Income + High Spending** → Premium shoppers; ideal targets for loyalty programs and upselling
- **Low Income + Low Spending** → Conservative shoppers; respond well to discounts and value deals
- **Middle Income + Moderate Spending** → Balanced segment; best suited for broad engagement campaigns

These insights can help businesses:
- Design targeted marketing campaigns
- Improve customer engagement and retention
- Optimize pricing and promotional strategies