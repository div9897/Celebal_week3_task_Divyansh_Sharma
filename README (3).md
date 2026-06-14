# Customer Intelligence System: Country Segmentation for HELP International

An end-to-end unsupervised + supervised learning project that segments countries by socio-economic and health development, then trains classifiers to automatically categorize countries into aid-priority tiers.

## Project Background

**HELP International** is an international humanitarian NGO focused on fighting poverty and providing basic amenities and disaster relief to people in underdeveloped countries. The NGO has raised around **$10 million** and needs to decide which countries are in the direst need of aid.

**Objective:** Use socio-economic and health factors to categorize countries by overall development level, identify the countries that should be prioritized for aid, and build a reusable model that can classify new/updated country data without re-running clustering from scratch.

## Dataset

- **Source:** [Unsupervised Learning on Country Data (Kaggle)](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data)
- **File:** `Country-data.csv` (167 countries, 9 socio-economic/health indicators)

| Column | Description |
|---|---|
| `country` | Name of the country |
| `child_mort` | Deaths of children under 5 per 1,000 live births |
| `exports` | Exports of goods/services per capita (% of GDP per capita) |
| `health` | Total health spending per capita (% of GDP per capita) |
| `imports` | Imports of goods/services per capita (% of GDP per capita) |
| `income` | Net income per person |
| `inflation` | Annual growth rate of total GDP |
| `life_expec` | Average life expectancy (years) |
| `total_fer` | Total fertility rate (children per woman) |
| `gdpp` | GDP per capita |

## Approach / Workflow

1. **Load & inspect** the raw data
2. **Clean & preprocess** — standardize column names, remove duplicates, handle missing values
3. **EDA** — correlation heatmap, boxplots, and distribution plots for all features
4. **Feature scaling** — `StandardScaler` applied to all numeric indicators
5. **K-Means clustering** — elbow method + silhouette scores to choose `k`, final model trained with `k=3`
6. **DBSCAN** — density-based clustering as a second method, used to flag outlier countries
7. **PCA visualization** — 2D projection of clusters from both K-Means and DBSCAN
8. **Cluster profiling** — average indicator values per cluster
9. **Derived label** — clusters ranked by `gdpp`, `income`, `life_expec`, and `child_mort` into a business-meaningful `development_category`: **High Need / Medium Need / Low Need**
10. **Classification** — Random Forest and XGBoost trained to predict `development_category` from raw features (so future country data can be scored without re-clustering)
11. **Evaluation** — accuracy, F1, confusion matrices, and feature importance for both models
12. **Final insights** — ranked list of countries most in need of aid

## Key Results

| Cluster | Label | # Countries |
|---|---|---|
| 0 | Low Need | 36 |
| 1 | High Need | 47 |
| 2 | Medium Need | 84 |

- K-Means silhouette score (k=3): **0.283**
- DBSCAN: 137 countries in the main cluster, 30 flagged as outliers (noise)
- Both **Random Forest** and **XGBoost** achieved **100% accuracy / F1** on the held-out test set when predicting `development_category` (expected, since the label is derived directly from the clustering — the classifiers learn to replicate the cluster boundaries from raw features, enabling fast scoring of new data)
- Across both models, `child_mort`, `income`, and `gdpp` were consistently the most important features for distinguishing development categories

### Top Aid-Priority Countries (High Need, lowest `gdpp`)

1. Burundi
2. Liberia
3. Congo, Dem. Rep.
4. Niger
5. Sierra Leone
6. Madagascar
7. Mozambique
8. Central African Republic
9. Malawi
10. Eritrea

*(Full ranked list of 47 High Need countries is in the notebook.)*

## Recommendation

HELP International should prioritize the **High Need** countries — those with the lowest `gdpp`/`income` and highest `child_mort`/`total_fer` — when allocating the $10M aid budget. The trained classifiers can be reused to score newly collected or updated country data into the same High/Medium/Low Need categories without repeating the full clustering pipeline.

## Files

| File | Description |
|---|---|
| `week_3_Divyansh_Sharma_Country_Data_Customer_Intelligence_System.ipynb` | Main Jupyter notebook with the full pipeline |
| `Country-data.csv` | Input dataset |
| `README.md` | This file |

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## How to Run

1. Place `Country-data.csv` in the same directory as the notebook
2. Open `week_3_Divyansh_Sharma_Country_Data_Customer_Intelligence_System.ipynb` in Jupyter
3. Run all cells from top to bottom

## Tech Stack

- **Clustering:** K-Means, DBSCAN
- **Dimensionality Reduction:** PCA
- **Classification:** Random Forest, XGBoost
- **Visualization:** Matplotlib, Seaborn
