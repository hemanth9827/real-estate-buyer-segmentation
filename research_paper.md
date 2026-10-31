# Machine Learning-Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

**Organization context:** Parcl Co. Limited  
**Program:** Unified Mentor  
**Project domain:** Financial Analytics & Real Estate Market Intelligence  
**Report type:** Research paper and implementation report  
**Dataset basis:** Uploaded `clients.csv` and `properties.csv`

## Abstract
This project develops an end-to-end buyer segmentation workflow using the customer and property transaction datasets supplied with the project. The customer table contains **2,000 unique clients** and the property table contains **10,000 listing records**. The analysis links clients to sold property records using `client_id` and `client_ref`, derives buyer-level purchase features, encodes categorical attributes, scales numeric attributes, and applies K-Means clustering. The number of clusters is assessed using the elbow method and silhouette score, while Ward hierarchical clustering provides a structural comparison. A Streamlit dashboard presents buyer distribution, investment and financing patterns, geographic distribution, cluster profiles and filters.

The model selected **K = 3** with a silhouette score of **0.148**. This indicates limited cluster separation, so the clusters are exploratory profiles rather than definitive customer personas. The project avoids claiming verified wealth, luxury status or first-time buyer status because those labels are not directly established by the available fields.

## 1. Introduction
Real estate customers can differ in purchase purpose, financing, location, referral source and transaction activity. Treating all buyers identically can lead to generic recommendations and inefficient marketing. Unsupervised machine learning can help discover groups with similar observed characteristics and transaction patterns.

## 2. Problem Statement
The project addresses the need for a data-driven understanding of buyer types, investment-related purchase behaviour, geographic distribution and financing patterns. The goal is to provide interpretable customer segments and an interactive analytics dashboard to support market intelligence.

## 3. Objectives
1. Clean and validate the customer and property datasets.
2. Link customer records with completed property transactions.
3. Derive age and buyer-level transaction indicators.
4. Encode categorical fields and scale numeric features.
5. Apply K-Means clustering and compare candidate cluster counts.
6. Validate cluster structure using hierarchical clustering.
7. Interpret cluster profiles and build a Streamlit dashboard.

## 4. Dataset Description

### 4.1 Customer dataset
The uploaded customer file contains **2,000 rows** and 12 columns: `client_id`, `client_type`, `first_name`, `last_name`, `date_of_birth`, `gender`, `country`, `region`, `acquisition_purpose`, `satisfaction_score`, `loan_applied`, and `referral_channel`. First and last names are excluded from modeling and exported analytics to reduce exposure of direct personal identifiers.

### 4.2 Property dataset
The uploaded property file contains **10,000 rows** and 9 columns, including `listing_id`, `transaction_date`, `unit_category`, `floor_area_sqft`, `sale_price`, `listing_status` and `client_ref`. The supplied records include **7,305 sold listings** and **2,695 available listings**. Only sold listings with a client reference are aggregated as completed purchase activity.

### 4.3 Join and engineered variables
The datasets are joined using `clients.client_id = properties.client_ref`. The property data is aggregated to buyer level: number of properties purchased, total purchase value, average purchase price, total and average floor area, apartment count, office count and office share. All **2,000 clients** have at least one linked sold property in this supplied extract.

Age is calculated from `date_of_birth` using 31 October 2026 as the fixed analysis reference date. Sale prices are converted from currency-formatted strings to numeric values. Skewed monetary features are log-transformed before scaling.

## 5. Data Cleaning and Preprocessing
- Removed duplicate client IDs and duplicate listing IDs.
- Standardized whitespace and missing categorical values.
- Parsed mixed-format dates and derived age.
- Converted satisfaction scores and sale prices to numeric values.
- Kept only `Sold` property records with a non-null `client_ref` for completed purchase aggregates.
- Excluded names and client identifiers from the feature matrix; `client_id` is retained only as a join/export key.
- Median-imputed numeric missing values and most-frequent-imputed categorical values within the preprocessing pipeline.

## 6. Feature Encoding and Scaling
Categorical features are one-hot encoded: `client_type`, `country`, `region`, `acquisition_purpose`, `loan_applied`, and `referral_channel`. Numeric features are median-imputed and standardized with `StandardScaler`: age, satisfaction score, number of properties purchased, log total purchase value, log average purchase price, total area and office share. Gender is not used as a clustering feature.

## 7. Clustering Methodology

### 7.1 K-Means
K-Means is fitted for candidate values of K from 2 to 8, using a fixed random seed and 20 initializations. Inertia is recorded for the elbow curve. Silhouette score is used as the primary statistical selection guide.

### 7.2 Selected cluster count
The highest silhouette score in the evaluated range was **0.148 at K = 3**. The score is modest, which suggests overlapping groups and limited separation. The elbow and business interpretability should therefore be reviewed alongside the metric rather than treating K as an unquestionable answer.

### 7.3 Hierarchical clustering
Ward linkage is applied to a deterministic sample of up to 500 buyers. The adjusted Rand index between K-Means and hierarchical assignments on that sample is **0.353**. This indicates some shared structure but also meaningful differences between the methods. Hierarchical clustering is used as a validation aid, not as the production assignment model.

## 8. Exploratory Data Analysis – Key Findings
- Unique client records after deduplication: **2,000**.
- Property records: **10,000**; sold records: **7,305**.
- Investment-purpose clients: **615 (30.8%)**.
- Home-purpose clients: **1,385 (69.2%)**.
- Clients with loan applied: **736 (36.8%)**.
- Mean satisfaction score: **3.03 / 5**.
- Mean age: **56.4 years**.
- Sold transaction value represented in the linked property data: **$2,520,750,961**.
- Average sold listing price: **$345,072**.
- Buyers represented across **10 countries** and **57 regions**.

## 9. Cluster Interpretation
Cluster IDs are arbitrary and are not equivalent to the predefined personas in the project brief. The profiles below are descriptions of the measured values.

| Cluster | Buyers | Avg. age | Investment share | Loan share | Avg. properties bought | Avg. total linked purchase value | Avg. unit purchase price |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 55 | 65.6 | 32.7% | 36.4% | 7.11 | $2,383,987 | $339,913 |
| 1 | 1035 | 55.8 | 29.0% | 35.9% | 3.72 | $1,440,427 | $392,627 |
| 2 | 910 | 56.6 | 32.6% | 37.8% | 3.37 | $987,681 | $295,731 |


### Practical profile descriptions
- **Cluster 0 – Higher-activity purchasers:** 55 buyers, the highest average purchase count (7.11) and highest average total linked purchase value ($2,383,987). This is a small group; it should not automatically be called high-net-worth or luxury investors.
- **Cluster 1 – Higher average unit-price purchasers:** 1035 buyers, with the highest average purchase price ($392,627). This is a descriptive transaction profile, not a verified wealth category.
- **Cluster 2 – Lower average unit-price purchasers:** 910 buyers, with the lowest average purchase price ($295,731) and lower average total linked purchase value ($987,681).

The investment shares and loan shares are relatively similar across clusters. This means the model's strongest observed differences are more closely associated with property purchase volume and value than with the investment-purpose or loan indicators.

## 10. Streamlit Application
The dashboard implements the requested modules:
1. **Buyer Segmentation Overview:** cluster size distribution, inertia/elbow curve and silhouette scores.
2. **Investor Behavior Dashboard:** acquisition purpose, loan application, referral channel and purchase-value comparisons by cluster.
3. **Geographic Buyer Analysis:** country-level choropleth, country/region/cluster treemap and geographic count table.
4. **Segment Insights Panel:** cluster descriptive statistics and buyer records.
5. **User controls:** country, region, acquisition purpose and client type filters.
6. **Export:** download filtered segmented buyer records as CSV.

## 11. Business Recommendations
1. Use the higher-activity purchaser segment for relationship-management review and tailored portfolio servicing, after business validation.
2. Compare average unit price and total purchase value by cluster when designing property recommendation journeys.
3. Review referral channel mix by segment to understand which acquisition sources are associated with each observed profile.
4. Examine geographic differences at country and region level before planning localized campaigns.
5. Treat loan application as an indicator for optional financing education, not as a proxy for creditworthiness.
6. Validate all proposed segment actions with business stakeholders and monitor segment stability over time.

## 12. Limitations and Responsible Use
- The silhouette score is low/modest (0.148); clusters overlap and should be treated as exploratory.
- The supplied schema has no verified income, net worth, assets, investor accreditation, first-time-buyer flag or explicit unit-ownership intent. Therefore the project cannot confirm “high-income”, “high-net-worth”, “luxury investor” or “first-time buyer” personas.
- The data uses `Company` as the client type value, not `Corporate`.
- The project uses transaction data from the supplied extract; results may not represent all customers or future market conditions.
- Gender and age can be sensitive attributes. Gender is excluded from the model; age is included only because it is part of the requested demographic analysis and should be reviewed for lawful and fair use.
- Clustering is not causal and must not be used to make discriminatory decisions about pricing, access to housing, credit or service eligibility.
- Geographic visualization is country-level because the dataset provides country and region labels but not latitude/longitude for each region.

## 13. Conclusion
This project implements a full workflow for real estate buyer segmentation using the supplied customer and property datasets. It combines customer attributes with sold-property transaction aggregates, applies K-Means and hierarchical clustering, evaluates candidate cluster counts and exposes the results in a Streamlit dashboard. The selected three-cluster solution provides descriptive differences in transaction activity and purchase values, but its modest silhouette score means that it should be treated as an exploratory market-intelligence tool. Additional validated financial and customer-intent fields, along with business review, would be needed to identify the exact personas proposed in the project brief.

## References
1. MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*.
2. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.
3. Ward, J. H. (1963). Hierarchical Grouping to Optimize an Objective Function. *Journal of the American Statistical Association*, 58(301), 236–244.
4. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
