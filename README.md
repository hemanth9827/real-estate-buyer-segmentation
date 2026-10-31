# Parcl Real Estate Buyer Segmentation & Investment Profiling

## Project contents
- `data/clients.csv`: customer dataset supplied for the project (2,000 rows).
- `data/properties.csv`: property listings/transactions dataset supplied for the project (10,000 rows).
- `src/pipeline.py`: preprocessing, client-property join, feature engineering, one-hot encoding, scaling, K-Means, silhouette/elbow evaluation and Ward hierarchical comparison.
- `app.py`: interactive Streamlit dashboard with filters and four requested modules.
- `outputs/`: generated segmentation, cluster profiles, evaluation metrics and EDA summary.
- `reports/research_paper.pdf`: formatted report based on the supplied datasets and project requirements.
- `reports/research_paper.md`: editable report text.

## Run on Windows
Open this folder in VS Code, then run in terminal:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/pipeline.py
streamlit run app.py
```
Open the local Streamlit URL shown in the terminal (normally http://localhost:8501).

## Data handling and methodology
The pipeline uses the supplied client and property files. It joins `clients.client_id` to `properties.client_ref`, and aggregates only records whose `listing_status` is `Sold`. Client names are removed before modeling. `client_id` is retained only as a key and is not a feature.

The model features include client type, country, region, acquisition purpose, loan status, referral channel, derived age, satisfaction, count of sold properties, total and average linked purchase values, area and office share. Categorical fields are one-hot encoded; numeric fields are median-imputed and standardized. Candidate K values 2–8 are evaluated. The highest silhouette score selects K. Ward hierarchical clustering is performed on a deterministic sample of up to 500 records and compared using adjusted Rand index.

## Important interpretation note
The PRD's suggested labels (Global Investors, First-Time Buyers, Corporate Buyers, Luxury Investors) are hypotheses, not labels present in the data. The provided columns do not contain an explicit first-time-buyer flag, verified income/net worth, or unit portfolio ownership beyond the linked transaction history. The report therefore describes the observed clusters using measured attributes and does not claim unsupported high-income/luxury status.

## Dataset caveats
- `client_type` uses `Company` in the supplied file, rather than `Corporate`.
- `acquisition_purpose` uses `Home` and `Investment`.
- Country values use abbreviations such as USA and UK.
- Property `sale_price` contains currency symbols and commas; pipeline parses it numerically.
- Available listings are not counted as completed purchases; only Sold rows with a client reference are aggregated.
- Date parsing uses month-first mixed-format parsing based on the supplied file's date examples.
- Review data governance, lawful use, and consent before production deployment.
