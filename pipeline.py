from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
REFERENCE_DATE = pd.Timestamp("2026-10-31")

def load_data(clients_path=None, properties_path=None):
    cp = Path(clients_path) if clients_path else DATA / "clients.csv"
    pp = Path(properties_path) if properties_path else DATA / "properties.csv"
    clients = pd.read_csv(cp)
    properties = pd.read_csv(pp)
    required_clients = ["client_id","client_type","date_of_birth","gender","country","region",
                        "acquisition_purpose","satisfaction_score","loan_applied","referral_channel"]
    required_properties = ["listing_id","transaction_date","unit_category","floor_area_sqft",
                           "sale_price","listing_status","client_ref"]
    for label, frame, required in [
        ("clients",clients,required_clients),("properties",properties,required_properties)]:
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"{label} data is missing required columns: {missing}")
    return clients, properties

def prepare_data(clients, properties):
    c = clients.copy()
    c = c.drop_duplicates(subset=["client_id"], keep="first")
    # PII fields are not used by the model or exported analytics.
    c = c.drop(columns=[x for x in ["first_name","last_name"] if x in c.columns])
    for col in ["client_type","gender","country","region","acquisition_purpose","loan_applied","referral_channel"]:
        c[col] = c[col].astype("string").str.strip().replace({"":pd.NA,"nan":pd.NA,"None":pd.NA})
        c[col] = c[col].fillna("Unknown").astype(str)
    c["dob_parsed"] = pd.to_datetime(c["date_of_birth"], format="mixed", errors="coerce", dayfirst=False)
    c["age"] = ((REFERENCE_DATE - c["dob_parsed"]).dt.days / 365.25).round()
    c.loc[~c["age"].between(18,100), "age"] = np.nan
    c["satisfaction_score"] = pd.to_numeric(c["satisfaction_score"], errors="coerce")
    c.loc[~c["satisfaction_score"].between(1,5), "satisfaction_score"] = np.nan

    p = properties.copy()
    p["sale_price_num"] = pd.to_numeric(p["sale_price"].astype(str).str.replace(r"[$,]","",regex=True),errors="coerce")
    p["floor_area_sqft"] = pd.to_numeric(p["floor_area_sqft"],errors="coerce")
    p["transaction_dt"] = pd.to_datetime(p["transaction_date"],format="mixed",errors="coerce",dayfirst=False)
    p["status_norm"] = p["listing_status"].astype(str).str.strip().str.lower()
    sold = p[(p["status_norm"]=="sold") & p["client_ref"].notna()].copy()
    sold = sold.drop_duplicates(subset=["listing_id"],keep="first")
    sold["unit_category"] = sold["unit_category"].astype(str).str.strip()
    agg = sold.groupby("client_ref").agg(
        properties_purchased=("listing_id","nunique"),
        total_purchase_value=("sale_price_num","sum"),
        avg_purchase_price=("sale_price_num","mean"),
        total_area_sqft=("floor_area_sqft","sum"),
        avg_area_sqft=("floor_area_sqft","mean"),
        apartment_units=("unit_category",lambda x:(x.str.lower()=="apartment").sum()),
        office_units=("unit_category",lambda x:(x.str.lower()=="office").sum()),
        first_purchase_date=("transaction_dt","min"),
        last_purchase_date=("transaction_dt","max")
    ).reset_index().rename(columns={"client_ref":"client_id"})
    m = c.merge(agg,on="client_id",how="left")
    # Buyers without a linked sold property remain in the analysis with zero transaction aggregates.
    transaction_cols=["properties_purchased","total_purchase_value","avg_purchase_price",
                      "total_area_sqft","avg_area_sqft","apartment_units","office_units"]
    m[transaction_cols] = m[transaction_cols].fillna(0)
    m["office_share"] = np.where(m["properties_purchased"]>0,
                                  m["office_units"]/m["properties_purchased"],0)
    m["log_total_purchase_value"] = np.log1p(m["total_purchase_value"])
    m["log_avg_purchase_price"] = np.log1p(m["avg_purchase_price"])
    return m, p, sold

CAT_FEATURES = ["client_type","country","region","acquisition_purpose","loan_applied","referral_channel"]
NUM_FEATURES = ["age","satisfaction_score","properties_purchased","log_total_purchase_value",
                "log_avg_purchase_price","total_area_sqft","office_share"]

def run_analysis(clients_path=None, properties_path=None, max_k=8):
    clients, properties = load_data(clients_path,properties_path)
    m, p, sold = prepare_data(clients,properties)
    categorical = Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])
    numeric = Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])
    prep = ColumnTransformer([("categorical",categorical,CAT_FEATURES),
                              ("numeric",numeric,NUM_FEATURES)])
    X = prep.fit_transform(m)
    dense = X.toarray() if hasattr(X,"toarray") else X
    rows=[]
    upper=min(max_k,len(m)-1)
    for k in range(2,upper+1):
        model=KMeans(n_clusters=k,random_state=42,n_init=20)
        labels=model.fit_predict(dense)
        sample = dense if len(dense)<=2000 else dense[np.linspace(0,len(dense)-1,2000,dtype=int)]
        sample_labels = labels if len(dense)<=2000 else labels[np.linspace(0,len(dense)-1,2000,dtype=int)]
        rows.append({"k":k,"inertia":float(model.inertia_),
                     "silhouette":float(silhouette_score(sample,sample_labels))})
    eval_df=pd.DataFrame(rows)
    # Use silhouette as the primary statistical guide, then validate interpretability.
    best_k=int(eval_df.sort_values(["silhouette","k"],ascending=[False,True]).iloc[0]["k"])
    kmeans=KMeans(n_clusters=best_k,random_state=42,n_init=20)
    labels=kmeans.fit_predict(dense)
    m["cluster"]=labels

    # Ward hierarchical clustering on a deterministic sample; comparison, not production labels.
    sample_n=min(500,len(m))
    sample_idx=np.linspace(0,len(m)-1,sample_n,dtype=int)
    Z=linkage(dense[sample_idx],method="ward")
    hlabels=fcluster(Z,t=best_k,criterion="maxclust")
    ari=adjusted_rand_score(labels[sample_idx],hlabels)

    # Output table excludes direct identifiers and names; retain client_id only for joining internally.
    export_cols=["client_id","client_type","gender","country","region","date_of_birth","age",
                 "acquisition_purpose","satisfaction_score","loan_applied","referral_channel",
                 "properties_purchased","total_purchase_value","avg_purchase_price",
                 "total_area_sqft","avg_area_sqft","apartment_units","office_units","office_share","cluster"]
    segmented=m[export_cols].copy()
    segmented.to_csv(OUT/"segmented_buyers.csv",index=False)
    eval_df.to_csv(OUT/"cluster_evaluation.csv",index=False)

    profile=m.groupby("cluster").agg(
        buyers=("client_id","count"),
        average_age=("age","mean"),
        average_satisfaction=("satisfaction_score","mean"),
        investment_share=("acquisition_purpose",lambda x:(x.astype(str).str.lower()=="investment").mean()),
        loan_share=("loan_applied",lambda x:(x.astype(str).str.lower()=="yes").mean()),
        company_share=("client_type",lambda x:(x.astype(str).str.lower()=="company").mean()),
        average_properties_purchased=("properties_purchased","mean"),
        average_total_purchase_value=("total_purchase_value","mean"),
        average_purchase_price=("avg_purchase_price","mean"),
        average_area_sqft=("avg_area_sqft","mean"),
        office_purchase_share=("office_share","mean")
    ).reset_index()
    profile.to_csv(OUT/"cluster_profiles.csv",index=False)

    # EDA summary
    eda = {
      "client_records_raw":int(len(clients)),
      "unique_clients_after_dedup":int(len(m)),
      "property_records":int(len(properties)),
      "sold_property_records":int(len(sold)),
      "buyers_with_sold_property":int(m["properties_purchased"].gt(0).sum()),
      "available_listings":int((p["status_norm"]=="available").sum()),
      "countries":int(m["country"].nunique()),
      "regions":int(m["region"].nunique()),
      "investment_clients":int((m["acquisition_purpose"].astype(str).str.lower()=="investment").sum()),
      "home_clients":int((m["acquisition_purpose"].astype(str).str.lower()=="home").sum()),
      "loan_yes":int((m["loan_applied"].astype(str).str.lower()=="yes").sum()),
      "mean_satisfaction":float(m["satisfaction_score"].mean()),
      "mean_age":float(m["age"].mean()),
      "total_sold_value":float(sold["sale_price_num"].sum()),
      "mean_sold_price":float(sold["sale_price_num"].mean()),
      "best_k":best_k,
      "best_silhouette":float(eval_df.loc[eval_df.k==best_k,"silhouette"].iloc[0]),
      "hierarchical_ari_sample":float(ari)
    }
    pd.Series(eda).to_json(OUT/"eda_summary.json",indent=2)
    return m, p, sold, eval_df, profile, eda

if __name__=="__main__":
    m,p,sold,ev,prof,eda=run_analysis()
    print("EDA SUMMARY")
    for k,v in eda.items(): print(f"{k}: {v}")
    print("\nCLUSTER PROFILES")
    print(prof.round(3).to_string(index=False))
    print("\nSaved model outputs to",OUT)
