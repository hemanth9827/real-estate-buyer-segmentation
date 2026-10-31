from pathlib import Path
import sys, tempfile
import streamlit as st
import pandas as pd
import plotly.express as px
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from pipeline import run_analysis

st.set_page_config(page_title="Parcl Buyer Intelligence",page_icon="🏢",layout="wide")
st.title("Real Estate Buyer Segmentation & Investment Profiling")
st.caption("Parcl Co. Limited | Customer and property transaction analytics | K-Means + Hierarchical validation")
st.info("This dashboard uses the client and property CSV files supplied with the project. Replace them with an updated authorized data extract when required.")
with st.sidebar:
    st.header("Data sources")
    client_upload=st.file_uploader("Upload updated clients CSV (optional)",type=["csv"],key="clients")
    property_upload=st.file_uploader("Upload updated properties CSV (optional)",type=["csv"],key="properties")
    st.caption("If no files are uploaded, the bundled project datasets are used.")
    run=st.button("Refresh analysis",type="primary")
if "analysis_done" not in st.session_state or run or client_upload is not None or property_upload is not None:
    cp=pp=None
    if client_upload is not None:
        cp=ROOT/"data"/"clients_uploaded.csv"; cp.write_bytes(client_upload.getvalue())
    if property_upload is not None:
        pp=ROOT/"data"/"properties_uploaded.csv"; pp.write_bytes(property_upload.getvalue())
    with st.spinner("Cleaning data, joining transactions and training clustering models..."):
        m,p,sold,ev,prof,eda=run_analysis(cp,pp)
    st.session_state["analysis_done"]=True
    st.session_state["analysis_cache"]=(m,p,sold,ev,prof,eda)
m,p,sold,ev,prof,eda=st.session_state["analysis_cache"]
# Filter UI
with st.sidebar:
    st.header("Buyer filters")
    countries=sorted(m.country.astype(str).unique())
    regions=sorted(m.region.astype(str).unique())
    purposes=sorted(m.acquisition_purpose.astype(str).unique())
    types=sorted(m.client_type.astype(str).unique())
    fc=st.multiselect("Country",countries,default=countries)
    fr=st.multiselect("Region",regions,default=regions)
    fp=st.multiselect("Acquisition purpose",purposes,default=purposes)
    ft=st.multiselect("Client type",types,default=types)
f=m[m.country.astype(str).isin(fc)&m.region.astype(str).isin(fr)&m.acquisition_purpose.astype(str).isin(fp)&m.client_type.astype(str).isin(ft)].copy()
if f.empty:
    st.warning("No records match the selected filters.")
    st.stop()
kpis=st.columns(5)
kpis[0].metric("Buyers",f"{len(f):,}")
kpis[1].metric("Investment buyers",f"{(f.acquisition_purpose.astype(str).str.lower()=='investment').mean():.1%}")
kpis[2].metric("Loan applicants",f"{(f.loan_applied.astype(str).str.lower()=='yes').mean():.1%}")
kpis[3].metric("Avg. satisfaction",f"{f.satisfaction_score.mean():.2f}/5")
kpis[4].metric("Clusters",str(f.cluster.nunique()))
tab1,tab2,tab3,tab4=st.tabs(["Buyer segmentation overview","Investor behaviour","Geographic analysis","Segment insights"])
with tab1:
    st.subheader("Buyer distribution by cluster")
    counts=f.cluster.value_counts().sort_index().rename_axis("Cluster").reset_index(name="Buyers")
    st.plotly_chart(px.pie(counts,names="Cluster",values="Buyers",hole=.45,title="Share of buyers in each cluster"),use_container_width=True)
    st.plotly_chart(px.bar(counts,x="Cluster",y="Buyers",color="Cluster",text="Buyers",title="Cluster size"),use_container_width=True)
    st.subheader("Choosing the number of clusters")
    a,b=st.columns(2)
    a.plotly_chart(px.line(ev,x="k",y="inertia",markers=True,title="Elbow method (inertia)"),use_container_width=True)
    b.plotly_chart(px.line(ev,x="k",y="silhouette",markers=True,title="Silhouette score"),use_container_width=True)
    st.caption(f"Selected K = {eda['best_k']} based on the highest silhouette score. Silhouette = {eda['best_silhouette']:.3f}. Inspect cluster sizes and business meaning before operational use.")
with tab2:
    st.subheader("Investment purpose by cluster")
    pur=f.groupby(["cluster","acquisition_purpose"]).size().reset_index(name="buyers")
    st.plotly_chart(px.bar(pur,x="cluster",y="buyers",color="acquisition_purpose",barmode="group"),use_container_width=True)
    loan=f.groupby(["cluster","loan_applied"]).size().reset_index(name="buyers")
    st.plotly_chart(px.bar(loan,x="cluster",y="buyers",color="loan_applied",barmode="group"),use_container_width=True)
    ch=f.groupby(["cluster","referral_channel"]).size().reset_index(name="buyers")
    st.plotly_chart(px.bar(ch,x="cluster",y="buyers",color="referral_channel",barmode="stack"),use_container_width=True)
    spend=f.groupby("cluster").agg(avg_total_purchase_value=("total_purchase_value","mean"),avg_purchase_price=("avg_purchase_price","mean")).reset_index()
    st.plotly_chart(px.bar(spend,x="cluster",y="avg_total_purchase_value",title="Average total linked purchase value by cluster"),use_container_width=True)
with tab3:
    st.subheader("Buyer geography")
    country_counts=f.groupby("country").size().reset_index(name="buyers")
    iso={"USA":"USA","UK":"GBR","Canada":"CAN","Germany":"DEU","France":"FRA","Belgium":"BEL","Mexico":"MEX","Russia":"RUS","Australia":"AUS","Denmark":"DNK"}
    country_counts["iso3"]=country_counts.country.map(iso)
    mapped=country_counts.dropna(subset=["iso3"])
    if not mapped.empty:
        fig=px.choropleth(mapped,locations="iso3",color="buyers",hover_name="country",locationmode="ISO-3",
                          title="Buyer count by country")
        st.plotly_chart(fig,use_container_width=True)
    reg=f.groupby(["country","region","cluster"]).size().reset_index(name="buyers")
    st.plotly_chart(px.treemap(reg,path=["country","region","cluster"],values="buyers",color="cluster",title="Buyer distribution by country, region and cluster"),use_container_width=True)
    st.dataframe(f.groupby(["country","region"]).size().reset_index(name="buyers").sort_values("buyers",ascending=False),use_container_width=True)
with tab4:
    st.subheader("Cluster profile table")
    st.dataframe(prof.round(2),use_container_width=True)
    for _,r in prof.iterrows():
        st.markdown(f"**Cluster {int(r.cluster)}** — {int(r.buyers)} buyers; average age {r.average_age:.1f}; investment share {r.investment_share:.1%}; loan share {r.loan_share:.1%}; company share {r.company_share:.1%}; average linked purchase value ${r.average_total_purchase_value:,.0f}.")
    st.subheader("Filtered buyer records")
    st.dataframe(f.drop(columns=["dob_parsed","first_purchase_date","last_purchase_date"],errors="ignore"),use_container_width=True)
st.download_button("Download filtered buyer segmentation CSV",f.to_csv(index=False).encode("utf-8"),"parcl_segmented_buyers.csv","text/csv")
st.caption("Privacy: first and last names are excluded from model features and dashboard outputs. Cluster IDs are arbitrary; clustering is exploratory and must not be used for discriminatory decisions.")
