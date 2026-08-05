"""Domain services - Streamlit-free wrappers over woffl.assembly and the
Databricks clients. All Databricks reads are TTL-cached here (server/cache.py)
with the same windows the Streamlit app used."""
