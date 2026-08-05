"""WOFFL web backend - FastAPI JSON API serving the React SPA.

Read-only in v1: no endpoint in this package may call
``databricks_client.execute_write`` / ``prop_hist_client.push_prop`` or any
other write path. The production write gate (``ALLOW_DATABRICKS_WRITES``)
is reported via /api/meta for UI display only.

Imports from ``woffl.gui`` are restricted to the Streamlit-free modules:
``params``, ``vogel``, ``ipr_anchor``, ``pump_identity``, ``tab_helpers``.
Everything else the server needs lives in ``server/services``.
"""
