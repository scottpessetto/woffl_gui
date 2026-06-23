"""Single-Well Analysis Page

Orchestrates the analysis views for single-well jetpump simulation.
Creates simulation objects from parameters and delegates to view renderers.

Uses ``st.segmented_control`` (not ``st.tabs``) so only the active view
executes per rerun — avoids the cost of running every heavy view (Batch,
PF Range, Pressure Profile, etc.) on every widget interaction.
"""

import streamlit as st

from woffl.gui.params import SimulationParams
from woffl.gui.tabs import (
    batch_run,
    jetpump_solver,
    jp_history_tab,
    power_fluid_range,
    pressure_profile,
    pump_equivalent,
    well_profile,
)
from woffl.gui.utils import (
    create_inflow,
    create_jetpump,
    create_pipes,
    create_reservoir_mix,
    create_well_profile,
    create_well_profile_from_survey,
    get_well_survey_data,
)


# Each view = (label, plain-language caption, renderer-key).
# The renderer-key is dispatched in _render_view below.
_VIEWS: list[tuple[str, str, str]] = [
    ("Solver", "Predicted oil, water, and power-fluid rates for the current pump.", "solver"),
    ("Batch Run", "Sweep across nozzle/throat combinations to find the best pump.", "batch"),
    ("PF Range", "How does oil rate trade off against power-fluid pressure?", "pf_range"),
    ("Pressure Profile", "Pressure traverse from surface to suction along the wellbore.", "pressure"),
    ("Well Profile", "Deviation survey (MD vs TVD) used by the simulator.", "profile"),
    ("Pump Equivalents", "Other nozzle/throat pairs with similar nozzle and throat areas.", "equivalents"),
    ("JP History", "Past pumps installed in this well.", "history"),
]


def _build_simulation_objects(params: SimulationParams):
    """Build the JetPump / pipes / inflow / res_mix / well_profile bundle once."""
    jetpump = create_jetpump(
        params.nozzle_no, params.area_ratio, params.ken, params.kth, params.kdi
    )
    _tube, _case, wellbore = create_pipes(
        params.tubing_od,
        params.tubing_thickness,
        params.casing_od,
        params.casing_thickness,
    )
    inflow = create_inflow(params.qwf, params.pwf, params.pres)
    res_mix = create_reservoir_mix(
        params.form_wc,
        params.form_gor,
        params.form_temp,
        params.field_model,
        oil_api=params.oil_api,
        gas_sg=params.gas_sg,
        wat_sg=params.wat_sg,
        bubble_point=params.bubble_point,
        model_as_water=params.model_as_water,
    )

    if params.selected_well != "Custom":
        wp = create_well_profile_from_survey(
            params.selected_well, params.jpump_tvd, params.field_model
        )
        survey_data = get_well_survey_data(params.selected_well)
        survey_present = survey_data is not None and not survey_data.empty
    else:
        wp = create_well_profile(params.field_model, params.jpump_tvd)
        survey_present = False

    return jetpump, wellbore, inflow, res_mix, wp, survey_present


def _has_jp_history(selected_well: str) -> bool:
    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None or selected_well == "Custom":
        return False
    well_jp = jp_hist[jp_hist["Well Name"] == selected_well].dropna(subset=["Date Set"])
    return not well_jp.empty


def _prefetch_jp_history_data(selected_well: str) -> None:
    """Fire-and-forget Databricks prefetch so JP History feels instant.

    The JP History view runs two ~slow Databricks queries (extended well tests
    + daily BHP) on first visit. Both are wrapped in ``@st.cache_data``, so we
    can warm the cache from a daemon thread while the user is still on Solver
    or Batch Run. By the time they click JP History, the data is ready.

    Per-well, per-session dedupe via session_state — Streamlit reruns the
    script on every widget interaction, but we only want to spawn one thread
    per well selection. Bounded threading (one daemon thread per new well
    selection, max ~one in-flight at a time in normal usage) fits inside the
    Databricks Apps medium-compute envelope (2 vCPUs).
    """
    if selected_well == "Custom":
        return
    if not _has_jp_history(selected_well):
        return

    prefetched = st.session_state.setdefault("_jp_hist_prefetched", set())
    if selected_well in prefetched:
        return

    jp_hist = st.session_state.get("jp_history_df")
    well_jp = jp_hist[jp_hist["Well Name"] == selected_well].dropna(
        subset=["Date Set"]
    )
    earliest = well_jp["Date Set"].min()

    from datetime import datetime

    start = earliest.strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")

    from woffl.assembly.well_test_client import _denormalize_well_name

    db_name = _denormalize_well_name(selected_well)

    def _worker() -> None:
        # Calling the @st.cache_data-wrapped fetchers populates the
        # process-wide cache; the JP History view will then hit warm cache.
        from woffl.gui.tabs.jp_history_tab import (
            _cached_bhp_daily,
            _cached_extended_tests,
        )

        try:
            _cached_extended_tests(db_name, start, end)
        except Exception:
            pass  # Tab will surface its own error path on click
        try:
            _cached_bhp_daily(db_name, start, end)
        except Exception:
            pass

    import threading

    from streamlit.runtime.scriptrunner import add_script_run_ctx

    thread = threading.Thread(target=_worker, daemon=True, name=f"jpprefetch-{selected_well}")
    add_script_run_ctx(thread)  # silences "missing ScriptRunContext" warnings
    thread.start()
    prefetched.add(selected_well)


def _render_view(
    view_key: str, params, jetpump, wellbore, wp, inflow, res_mix
) -> None:
    """Dispatch table for the active view."""
    if view_key == "solver":
        jetpump_solver.render_tab(params, jetpump, wellbore, wp, inflow, res_mix)
    elif view_key == "batch":
        batch_run.render_tab(params, wellbore, wp, inflow, res_mix)
    elif view_key == "pf_range":
        power_fluid_range.render_tab(params, wellbore, wp, inflow, res_mix)
    elif view_key == "pressure":
        pressure_profile.render_tab(params, jetpump, wellbore, wp, inflow, res_mix)
    elif view_key == "profile":
        well_profile.render_tab(params, wp)
    elif view_key == "equivalents":
        # Equivalents tab has its own nozzle/throat selectors so it
        # doesn't depend on the sidebar pump — no jetpump needed.
        pump_equivalent.render_tab(params)
    elif view_key == "history":
        jp_history_tab.render_tab(params)


def _handle_print_request(
    params: SimulationParams, jetpump, wellbore, wp, inflow, res_mix
) -> None:
    """If the sidebar requested a PDF, generate it now and trigger a rerun.

    The sidebar can't run this itself because the simulation objects are
    built here in the main page. So the sidebar sets ``_print_pdf_pending``
    and we pick it up at the top of ``run_single_well_page``.

    On success: stores bytes + filename in session_state, clears the
    pending flag, calls ``st.rerun()`` so the sidebar's download button
    appears on the next render. On failure: shows an error in the main
    panel and clears the flag (no rerun — keep the error visible).
    """
    if not st.session_state.get("_print_pdf_pending"):
        return

    import datetime as _dt

    from woffl.gui.pdf_export import generate_report

    engineer = st.session_state.get("print_engineer_name", "")
    notes = st.session_state.get("print_notes", "")

    with st.spinner(
        "Generating report — running batch sweep, this takes ~30 seconds…"
    ):
        try:
            pdf_bytes = generate_report(
                params, jetpump, wellbore, wp, inflow, res_mix,
                engineer_name=engineer, notes=notes,
            )
        except Exception as e:
            st.error(f"Report generation failed: {e}")
            st.session_state.pop("_print_pdf_pending", None)
            return

    today = _dt.datetime.now().strftime("%Y%m%d")
    well_slug = params.selected_well.replace("-", "")
    pump_slug = f"{params.nozzle_no}{params.area_ratio}"
    filename = f"{well_slug}_{pump_slug}_{today}.pdf"

    st.session_state["_print_pdf_bytes"] = pdf_bytes
    st.session_state["_print_pdf_filename"] = filename
    st.session_state.pop("_print_pdf_pending", None)
    st.rerun()


def run_single_well_page(params: SimulationParams) -> None:
    """Run the single-well analysis page.

    Builds the simulation object bundle once, then renders only the
    selected view. Switching views triggers a fresh render (and only that
    view's compute), so the headline Solver result stays snappy.
    """
    jetpump, wellbore, inflow, res_mix, wp, _survey_present = _build_simulation_objects(
        params
    )

    # Sidebar-driven PDF report. Has to run AFTER sim objects are built but
    # BEFORE the view renders — generation re-runs the batch and we don't
    # want both happening in the same render.
    _handle_print_request(params, jetpump, wellbore, wp, inflow, res_mix)

    # Survey-presence indicator lives in the Well Profile view (where it's
    # actionable) rather than as a header banner that repeats on every tab.

    # Filter out JP History when the well has none
    available_views = [
        v for v in _VIEWS
        if v[2] != "history" or _has_jp_history(params.selected_well)
    ]
    labels = [v[0] for v in available_views]

    # Reset persisted selection if it's no longer available — must happen
    # *before* rendering the widget (Streamlit blocks writes to a widget's
    # session_state key after render).
    if st.session_state.get("sw_active_view") not in labels:
        st.session_state["sw_active_view"] = "Solver"

    selected_label = st.segmented_control(
        "View",
        options=labels,
        key="sw_active_view",
        label_visibility="collapsed",
    )
    # st.segmented_control allows deselection — fall back to Solver
    if selected_label is None:
        selected_label = "Solver"

    # Caption beneath the view selector — what question does this view answer?
    selected_view = next(v for v in available_views if v[0] == selected_label)
    st.caption(selected_view[1])

    # Warm the JP History cache in the background while the user looks at
    # Solver/Batch Run. No-op if the user is already on JP History (its own
    # render will fetch synchronously) or has no JP history.
    if selected_view[2] != "history":
        _prefetch_jp_history_data(params.selected_well)

    with st.spinner("Running…"):
        _render_view(
            selected_view[2], params, jetpump, wellbore, wp, inflow, res_mix
        )


def show_welcome_message() -> None:
    """Display the welcome/instructions message when no simulation is running."""
    st.info("👈 Pick a well from the sidebar to get started.")

    st.markdown("""
    ### How it works
    1. **Select a well** in the sidebar — pump, geometry, and IPR auto-populate from
       Databricks and the simulation runs automatically.
    2. **Tweak the pump or pressures** in the sidebar — the analysis re-runs each time.
    3. **Switch views** at the top of the main panel:
       - **Solver** — predicted oil/water/PF rates and Model-vs-Actual comparison
       - **Batch Run** — sweep across nozzle/throat combinations
       - **PF Range** — how oil rate trades off against power-fluid pressure
       - **Pressure Profile** / **Well Profile** / **Pump Equivalents** — supporting views

    For a one-off sensitivity, pick **Custom** in the sidebar, fill the values manually,
    and click **Re-run Simulation** at the bottom of the sidebar.
    """)
