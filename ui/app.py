"""Streamlit UI. Talks to the API only (the same contract the judges use)."""
from __future__ import annotations

import json
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Order Allocation", layout="wide")


def call(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=300, **kw)
        return r.json() if r.headers.get("content-type", "").startswith("application/json") else {"error": r.text}
    except Exception as exc:
        return {"error": str(exc)}


def tradeoff_chart(points: list[dict], current: dict | None = None):
    """R11: one chart with cost, carbon and delivery (late orders as label and marker size)."""
    df = pd.DataFrame([p for p in points if p.get("feasible", True)])
    if df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["carbon_t"], y=df["cost"], mode="lines+markers+text",
        marker=dict(size=10 + 1.2 * df["late_orders"], color=df["late_orders"], colorscale="OrRd",
                    showscale=True, colorbar=dict(title="late orders")),
        text=[f"-{c:g}% | {l} late" for c, l in zip(df["carbon_cut_pct"], df["late_orders"])],
        textposition="top center", name="cost-carbon frontier",
        hovertemplate="carbon %{x:.1f} t<br>cost %{y:,.0f}<extra></extra>"))
    if current:
        fig.add_trace(go.Scatter(x=[current["total_carbon_t"]], y=[current["total_cost"]], mode="markers",
                                 marker=dict(size=18, symbol="star", color="#1f6f5c"), name="current plan"))
    fig.update_layout(xaxis_title="Total carbon (t CO2e)", yaxis_title="Total landed cost", height=420,
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    return fig


def show(resp: dict):
    if "error" in resp and "answer_type" not in resp:
        st.error(resp["error"])
        return
    at = resp.get("answer_type")
    colour = {"allocation": "green", "scenario": "blue", "brief": "violet", "infeasible": "orange",
              "clarification_needed": "gray", "refused": "red"}.get(at, "gray")
    st.markdown(f"**Answer type:** :{colour}[{at}] &nbsp; run `{resp.get('run_id')}` &nbsp; "
                f"{resp.get('latency_ms', 0) / 1000:.1f}s")
    st.markdown(resp.get("summary", ""))
    k = resp.get("kpis")
    if k:
        c = st.columns(5)
        c[0].metric("Landed cost", f"{k['total_cost']:,.0f}")
        c[1].metric("Carbon (t)", f"{k['total_carbon_t']:,.1f}")
        c[2].metric("Late orders", k["late_orders"])
        c[3].metric("Unserved", k.get("unserved_orders", 0))
        v = resp.get("validator") or {}
        c[4].metric("Checker", "valid" if v.get("valid") else "issues")
        st.bar_chart(pd.Series(k["plant_utilisation"], name="utilisation"))
    if resp.get("tradeoff"):
        fig = tradeoff_chart(resp["tradeoff"], k)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame(resp["tradeoff"]), width="stretch", hide_index=True)
    if resp.get("moved_orders"):
        st.subheader(f"Orders that moved ({len(resp['moved_orders'])})")
        st.dataframe(pd.DataFrame(resp["moved_orders"]), width="stretch", hide_index=True)
    if resp.get("explanation"):
        st.subheader("Options for this order")
        st.dataframe(pd.DataFrame(resp["explanation"]["options"]), width="stretch", hide_index=True)
    if resp.get("flagged_orders"):
        with st.expander(f"Flagged orders ({len(resp['flagged_orders'])})"):
            st.dataframe(pd.json_normalize(resp["flagged_orders"]), width="stretch", hide_index=True)
    if resp.get("recommendations"):
        with st.expander("Recommendations"):
            st.json(resp["recommendations"])
    if resp.get("allocation"):
        with st.expander(f"Allocation ({len(resp['allocation'])} orders)"):
            st.dataframe(pd.DataFrame(resp["allocation"]), width="stretch", hide_index=True)
    with st.expander("Assumptions and limitations"):
        for a in resp.get("assumptions", []):
            st.markdown(f"- {a}")
        for a in resp.get("limitations", []):
            st.markdown(f"- *{a}*")
    if resp.get("run_id"):
        with st.expander("Trace"):
            st.json(call("GET", f"/trace/{resp['run_id']}"))


st.title("Multi-Plant Order Allocation")
h = call("GET", "/health")
st.caption(f"API {h.get('status', 'unreachable')} · data {h.get('data_store', {}).get('rows', {})} · "
           f"LLM {h.get('llm')} · solver {h.get('config', {}).get('solver')}")

with st.sidebar:
    st.header("Weights")
    w_cost = st.slider("Cost weight", 0.0, 1.0, 0.7, 0.05, key="w_cost")
    st.caption(f"Carbon weight {1 - w_cost:.2f}. Default 0.7/0.3 (see README).")
    w_late = st.slider("Lateness weight", 0.0, 5.0, 1.0, 0.5, key="w_late")
    st.header("Capacity what-if")
    plant = st.selectbox("Plant", ["A", "B", "C"], key="cap_plant")
    loss = st.slider("Capacity lost (%)", 0, 100, 30, 5, key="cap_loss")
    weeks = st.multiselect("Weeks (empty = whole horizon)", [1, 2, 3, 4, 5, 6], key="cap_weeks")
    run_whatif = st.button("Re-plan with this loss", key="btn_whatif")

tab_ask, tab_trade, tab_data = st.tabs(["Ask", "Trade-off", "Data"])
weights = {"cost": w_cost, "carbon": round(1 - w_cost, 2), "lateness": w_late}

with tab_ask:
    examples = ["Allocate this week's orders.", "What is the cheapest plan that is on time?",
                "What does a 10% carbon cut cost?", "Plant B loses 30% capacity. Re-plan.",
                "Re-plan for the capacity event", "Why was order 77 built in Plant C?",
                "Write the brief for the leadership team."]
    pick = st.selectbox("Examples", examples, key="example")
    q = st.text_area("Request", value=pick, key="request", height=80)
    use_w = st.checkbox("Send the sidebar weights with the request", value=False, key="use_w")
    if st.button("Run", type="primary", key="btn_run"):
        with st.spinner("Planning..."):
            st.session_state["last"] = call("POST", "/analyze", json={"request": q, "weights": weights if use_w else None})
    if run_whatif:
        sc = {"capacity_changes": [{"plant": plant, "loss_pct": loss, **({"weeks": weeks} if weeks else {})}]}
        with st.spinner("Re-planning..."):
            st.session_state["last"] = call("POST", "/analyze", json={"request": f"Plant {plant} loses {loss}% capacity. Re-plan.",
                                                                     "weights": weights, "scenario": sc})
    if st.session_state.get("last"):
        show(st.session_state["last"])

with tab_trade:
    st.write("Cost of each carbon cut, from the solver (epsilon-constraint). Marker size and colour = late orders.")
    if st.button("Compute trade-off", key="btn_trade"):
        with st.spinner("Solving one plan per carbon cut..."):
            st.session_state["trade"] = call("POST", "/analyze", json={"request": "Show the cost-carbon trade-off",
                                                                      "weights": {"lateness": w_late}})
    if st.session_state.get("trade"):
        show(st.session_state["trade"])

with tab_data:
    st.subheader("Load a CSV (Annex B schema)")
    table = st.selectbox("Table", ["orders", "plants", "transport_duties", "capacity_events"], key="ing_table")
    f = st.file_uploader("CSV file", type=["csv"], key="ing_file")
    if f and st.button("Upload", key="btn_upload"):
        st.json(call("POST", "/ingest", files={"file": (f.name, f.getvalue(), "text/csv")},
                     data={"meta": json.dumps({"table": table})}))
    st.subheader("Validation report")
    if st.button("Run validation", key="btn_validate"):
        rep = call("GET", "/validate")
        if "checks" in rep:
            st.dataframe(pd.DataFrame(rep["checks"]), width="stretch", hide_index=True)
            st.write("Valid" if rep["valid"] else "Invalid", rep.get("row_counts"))
        else:
            st.json(rep)
    st.subheader("Look up a trace")
    rid = st.text_input("run_id", key="trace_id")
    if rid:
        st.json(call("GET", f"/trace/{rid}"))
