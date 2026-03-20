"""
Export SFD to Vensim .mdl format.

Generates a valid .mdl file following Vensim conventions:
  - Stocks use INTEG(inflow - outflow, initial_value)
  - Flows use "A FUNCTION OF(dependencies)" — Vensim computes
    the equation from the sketch structure
  - Information dependencies are encoded as causal arrows
  - Material flows use pipe arrows connecting stocks through valves

Layout coordinates are generated but kept minimal — the user can
rearrange elements in Vensim's GUI.

Reference: https://www.vensim.com/documentation/ref_sketch_format.html
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from fedsfd.sfd.model import AuxVariable, Flow, SFD, Stock


def _vensim_name(name: str) -> str:
    """Convert an internal name to a Vensim variable name."""
    return name.replace("_", " ").strip()


def _flow_equation_str(flow: Flow, sfd: SFD) -> str:
    """Generate the Vensim equation for a flow.

    In proper SFD style, the flow's equation references the stocks
    and variables it depends on.  We use "A FUNCTION OF(...)"
    with signed stock references for the material-flow connections
    and info-dependency variables.
    """
    deps = []

    # Material flow connection: source stock only (the flow rate depends
    # on the stock it drains, NOT on the sink that receives material)
    if flow.source and not flow.source.is_cloud:
        deps.append(_vensim_name(flow.source.name))

    # Information dependencies
    for info_dep in sfd.get_dependencies_for(flow):
        vname = _vensim_name(info_dep.source.name)
        if vname not in deps and f"-{vname}" not in deps:
            deps.append(vname)

    if deps:
        return f"A FUNCTION OF({','.join(deps)})"
    return "A FUNCTION OF( )"


def _stock_equation_str(stock: Stock, sfd: SFD) -> str:
    """Generate the INTEG equation for a stock.

    Uses signed flow names: inflows with + (or just the name),
    outflows with - prefix.
    """
    inflows = sfd.get_stock_inflows(stock)
    outflows = sfd.get_stock_outflows(stock)

    flow_terms = []
    for f in inflows:
        flow_terms.append(_vensim_name(f.name))
    for f in outflows:
        flow_terms.append(f"-{_vensim_name(f.name)}")

    if not flow_terms:
        return f"{stock.initial_value:.6f}"

    net_flow = "+".join(flow_terms).replace("+-", "-")
    return f"INTEG({net_flow},{stock.initial_value:.6f})"


def _aux_equation_str(aux: AuxVariable, sfd: SFD) -> str:
    """Generate the equation for an auxiliary variable."""
    deps = sfd.get_dependencies_for(aux)
    if deps:
        dep_names = [_vensim_name(d.source.name) for d in deps]
        return f"A FUNCTION OF({','.join(dep_names)})"
    return "A FUNCTION OF( )"


def export_to_mdl(
    sfd: SFD,
    output_path: str | Path,
    time_horizon: int = 100,
    dt: float = 1.0,
) -> None:
    """Export an SFD to Vensim .mdl format.

    Produces a file that Vensim can open with proper stock-flow structure:
    stocks as boxes, flows as valves with pipe arrows, auxiliaries as
    plain variables, and information dependencies as thin arrows.
    """
    output_path = Path(output_path)
    lines = ["{UTF-8}"]

    # --- Equations section ---

    # Stocks
    for stock in sfd.stocks:
        vname = _vensim_name(stock.name)
        eq = _stock_equation_str(stock, sfd)
        lines.append(f"{vname}= {eq}")
        lines.append("\t~\t")
        lines.append(f"\t~\t|")
        lines.append("")

    # Flows
    for flow in sfd.flows:
        vname = _vensim_name(flow.name)
        eq = _flow_equation_str(flow, sfd)
        lines.append(f"{vname}=")
        lines.append(f"\t{eq}")
        lines.append("\t~\t")
        lines.append(f"\t~\t|")
        lines.append("")

    # Auxiliaries
    for aux in sfd.auxiliaries:
        vname = _vensim_name(aux.name)
        eq = _aux_equation_str(aux, sfd)
        lines.append(f"{vname}=")
        lines.append(f"\t{eq}")
        lines.append("\t~\t")
        lines.append(f"\t~\t|")
        lines.append("")

    # Simulation control
    lines.append("********************************************************")
    lines.append("\t.Control")
    lines.append("********************************************************~")
    lines.append("\t\tSimulation Control Parameters")
    lines.append("\t|")
    lines.append("")

    lines.append("FINAL TIME=")
    lines.append(f"\t{time_horizon}")
    lines.append("\t~\tTime")
    lines.append("\t~\t|")
    lines.append("")
    lines.append("INITIAL TIME=")
    lines.append("\t0")
    lines.append("\t~\tTime")
    lines.append("\t~\t|")
    lines.append("")
    lines.append("SAVEPER=")
    lines.append("\tTIME STEP")
    lines.append("\t~\tTime")
    lines.append("\t~\t|")
    lines.append("")
    lines.append("TIME STEP=")
    lines.append(f"\t{dt}")
    lines.append("\t~\tTime")
    lines.append("\t~\t|")
    lines.append("")

    # --- Sketch section ---
    lines.append("\\\\\\---/// Sketch information - do not modify anything except names")
    lines.append("V300  Do not put anything below this section - it will be ignored")
    lines.append("*View 1")
    lines.append("$-1--1--1,0,|12||-1--1--1|-1--1--1|-1--1--1|-1--1--1|-1--1--1|96,96,100,0")

    entity_id = 0
    id_map = {}  # variable_name -> entity_id

    # Assign layout coordinates
    # Stocks in left column, flows to their right, auxiliaries further right
    y_pos = 200
    stock_positions = {}
    flow_positions = {}
    aux_positions = {}

    for i, stock in enumerate(sfd.stocks):
        stock_positions[stock.name] = (200, y_pos + i * 150)
    for i, flow in enumerate(sfd.flows):
        flow_positions[flow.name] = (450, y_pos + i * 120)
    for i, aux in enumerate(sfd.auxiliaries):
        aux_positions[aux.name] = (650, y_pos + i * 100)

    # Emit stocks (type 10, shape 3 = box)
    for stock in sfd.stocks:
        entity_id += 1
        id_map[stock.name] = entity_id
        vname = _vensim_name(stock.name)
        x, y = stock_positions[stock.name]
        lines.append(
            f"10,{entity_id},{vname},{x},{y},40,20,3,3,0,0,-1,0,0,0,0,0,0,0,0,0"
        )

    # For each flow: emit a cloud (type 12), pipe arrows (type 1 with
    # bits 22), a valve (type 11), and the flow label (type 10, shape 40)
    for flow in sfd.flows:
        vname = _vensim_name(flow.name)
        fx, fy = flow_positions[flow.name]

        # Determine source and sink entity IDs
        src_id = id_map.get(flow.source.name) if flow.source and not flow.source.is_cloud else None
        snk_id = id_map.get(flow.sink.name) if flow.sink and not flow.sink.is_cloud else None

        # Cloud for the side that connects to environment
        cloud_id = None
        if src_id is None:
            # Source is cloud: place cloud to the left of valve
            entity_id += 1
            cloud_id = entity_id
            cx, cy = fx - 80, fy
            lines.append(f"12,{entity_id},48,{cx},{cy},10,8,0,3,0,0,-1,0,0,0,0,0,0,0,0,0")
            src_id = cloud_id

        if snk_id is None and cloud_id is None:
            # Sink is cloud: place cloud to the right of valve
            entity_id += 1
            cloud_id = entity_id
            cx, cy = fx + 80, fy
            lines.append(f"12,{entity_id},48,{cx},{cy},10,8,0,3,0,0,-1,0,0,0,0,0,0,0,0,0")
            snk_id = cloud_id
        elif snk_id is None:
            snk_id = cloud_id  # reuse same cloud for both sides if needed

        # Pipe arrow: source → valve (type 1, bits field includes 22 for pipe)
        entity_id += 1
        valve_arrow1_id = entity_id
        valve_id_ref = entity_id + 2  # will be the valve entity
        lines.append(
            f"1,{entity_id},{valve_id_ref},{snk_id},4,0,0,22,0,192,0,-1--1--1,,1|({fx},{fy})|"
        )

        # Pipe arrow: valve → sink
        entity_id += 1
        lines.append(
            f"1,{entity_id},{valve_id_ref},{src_id},100,0,0,22,0,192,0,-1--1--1,,1|({fx - 30},{fy})|"
        )

        # Valve (type 11)
        entity_id += 1
        valve_id = entity_id
        id_map[flow.name + "_valve"] = valve_id
        lines.append(
            f"11,{entity_id},0,{fx},{fy},8,6,33,3,0,0,1,0,0,0,0,0,0,0,0,0"
        )

        # Flow label (type 10, shape 40 = rate variable label)
        entity_id += 1
        id_map[flow.name] = entity_id
        # Also map the flow name to the valve for info dependency arrows
        id_map[flow.name + "_info_target"] = valve_id
        lines.append(
            f"10,{entity_id},{vname},{fx},{fy + 25},40,11,40,3,0,0,-1,0,0,0,0,0,0,0,0,0"
        )

    # Emit auxiliaries (type 10, shape 8 = circle/diamond)
    for aux in sfd.auxiliaries:
        entity_id += 1
        id_map[aux.name] = entity_id
        vname = _vensim_name(aux.name)
        x, y = aux_positions[aux.name]
        lines.append(
            f"10,{entity_id},{vname},{x},{y},40,11,8,3,0,0,0,0,0,0,0,0,0,0,0,0"
        )

    # Emit information dependency arrows (type 1, plain causal arrow)
    # For flow targets, connect to the valve (type 11), not the label
    for dep in sfd.dependencies:
        src_eid = id_map.get(dep.source.name)
        # Use valve entity for flow targets so Vensim renders them
        # as info arrows into the valve, not into the rate label
        tgt_eid = id_map.get(dep.target.name + "_info_target")
        if tgt_eid is None:
            tgt_eid = id_map.get(dep.target.name)
        if src_eid and tgt_eid:
            entity_id += 1
            lines.append(
                f"1,{entity_id},{src_eid},{tgt_eid},0,0,0,0,0,64,0,-1--1--1,,1|(0,0)|"
            )

    lines.append("///---\\\\\\")

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_sfd_parameters(
    sfd: SFD,
    output_path: str | Path,
) -> None:
    """Export SFD parameters to a readable text file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"SFD Parameters: {sfd.name}\n")
        f.write(f"Organization: {sfd.org}\n")
        f.write("=" * 60 + "\n\n")

        f.write("STOCKS:\n")
        for s in sfd.stocks:
            inflows = [fl.name for fl in sfd.get_stock_inflows(s)]
            outflows = [fl.name for fl in sfd.get_stock_outflows(s)]
            f.write(f"  {s.name}: initial={s.initial_value:.2f}\n")
            f.write(f"    inflows:  {inflows}\n")
            f.write(f"    outflows: {outflows}\n")

        f.write("\nFLOWS (with mapf source → sink):\n")
        for flow in sfd.flows:
            src = flow.source.name if flow.source else "?"
            snk = flow.sink.name if flow.sink else "?"
            f.write(f"  {flow.name}: {src} → {snk}\n")
            if flow.equation_params:
                p = flow.equation_params
                f.write(f"    intercept: {p.get('intercept', 0):.6f}\n")
                for name, lag, coef in zip(
                    p.get("dep_names", []),
                    p.get("dep_lags", []),
                    p.get("coefficients", []),
                ):
                    f.write(f"    + {coef:.6f} * {name} (lag={lag})\n")

        if sfd.auxiliaries:
            f.write("\nAUXILIARIES:\n")
            for a in sfd.auxiliaries:
                f.write(f"  {a.name}\n")

        f.write(f"\nINFORMATION DEPENDENCIES ({len(sfd.dependencies)}):\n")
        for d in sfd.dependencies:
            f.write(
                f"  {d.source.name} → {d.target.name}  "
                f"lag={d.lag}  corr={d.correlation:.4f}\n"
            )