"""Self-contained per-bag interactive INS/Visual comparison reports."""

from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np


_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@dataclass(frozen=True)
class InteractiveBagReportSummary:
    dataset: str
    bag_id: str
    relative_path: str
    common_frame_count: int
    candidate_ids: tuple[str, ...]


def _safe_name(value: str, field: str) -> str:
    text = str(value)
    if not _SAFE_NAME.fullmatch(text):
        raise ValueError(f"{field} is not a safe file-name component: {text}")
    return text


def write_interactive_bag_report(
    rows: Sequence[Mapping[str, object]],
    *,
    dataset: str,
    bag_id: str,
    output_path: Path,
    bag_identity_sha256: str | None = None,
    retention_by_candidate: Mapping[str, float] | None = None,
    tag0_map_origin_by_candidate: Mapping[str, Sequence[float]] | None = None,
) -> InteractiveBagReportSummary:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise RuntimeError("interactive bag reports require plotly") from error
    dataset_name = _safe_name(dataset, "dataset")
    bag_name = _safe_name(bag_id, "bag_id")
    values = [
        row
        for row in rows
        if str(row["dataset"]) == dataset_name and str(row["bag_id"]) == bag_name
    ]
    if not values:
        raise ValueError(f"interactive report has no rows for {dataset_name}/{bag_name}")
    numeric_fields = (
        "sample_index",
        "stamp_s",
        "elapsed_s",
        "camera_tag0_tz_m",
        "reference_x_m",
        "reference_y_m",
        "visual_x_m",
        "visual_y_m",
        "reference_x_local_m",
        "reference_y_local_m",
        "visual_x_local_m",
        "visual_y_local_m",
        "reference_yaw_display_deg",
        "visual_yaw_display_deg",
        "error_x_m",
        "error_y_m",
        "error_yaw_deg",
    )
    if any(
        not np.isfinite(float(row[field]))
        for row in values
        for field in numeric_fields
    ):
        raise ValueError("interactive report rows must contain only finite numbers")
    candidate_ids = tuple(sorted({str(row["candidate_id"]) for row in values}))
    by_candidate = {
        candidate_id: sorted(
            [row for row in values if row["candidate_id"] == candidate_id],
            key=lambda row: (float(row["stamp_s"]), str(row["frame_key"])),
        )
        for candidate_id in candidate_ids
    }
    frame_counts = {len(items) for items in by_candidate.values()}
    if len(frame_counts) != 1:
        raise ValueError("interactive candidate rows do not share a common frame set")
    default_candidate = (
        "safe_dynamic_rxy"
        if "safe_dynamic_rxy" in candidate_ids
        else ("full_static_rxy" if "full_static_rxy" in candidate_ids else candidate_ids[0])
    )
    reference = by_candidate[default_candidate]
    if tag0_map_origin_by_candidate is None:
        raise ValueError("Tag0 map origins are required for interactive reports")
    if set(tag0_map_origin_by_candidate) != set(candidate_ids):
        raise ValueError("Tag0 map origins must cover exactly all candidates")
    origins = {}
    for candidate_id in candidate_ids:
        value = np.asarray(
            tag0_map_origin_by_candidate[candidate_id], dtype=np.float64
        ).reshape(-1)
        if value.size < 2 or not np.all(np.isfinite(value[:2])):
            raise ValueError("Tag0 map origins must contain finite X/Y values")
        origins[candidate_id] = (float(value[0]), float(value[1]))
    if bag_identity_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", str(bag_identity_sha256)
    ):
        raise ValueError("bag_identity_sha256 must be a lowercase SHA256")
    retention = {
        str(key): float(value)
        for key, value in (retention_by_candidate or {}).items()
    }
    if any(
        not np.isfinite(value) or value < 0.0 or value > 1.0
        for value in retention.values()
    ):
        raise ValueError("candidate retention must be finite and within [0, 1]")
    first_yaw = float(reference[0]["reference_yaw_display_deg"])
    default_origin_x, default_origin_y = origins[default_candidate]
    candidate_payload = {
        candidate_id: [
            [
                float(row["elapsed_s"]),
                float(row["visual_x_m"]) - origins[candidate_id][0],
                float(row["visual_y_m"]) - origins[candidate_id][1],
                float(row["visual_yaw_display_deg"]) - first_yaw,
                float(row["error_x_m"]),
                float(row["error_y_m"]),
                float(row["error_yaw_deg"]),
                int(row["sample_index"]),
                float(row["camera_tag0_tz_m"]),
                str(row["structural_state"]),
                float(row["reference_x_m"]) - origins[candidate_id][0],
                float(row["reference_y_m"]) - origins[candidate_id][1],
                float(row["reference_yaw_display_deg"]) - first_yaw,
            ]
            for row in by_candidate[candidate_id]
        ]
        for candidate_id in candidate_ids
    }
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("INS / Visual XY trajectory", "X comparison", "Y comparison", "Yaw comparison"),
        specs=[[{}, {"secondary_y": True}], [{"secondary_y": True}, {"secondary_y": True}]],
    )
    def add(trace, row, col, *, secondary_y=False):
        location = {"row": row, "col": col}
        if secondary_y:
            location["secondary_y"] = True
        figure.add_trace(trace, **location)

    elapsed = [float(row["elapsed_s"]) for row in reference]
    custom = np.asarray(candidate_payload[default_candidate], dtype=object)
    add(
        go.Scattergl(
            x=[float(row["reference_x_m"]) - default_origin_x for row in reference],
            y=[float(row["reference_y_m"]) - default_origin_y for row in reference],
            mode="lines+markers",
            name="INS trajectory",
            customdata=custom,
            hovertemplate="INS x=%{x:.3f} m<br>y=%{y:.3f} m<br>elapsed=%{customdata[0]:.3f} s<br>sample=%{customdata[7]}<br>depth=%{customdata[8]:.3f} m<br>state=%{customdata[9]}<extra></extra>",
        ),
        1,
        1,
    )
    for axis, row_index, column_index in (("x", 1, 2), ("y", 2, 1)):
        add(
            go.Scattergl(
                x=elapsed,
                y=[
                    float(row[f"reference_{axis}_m"])
                    - (default_origin_x if axis == "x" else default_origin_y)
                    for row in reference
                ],
                mode="lines",
                name=f"INS {axis.upper()}",
                legendgroup=f"reference_{axis}",
                customdata=custom,
                hovertemplate=(
                    f"INS {axis.upper()}=%{{y:.3f}} m<br>elapsed=%{{x:.3f}} s"
                    "<br>sample=%{customdata[7]}<br>depth=%{customdata[8]:.3f} m"
                    "<br>state=%{customdata[9]}<extra></extra>"
                ),
            ),
            row_index,
            column_index,
        )
    add(
        go.Scattergl(
            x=elapsed,
            y=[float(row["reference_yaw_display_deg"]) - first_yaw for row in reference],
            mode="lines",
            name="INS yaw",
            legendgroup="reference_yaw",
            customdata=custom,
            hovertemplate="INS yaw=%{y:.3f}°<br>elapsed=%{x:.3f} s<br>sample=%{customdata[7]}<br>depth=%{customdata[8]:.3f} m<br>state=%{customdata[9]}<extra></extra>",
        ),
        2,
        2,
    )

    # Candidate traces are intentionally created only once.  Their data is
    # replaced lazily in JavaScript when the selector changes.  Hidden Plotly
    # traces still build hover/render indexes, which made large bags sluggish.
    candidate_trace_start = len(figure.data)
    empty: list[float] = []
    add(
        go.Scattergl(
            x=empty,
            y=empty,
            mode="lines+markers",
            name="Visual trajectory",
            hovertemplate="Visual x=%{x:.3f} m<br>y=%{y:.3f} m<br>elapsed=%{customdata[0]:.3f} s<br>sample=%{customdata[7]}<br>depth=%{customdata[8]:.3f} m<br>state=%{customdata[9]}<br>INS x=%{customdata[10]:.3f} m, y=%{customdata[11]:.3f} m<br>dx=%{customdata[4]:.3f} m<br>dy=%{customdata[5]:.3f} m<br>dyaw=%{customdata[6]:.3f}°<br>candidate=%{fullData.legendgroup}<extra></extra>",
        ),
        1,
        1,
    )
    add(
        go.Scattergl(
            x=empty,
            y=empty,
            mode="lines",
            name="Visual X",
            hovertemplate="Visual X=%{y:.3f} m<br>elapsed=%{x:.3f} s<br>sample=%{customdata[7]}<br>depth=%{customdata[8]:.3f} m<br>state=%{customdata[9]}<br>INS X=%{customdata[10]:.3f} m<br>dX=%{customdata[4]:.3f} m<br>candidate=%{fullData.legendgroup}<extra></extra>",
        ),
        1,
        2,
    )
    add(
        go.Scattergl(
            x=empty,
            y=empty,
            mode="lines+markers",
            name="X error",
            hovertemplate="X error=%{y:.3f} m<br>elapsed=%{x:.3f} s<br>sample=%{customdata[7]}<br>depth=%{customdata[8]:.3f} m<br>state=%{customdata[9]}<br>candidate=%{fullData.legendgroup}<extra></extra>",
        ),
        1,
        2,
        secondary_y=True,
    )
    add(
        go.Scattergl(
            x=empty,
            y=empty,
            mode="lines",
            name="Visual Y",
            hovertemplate="Visual Y=%{y:.3f} m<br>elapsed=%{x:.3f} s<br>sample=%{customdata[7]}<br>depth=%{customdata[8]:.3f} m<br>state=%{customdata[9]}<br>INS Y=%{customdata[11]:.3f} m<br>dY=%{customdata[5]:.3f} m<br>candidate=%{fullData.legendgroup}<extra></extra>",
        ),
        2,
        1,
    )
    add(
        go.Scattergl(
            x=empty,
            y=empty,
            mode="lines+markers",
            name="Y error",
            hovertemplate="Y error=%{y:.3f} m<br>elapsed=%{x:.3f} s<br>sample=%{customdata[7]}<br>depth=%{customdata[8]:.3f} m<br>state=%{customdata[9]}<br>candidate=%{fullData.legendgroup}<extra></extra>",
        ),
        2,
        1,
        secondary_y=True,
    )
    add(
        go.Scattergl(
            x=empty,
            y=empty,
            mode="lines",
            name="Visual yaw",
            hovertemplate="Visual yaw=%{y:.3f}°<br>elapsed=%{x:.3f} s<br>sample=%{customdata[7]}<br>depth=%{customdata[8]:.3f} m<br>state=%{customdata[9]}<br>INS yaw=%{customdata[12]:.3f}°<br>dyaw=%{customdata[6]:.3f}°<br>candidate=%{fullData.legendgroup}<extra></extra>",
        ),
        2,
        2,
    )
    add(
        go.Scattergl(
            x=empty,
            y=empty,
            mode="lines+markers",
            name="Yaw error",
            hovertemplate="Yaw error=%{y:.3f}°<br>elapsed=%{x:.3f} s<br>sample=%{customdata[7]}<br>depth=%{customdata[8]:.3f} m<br>state=%{customdata[9]}<br>candidate=%{fullData.legendgroup}<extra></extra>",
        ),
        2,
        2,
        secondary_y=True,
    )
    candidate_trace_indices = list(range(candidate_trace_start + 7))
    figure.update_layout(
        title=f"{dataset_name}/{bag_name} · {default_candidate}",
        height=950,
        hovermode="closest",
        legend={"groupclick": "toggleitem"},
    )
    figure.update_xaxes(title_text="map X relative to Tag0 origin (m)", row=1, col=1)
    figure.update_yaxes(title_text="map Y relative to Tag0 origin (m)", row=1, col=1)
    for row_index, column_index in ((1, 2), (2, 1), (2, 2)):
        figure.update_xaxes(title_text="elapsed source time (s)", row=row_index, col=column_index)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stamps = [float(row["stamp_s"]) for row in reference]
    depths = [float(row["camera_tag0_tz_m"]) for row in reference]
    metadata = (
        f"<p><strong>Dataset:</strong> {html.escape(dataset_name)} &nbsp; "
        f"<strong>Bag:</strong> {html.escape(bag_name)} &nbsp; "
        f"<strong>Common frames:</strong> {len(reference)} &nbsp; "
        f"<strong>Source time:</strong> {min(stamps):.6f}–{max(stamps):.6f} s &nbsp; "
        f"<strong>Depth:</strong> {min(depths):.3f}–{max(depths):.3f} m</p>"
        "<p><strong>Position origin:</strong> candidate T_map_tag0 translation; "
        "map X/Y axis directions are retained (translation only, no rotation).</p>"
    )
    if bag_identity_sha256 is not None:
        metadata += (
            "<p><strong>Bag identity SHA256:</strong> "
            f"<code>{html.escape(str(bag_identity_sha256))}</code></p>"
        )
    if retention:
        metadata += "<p><strong>Candidate retention:</strong> " + ", ".join(
            f"{html.escape(candidate_id)}={value:.1%}"
            for candidate_id, value in sorted(retention.items())
        ) + "</p>"
    panel_contract = (
        '<div hidden id="report-panel-contract">'
        '<span id="trajectory_xy"></span><span id="comparison_x"></span>'
        '<span id="comparison_y"></span><span id="comparison_yaw"></span></div>'
    )
    candidate_json = json.dumps(
        candidate_payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    origin_json = json.dumps(
        origins,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    plot_id = f"fae_plot_{dataset_name}_{bag_name}"
    selector_id = f"candidate_selector_{dataset_name}_{bag_name}"
    status_id = f"candidate_status_{dataset_name}_{bag_name}"
    axis_lock_id = f"xy_axis_lock_{dataset_name}_{bag_name}"
    options = "".join(
        '<option value="{}"{}>{}</option>'.format(
            html.escape(candidate_id, quote=True),
            " selected" if candidate_id == default_candidate else "",
            html.escape(candidate_id),
        )
        for candidate_id in candidate_ids
    )
    selector = (
        f'<p><label for="{selector_id}"><strong>Candidate:</strong></label> '
        f'<select id="{selector_id}">{options}</select> '
        f'<span id="{status_id}">loading…</span> &nbsp; '
        f'<label for="{axis_lock_id}">'
        f'<input type="checkbox" id="{axis_lock_id}"> XY 等比例</label></p>'
    )
    plot = figure.to_html(
        include_plotlyjs="inline",
        full_html=False,
        div_id=plot_id,
    )
    lazy_script = f"""
<script id="lazy-candidate-renderer">
(() => {{
  const plot = document.getElementById({json.dumps(plot_id)});
  const selector = document.getElementById({json.dumps(selector_id)});
  const status = document.getElementById({json.dumps(status_id)});
  const axisLock = document.getElementById({json.dumps(axis_lock_id)});
  const candidates = {candidate_json};
  const tag0MapOrigins = {origin_json};
  const candidateTraceIndices = {json.dumps(candidate_trace_indices)};
  const dataset = {json.dumps(dataset_name)};
  const bag = {json.dumps(bag_name)};
  function column(rows, index) {{ return rows.map(row => row[index]); }}
  function updateCandidate(candidateId) {{
    const rows = candidates[candidateId];
    if (!rows) {{ throw new Error(`unknown candidate: ${{candidateId}}`); }}
    const elapsed = column(rows, 0);
    const visualX = column(rows, 1);
    const visualY = column(rows, 2);
    const visualYaw = column(rows, 3);
    const errorX = column(rows, 4);
    const errorY = column(rows, 5);
    const errorYaw = column(rows, 6);
    const referenceX = column(rows, 10);
    const referenceY = column(rows, 11);
    const referenceYaw = column(rows, 12);
    const custom = candidateTraceIndices.map(() => rows);
    selector.disabled = true;
    status.textContent = `rendering ${{candidateId}}…`;
    const dataUpdate = {{
      x: [
        referenceX, elapsed, elapsed, elapsed,
        visualX, elapsed, elapsed, elapsed, elapsed, elapsed, elapsed
      ],
      y: [
        referenceY, referenceX, referenceY, referenceYaw,
        visualY, visualX, errorX, visualY, errorY, visualYaw, errorYaw
      ],
      customdata: custom,
      name: [
        'INS trajectory', 'INS X', 'INS Y', 'INS yaw',
        `Visual trajectory · ${{candidateId}}`,
        `Visual X · ${{candidateId}}`, `X error · ${{candidateId}}`,
        `Visual Y · ${{candidateId}}`, `Y error · ${{candidateId}}`,
        `Visual yaw · ${{candidateId}}`, `Yaw error · ${{candidateId}}`
      ],
      legendgroup: [
        'reference_trajectory', 'reference_x', 'reference_y', 'reference_yaw',
        candidateId, candidateId, candidateId, candidateId,
        candidateId, candidateId, candidateId
      ]
    }};
    const layoutUpdate = {{'title.text': `${{dataset}}/${{bag}} · ${{candidateId}}`}};
    for (let index = 1; index <= 4; index += 1) {{
      layoutUpdate[index === 1 ? 'xaxis.autorange' : `xaxis${{index}}.autorange`] = true;
    }}
    for (let index = 1; index <= 7; index += 1) {{
      layoutUpdate[index === 1 ? 'yaxis.autorange' : `yaxis${{index}}.autorange`] = true;
    }}
    return Plotly.update(plot, dataUpdate, layoutUpdate, candidateTraceIndices)
      .then(() => {{
        const origin = tag0MapOrigins[candidateId];
        status.textContent = `${{rows.length}} frames · Tag0 map origin=(${{origin[0].toFixed(3)}}, ${{origin[1].toFixed(3)}}) m`;
      }})
      .catch(error => {{
        status.textContent = `render failed: ${{error.message}}`;
        throw error;
      }})
      .finally(() => {{ selector.disabled = false; }});
  }}
  selector.addEventListener('change', () => updateCandidate(selector.value));
  axisLock.addEventListener('change', () => {{
    Plotly.relayout(plot, {{
      'yaxis.scaleanchor': axisLock.checked ? 'x' : null,
      'yaxis.scaleratio': axisLock.checked ? 1 : null,
      'xaxis.autorange': true,
      'yaxis.autorange': true
    }});
  }});
  window.requestAnimationFrame(() => updateCandidate({json.dumps(default_candidate)}));
}})();
</script>"""
    document = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(dataset_name)}/{html.escape(bag_name)}</title></head>"
        f"<body>{metadata}{selector}{panel_contract}{plot}{lazy_script}</body></html>"
    )
    destination.write_text(document, encoding="utf-8")
    return InteractiveBagReportSummary(
        dataset=dataset_name,
        bag_id=bag_name,
        relative_path=f"{dataset_name}/{destination.name}",
        common_frame_count=next(iter(frame_counts)),
        candidate_ids=candidate_ids,
    )


def write_interactive_bag_index(
    reports: Sequence[InteractiveBagReportSummary],
    output_path: Path,
) -> Path:
    rows = "\n".join(
        "<li><a href=\"{}\">{}/{}</a> — {} common frames; candidates: {}</li>".format(
            html.escape(item.relative_path, quote=True),
            html.escape(item.dataset),
            html.escape(item.bag_id),
            item.common_frame_count,
            html.escape(", ".join(item.candidate_ids)),
        )
        for item in sorted(reports, key=lambda value: (value.dataset, value.bag_id))
    )
    payload = (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Interactive bag reports</title></head>"
        f"<body><h1>Interactive bag reports</h1><ul>{rows}</ul></body></html>"
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8")
    return destination
