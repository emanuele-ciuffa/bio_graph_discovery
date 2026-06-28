# pip install dash dash-cytoscape neo4j pandas

import re
from urllib.parse import parse_qs
from dash import Dash, html, dcc, Input, Output
import dash_cytoscape as cyto
from neo4j import GraphDatabase
from src.utils.config_handler import Config_handler
from src.utils import db_info_utils

# Reading organism from DB
selected_organism = db_info_utils.get_db_organism()


# Instantiate dashboard config handler
dashboard_config_handler = Config_handler("config-dashboard.yml")

# Reading title from dashboard config file
TITLE = dashboard_config_handler.read_property("link_prediction.first_level_view.title")

# Reading port from dashboard config file
PORT = dashboard_config_handler.read_property("link_prediction.second_level_view.port")

# Instatiate config handler for retrieving neo4j parameters
config_handler_neo4j = Config_handler("config-neo4j.yml")

# Reading references to connect to neo4j DB
URI = config_handler_neo4j.read_property("neo4j.uri")

USER = config_handler_neo4j.read_property("neo4j.user")
PASSWORD = config_handler_neo4j.read_property("neo4j.password")

# --- 1. NEO4J CONNECTION & DATA FETCHING ---
AUTH = (USER, PASSWORD)

# UNCHANGED CYPHER QUERY AS REQUESTED
cypher_query = """
MATCH (n)
WHERE n.id IN [
  $source_id,
  $target_id
]

WITH collect(n) AS targetNodes

// Existing real relationships between selected nodes
OPTIONAL MATCH (n1)-[r]->(n2)
WHERE n1 IN targetNodes
  AND n2 IN targetNodes

WITH targetNodes,
     collect({
       source: n1,
       rel: r,
       target: n2
     }) AS baseRels

// Predicted link nodes
MATCH (a)
WHERE a.id = $source_id

MATCH (b)
WHERE b.id = $target_id

// Create virtual relationship
WITH targetNodes,
     baseRels,
     a,
     b,
     apoc.create.vRelationship(
       a,
       "PREDICTED_LINK",
       {score: $score},
       b
     ) AS predictedRel

// Expand reactions if present
OPTIONAL MATCH (reactionNode)
WHERE reactionNode IN targetNodes
  AND (
       reactionNode:reaction
       OR "reaction" IN labels(reactionNode)
  )

OPTIONAL MATCH (reactionNode)-[r2]->(extraNode)
WHERE r2.relationship IN ["HAS_AGENT", "HAS_TARGET"]

RETURN
  targetNodes,
  baseRels,
  predictedRel,
  collect(DISTINCT {
    reactionNode: reactionNode,
    rel: r2,
    extraNode: extraNode
  }) AS reactionExpansions
"""


def fetch_graph_elements(source_id, target_id, score_str):
    """Fetches graph data dynamically using IDs supplied by request parameters"""
    nodes = {}
    edges = {}

    def clean_string(val):
        if val is None:
            return ""
        return str(val).strip("'\"")

    def add_node_to_dict(node):
        if node is None:
            return
        node_type = (
            list(node.labels)[0].lower()
            if node.labels
            else "unknown"
        )

        raw_name = node.get("name") or node.get("id") or "Unknown"
        display_label = clean_string(raw_name)

        if len(display_label) > 25 and "_" in display_label:
            display_label = display_label[:15] + "..."

        nodes[node.element_id] = {
            "id": node.element_id,
            "url_id": clean_string(node.get("id")),
            "label": display_label,
            "type": node_type,
            "name": clean_string(node.get("name", "N/A")),
            "action": clean_string(node.get("action", "")),
            "gene_ID": clean_string(node.get("gene_ID", "")),
            "chem_ID": clean_string(node.get("chem_ID", "")),
        }

    try:
        score_val = float(score_str)
    except (ValueError, TypeError):
        score_val = 0.0

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            result = session.run(
                cypher_query,
                parameters={
                    "source_id": source_id,
                    "target_id": target_id,
                    "score": score_val
                }
            )

            for record in result:
                for node in (record["targetNodes"] or []):
                    add_node_to_dict(node)

                for item in (record["baseRels"] or []):
                    n1, r, n2 = item.get("source"), item.get("rel"), item.get("target")
                    if n1 and r and n2:
                        add_node_to_dict(n1)
                        add_node_to_dict(n2)
                        edge_key = f"{n1.element_id}->{n2.element_id}::{r.type}"
                        edges[edge_key] = {
                            "id": edge_key,
                            "source": n1.element_id,
                            "target": n2.element_id,
                            "label": r.type,
                        }

                v_rel = record["predictedRel"]
                if v_rel:
                    add_node_to_dict(v_rel.start_node)
                    add_node_to_dict(v_rel.end_node)

                    src_id = v_rel.start_node.element_id
                    tgt_id = v_rel.end_node.element_id
                    edge_key = f"{src_id}->{tgt_id}::{v_rel.type}"
                    edges[edge_key] = {
                        "id": edge_key,
                        "source": src_id,
                        "target": tgt_id,
                        "label": v_rel.type,
                        "score": score_val
                    }

                for item in (record["reactionExpansions"] or []):
                    r_node = item.get("reactionNode")
                    r2 = item.get("rel")
                    ex_node = item.get("extraNode")

                    if r_node and r2 and ex_node:
                        add_node_to_dict(r_node)
                        add_node_to_dict(ex_node)

                        rel_label = r2.get("relationship", r2.type)
                        edge_key = f"{r_node.element_id}->{ex_node.element_id}::{rel_label}"
                        edges[edge_key] = {
                            "id": edge_key,
                            "source": r_node.element_id,
                            "target": ex_node.element_id,
                            "label": rel_label,
                        }

    cytoscape_elements = []
    for n_id, n_data in nodes.items():
        cytoscape_elements.append({"data": n_data})
    for e_key, e_data in edges.items():
        cytoscape_elements.append({"data": e_data})

    return cytoscape_elements


def build_html_table(elements, type_colors):
    node_elements = [el["data"] for el in elements if "source" not in el["data"]]
    headers = ["Name", "Type", "Chem_ID", "Gene_ID", "Action"]

    table_header = html.Thead(
        html.Tr(
            [
                html.Th(h, style={
                    "borderBottom": "2px solid #ddd", "padding": "12px",
                    "textAlign": "left", "backgroundColor": "#f4f4f4", "fontFamily": "Arial"
                }) for h in headers
            ]
        )
    )

    table_header = html.Thead(
        html.Tr(
            [
                html.Th(h, style={
                    "borderBottom": "2px solid #ddd", "padding": "12px",
                    "textAlign": "left", "backgroundColor": "#f4f4f4", "fontFamily": "Arial"
                }) for h in headers
            ]
        )
    )

    table_rows = []
    for node in node_elements:
        name = node.get("name", "N/A")
        raw_type = str(node.get("type", "unknown")).lower()
        ntype = raw_type.capitalize()
        chem_id = node.get("chem_ID") or ""
        gene_id = node.get("gene_ID") or ""
        action = node.get("action") or ""
        row_bg = type_colors.get(raw_type, "#ffffff")

        table_rows.append(
            html.Tr(
                [
                    html.Td(name, style={"padding": "10px", "borderBottom": "1px solid #ddd"}),
                    html.Td(ntype, style={"padding": "10px", "borderBottom": "1px solid #ddd", "fontWeight": "bold"}),
                    html.Td(chem_id, style={"padding": "10px", "borderBottom": "1px solid #ddd"}),
                    html.Td(gene_id, style={"padding": "10px", "borderBottom": "1px solid #ddd"}),
                    html.Td(action, style={"padding": "10px", "borderBottom": "1px solid #ddd"}),
                ],
                style={"backgroundColor": row_bg, "fontFamily": "Arial", "fontSize": "14px", "color": "#222222"},
            )
        )

    return html.Table([table_header, html.Tbody(table_rows)],
                      style={"width": "100%", "borderCollapse": "collapse", "marginTop": "20px"})


# --- DASH APPLICATION SETUP WITH EXCEPTION SUPPRESSION ---
app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])


# --- CALLBACK TO PARSE REQUEST PARAMS AND GENERATE THE LAYOUT ---
@app.callback(
    Output("page-content", "children"),
    Input("url", "search")
)
def update_layout_from_url(search_params):
    source_id = "chemical_109855_474261"
    target_id = "main-action_102673"
    score_val = "N/A"

    if search_params:
        parsed = parse_qs(search_params.lstrip("?"))
        if "source_id" in parsed:
            source_id = parsed["source_id"][0]
        if "target_id" in parsed:
            target_id = parsed["target_id"][0]
        if "score" in parsed:
            score_val = parsed["score"][0]
        elif "probability" in parsed:
            score_val = parsed["probability"][0]

    graph_elements = fetch_graph_elements(source_id, target_id, score_val)

    type_colors = {
        "reaction": "#FFEAD2",
        "protein": "#E2F0D9",
        "chemical": "#D9E1F2"
    }

    # Extract target node metadata to determine dynamic backgrounds and NAMES safely
    source_type = "unknown"
    target_type = "unknown"
    source_name = "Unknown Source"
    target_name = "Unknown Target"

    for element in graph_elements:
        data = element["data"]
        if "source" in data:
            continue
        if data.get("url_id") == source_id:
            source_type = data.get("type", "unknown").lower()
            source_name = data.get("name") or source_id
        if data.get("url_id") == target_id:
            target_type = data.get("type", "unknown").lower()
            target_name = data.get("name") or target_id

    source_bg = type_colors.get(source_type, "#ffffff")
    target_bg = type_colors.get(target_type, "#ffffff")

    base_card_style = {
        "flex": "1",
        "minWidth": "200px",
        "maxWidth": "320px",
        "padding": "12px",
        "border": "1px solid #e2e8f0",
        "borderRadius": "6px",
        "textAlign": "center",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.05)"
    }

    return html.Div(
        [
            html.H1(TITLE,
                    style={"textAlign": "center", "fontFamily": "Arial", "marginBottom": "25px"}),

            # --- HORIZONTAL FLEX CONTAINER FOR SOURCE, SCORE, TARGET, AND ORGANISM CARD BLOCKS ---
            html.Div(
                [
                    # Left KPI Card: Displays Name, embeds actual source_id as custom HTML data-attribute
                    html.Div(
                        [
                            html.Div(f"SOURCE NODE ({source_type.upper()})",
                                     style={"fontSize": "11px", "fontWeight": "bold", "color": "#7f8c8d",
                                            "letterSpacing": "1.2px"}),
                            html.Div(source_name, title=f"ID: {source_id}",
                                     style={"fontSize": "18px", "fontWeight": "bold", "color": "#2c3e50",
                                            "marginTop": "8px", "wordBreak": "break-word"})
                        ],
                        id="kpi-source-card",
                        style={**base_card_style, "backgroundColor": source_bg},
                        **{"data-node-id": source_id}  # Placed inside HTML attributes
                    ),

                    # Center-Left KPI Card: Link Score
                    html.Div(
                        [
                            html.Div("LINK PREDICTION SCORE",
                                     style={"fontSize": "11px", "fontWeight": "bold", "color": "#9b59b6",
                                            "letterSpacing": "1.2px"}),
                            html.Div(score_val,
                                     style={"fontSize": "28px", "fontWeight": "bold", "color": "#8e44ad",
                                            "marginTop": "2px"})
                        ],
                        style={**base_card_style, "backgroundColor": "#fbf4ff", "borderColor": "#e8d5f5"}
                    ),

                    # Center-Right KPI Card: Displays Name, embeds actual target_id as custom HTML data-attribute
                    html.Div(
                        [
                            html.Div(f"TARGET NODE ({target_type.upper()})",
                                     style={"fontSize": "11px", "fontWeight": "bold", "color": "#7f8c8d",
                                            "letterSpacing": "1.2px"}),
                            html.Div(target_name, title=f"ID: {target_id}",
                                     style={"fontSize": "18px", "fontWeight": "bold", "color": "#2c3e50",
                                            "marginTop": "8px", "wordBreak": "break-word"})
                        ],
                        id="kpi-target-card",
                        style={**base_card_style, "backgroundColor": target_bg},
                        **{"data-node-id": target_id}  # Placed inside HTML attributes
                    ),

                    # Right KPI Card: Selected Organism
                    html.Div(
                        [
                            html.Div("ORGANISM",
                                     style={"fontSize": "11px", "fontWeight": "bold", "color": "#e67e22",
                                            "letterSpacing": "1.2px"}),
                            html.Div(str(selected_organism),
                                     style={"fontSize": "18px", "fontWeight": "bold", "color": "#2c3e50",
                                            "marginTop": "8px", "wordBreak": "break-word"})
                        ],
                        style={**base_card_style, "backgroundColor": "#fef9f3", "borderColor": "#fbe3cc"}
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "center",
                    "alignItems": "stretch",
                    "gap": "20px",
                    "maxWidth": "1200px",
                    "margin": "0 auto 35px auto",
                    "fontFamily": "Arial"
                }
            ),

            html.Div(
                id="cytoscape-wrapper",
                title="",
                children=[
                    cyto.Cytoscape(
                        id="cytoscape-neo4j",
                        elements=graph_elements,
                        layout={
                            "name": "cose",
                            "idealEdgeLength": 120,
                            "nodeOverlap": 30,
                            "refresh": 20,
                            "fit": True,
                            "padding": 40,
                            "randomize": False,
                            "componentSpacing": 120,
                            "nodeRepulsion": 500000,
                            "edgeElasticity": 80,
                            "nestingFactor": 5,
                        },
                        style={"width": "100%", "height": "600px", "border": "1px solid #ccc"},
                        stylesheet=[
                            {
                                "selector": "node",
                                "style": {
                                    "label": "data(label)",
                                    "background-color": "#777777",
                                    "color": "#222",
                                    "font-size": "11px",
                                    "text-valign": "top",
                                    "text-halign": "center",
                                    "text-margin-y": "-6px",
                                    "width": "32px",
                                    "height": "32px"
                                },
                            },
                            {"selector": "node[type = 'chemical']", "style": {"background-color": "#0074D9"}},
                            {"selector": "node[type = 'protein']", "style": {"background-color": "#2ECC40"}},
                            {"selector": "node[type = 'reaction']", "style": {"background-color": "#FF851B"}},
                            {
                                "selector": "edge",
                                "style": {
                                    "label": "data(label)",
                                    "line-color": "#FF4136",
                                    "target-arrow-color": "#FF4136",
                                    "target-arrow-shape": "triangle",
                                    "curve-style": "bezier",
                                    "control-point-step-size": "40px",
                                    "text-rotation": "autorotate",
                                    "font-size": "10px",
                                    "text-margin-y": "-10px",
                                },
                            },
                            {
                                "selector": "edge[label = 'PREDICTED_LINK']",
                                "style": {
                                    "line-style": "dashed",
                                    "line-color": "#B10DC9",
                                    "target-arrow-color": "#B10DC9"
                                },
                            },
                            {
                                "selector": "edge[label = 'HAS_AGENT'], edge[label = 'HAS_TARGET']",
                                "style": {"line-color": "#999999", "target-arrow-color": "#999999"},
                            },
                        ],
                    )
                ],
            ),
            html.Div(
                [
                    html.H3("Represented elements", style={"fontFamily": "Arial", "marginTop": "40px"}),
                    build_html_table(graph_elements, type_colors),
                ],
                style={"width": "100%", "padding": "0 10px", "boxSizing": "border-box"},
            ),
        ]
    )


# --- COMBINED CALLBACK FOR TOOLTIPS (NODES & SPECIFIC EDGES) ---
@app.callback(
    Output("cytoscape-wrapper", "title"),
    Input("cytoscape-neo4j", "mouseoverNodeData"),
    Input("cytoscape-neo4j", "mouseoverEdgeData"),
)
def display_hover_data(node_data, edge_data):
    from dash import callback_context
    triggered_id = [p['prop_id'] for p in callback_context.triggered]

    if not triggered_id:
        return ""

    if "mouseoverEdgeData" in triggered_id[0] and edge_data:
        if edge_data.get("label") == "PREDICTED_LINK":
            score = edge_data.get("score", "N/A")
            return f"Relationship: PREDICTED_LINK\nScore: {score}"
        return ""

    if "mouseoverNodeData" in triggered_id[0] and node_data:
        tooltip_lines = [
            f"Name: {node_data.get('name', 'N/A')}",
            f"Type: {node_data.get('type', 'Unknown').capitalize()}",
        ]
        if node_data.get("gene_ID"):
            tooltip_lines.append(f"Gene ID: {node_data['gene_ID']}")
        if node_data.get("chem_ID"):
            tooltip_lines.append(f"Chemical ID: {node_data['chem_ID']}")
        if node_data.get("action"):
            tooltip_lines.append(f"Action: {node_data['action']}")
        return "\n".join(tooltip_lines)

    return ""


if __name__ == "__main__":
    app.run(debug=False, port=PORT)