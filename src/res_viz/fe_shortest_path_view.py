# pip install dash dash-cytoscape neo4j pandas

from dash import Dash, html, dcc, Input, Output
import dash_cytoscape as cyto
from neo4j import GraphDatabase
import urllib.parse
from src.utils.config_handler import Config_handler
from src.utils import db_info_utils

# Reading organism from DB
selected_organism = db_info_utils.get_db_organism()

# Instantiate dashboard config handler
dashboard_config_handler = Config_handler("config-dashboard.yml")

TITLE = dashboard_config_handler.read_property("shortest_path.second_level_view.title")

# Reading port from dashboard config file
PORT = dashboard_config_handler.read_property("shortest_path.second_level_view.port")

# Instatiate config handler for retrieving neo4j parameters
config_handler_neo4j = Config_handler("config-neo4j.yml")

# Reading references to connect to neo4j DB
URI = config_handler_neo4j.read_property("neo4j.uri")

USER = config_handler_neo4j.read_property("neo4j.user")
PASSWORD = config_handler_neo4j.read_property("neo4j.password")

# --- 1. NEO4J CONNECTION & DATA FETCHING ---
AUTH = (USER, PASSWORD)


def fetch_node_name_by_id(session, node_id):
    """Helper method to quickly retrieve a specific node's name by its ID."""
    if not node_id or node_id == "N/A":
        return "N/A"
    query = "MATCH (n) WHERE n.id = $node_id RETURN n.name AS name LIMIT 1"
    result = session.run(query, node_id=node_id)
    record = result.single()
    return record["name"] if record and record["name"] else node_id


def fetch_graph_elements_and_entities(target_ids, source_id, target_id):
    # Determine which cypher query to use based on target_ids values
    use_alternative_query = False
    if target_ids:
        for idx in target_ids:
            clean_id = str(idx).strip()
            # print(f"clean_id: {clean_id}")
            if clean_id.startswith("reaction_") or clean_id.startswith("n_reaction_") or clean_id.startswith(
                    "main-action_"):
                use_alternative_query = True
                break

    if use_alternative_query:
        # More complex query used to display ontological relationships when at least a reaction is present
        cypher_query = """
               MATCH (n)
               WHERE n.id IN $id_list

               WITH collect(n) AS targetNodes

               MATCH (n1)-[r]->(n2)
               WHERE n1 IN targetNodes AND n2 IN targetNodes
               WITH targetNodes, collect({source: n1, rel: r, target: n2}) AS baseRels

               UNWIND targetNodes AS reactionNode
               WITH reactionNode, targetNodes, baseRels
               WHERE reactionNode:reaction OR "reaction" IN labels(reactionNode)

               MATCH (reactionNode)-[r2]->(extraNode)
               WHERE r2.relationship IN ["HAS_AGENT", "HAS_TARGET"]

               RETURN
                 baseRels,
                 reactionNode,
                 r2,
                 extraNode
               """
    else:
        # Simplest query to display direct functional relationships (no reaction nodes are present)
        cypher_query = """
                        MATCH (n)
                        WHERE n.id IN $id_list
                        WITH collect(n) AS targetNodes              

                        // Find relationships where BOTH ends are in your target list
                        MATCH (n1)-[r]->(n2)
                        WHERE n1 IN targetNodes AND n2 IN targetNodes
                        RETURN n1, r, n2
                        """

    nodes = {}
    edges = {}  # Will store unique (source, target, label) combinations
    source_entity_name = "N/A"
    target_entity_name = "N/A"

    # Identify what the business ID of the last node should be
    last_target_id = target_ids[-1] if target_ids else None

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            # 1. Fetch resolved names for source and target KPIs
            if source_id and source_id != "N/A":
                source_entity_name = fetch_node_name_by_id(session, source_id)
            if target_id and target_id != "N/A":
                target_entity_name = fetch_node_name_by_id(session, target_id)

            # 2. Extract main network subgraph components
            result = session.run(cypher_query, id_list=target_ids if target_ids else [])

            for record in result:
                if not use_alternative_query:
                    # --- Parse fields for the Direct Functional Relationships query ---
                    n1, r, n2 = record["n1"], record["r"], record["n2"]
                    for node in [n1, n2]:
                        node_type = (
                            list(node.labels)[0].lower()
                            if node.labels
                            else "unknown"
                        )
                        nodes[node.element_id] = {
                            "id": node.element_id,
                            "business_id": node.get("id", node.element_id),
                            "label": node.get("name", node.get("id")),
                            "type": node_type,
                            "name": node.get("name", "N/A"),
                            "action": node.get("action", ""),
                            "gene_ID": node.get("gene_ID", ""),
                            "chem_ID": node.get("chem_ID", ""),
                        }

                    edge_key = f"{n1.element_id}->{n2.element_id}::{r.type}"

                    # Check if this edge links into the final destination node path
                    is_last_edge = (node_to_check := nodes.get(n2.element_id)) and node_to_check.get(
                        "business_id") == last_target_id

                    edges[edge_key] = {
                        "id": edge_key,
                        "source": n1.element_id,
                        "target": n2.element_id,
                        "label": r.type,
                        "is_last": is_last_edge
                    }
                else:
                    # --- Parse fields for the Complex Ontological Relationships query ---
                    for item in record["baseRels"]:
                        n1, r, n2 = item["source"], item["rel"], item["target"]
                        for node in [n1, n2]:
                            node_type = (
                                list(node.labels)[0].lower()
                                if node.labels
                                else "unknown"
                            )
                            nodes[node.element_id] = {
                                "id": node.element_id,
                                "business_id": node.get("id", node.element_id),
                                "label": node.get("name", node.get("id")),
                                "type": node_type,
                                "name": node.get("name", "N/A"),
                                "action": node.get("action", ""),
                                "gene_ID": node.get("gene_ID", ""),
                                "chem_ID": node.get("chem_ID", ""),
                            }

                        edge_key = f"{n1.element_id}->{n2.element_id}::{r.type}"
                        is_last_edge = (node_to_check := nodes.get(n2.element_id)) and node_to_check.get(
                            "business_id") == last_target_id

                        edges[edge_key] = {
                            "id": edge_key,
                            "source": n1.element_id,
                            "target": n2.element_id,
                            "label": r.type,
                            "is_last": is_last_edge
                        }

                    rNode = record["reactionNode"]
                    exNode = record["extraNode"]
                    r2 = record["r2"]

                    if rNode and exNode and r2:
                        for node in [rNode, exNode]:
                            node_type = (
                                list(node.labels)[0].lower()
                                if node.labels
                                else "unknown"
                            )
                            nodes[node.element_id] = {
                                "id": node.element_id,
                                "business_id": node.get("id", node.element_id),
                                "label": node.get("name", node.get("id")),
                                "type": node_type,
                                "name": node.get("name", "N/A"),
                                "action": node.get("action", ""),
                                "gene_ID": node.get("gene_ID", ""),
                                "chem_ID": node.get("chem_ID", ""),
                            }

                        rel_label = r2.get("relationship", r2.type)
                        edge_key = f"{rNode.element_id}->{exNode.element_id}::{rel_label}"
                        is_last_edge = (node_to_check := nodes.get(exNode.element_id)) and node_to_check.get(
                            "business_id") == last_target_id

                        edges[edge_key] = {
                            "id": edge_key,
                            "source": rNode.element_id,
                            "target": exNode.element_id,
                            "label": rel_label,
                            "is_last": is_last_edge
                        }

    # --- FILTER OUT REVERSE-DIRECTION EDGES BASED ON PATH ORDER ---
    path_order = {b_id: idx for idx, b_id in enumerate(target_ids)} if target_ids else {}

    cytoscape_elements = []

    # Map out and append nodes
    for n_id, n_data in nodes.items():
        cytoscape_elements.append({"data": n_data})

    # Filter and append edges conditionally
    for e_key, e_data in edges.items():
        source_node = nodes.get(e_data["source"], {})
        target_node = nodes.get(e_data["target"], {})

        source_biz_id = source_node.get("business_id")
        target_biz_id = target_node.get("business_id")

        source_pos = path_order.get(source_biz_id)
        target_pos = path_order.get(target_biz_id)

        # Eliminate backwards or self-loop paths entirely
        if source_pos is not None and target_pos is not None:
            if source_pos >= target_pos:
                continue

        cytoscape_elements.append({"data": e_data})

    return cytoscape_elements, source_entity_name, target_entity_name


# --- HELPER FUNCTION TO GENERATE THE TYPE-COLORED HTML TABLE ---
def build_html_table(elements, target_ids=None):
    node_elements = [el["data"] for el in elements if "source" not in el["data"]]

    if target_ids:
        id_order = {node_id: idx for idx, node_id in enumerate(target_ids)}
        node_elements.sort(key=lambda node: id_order.get(node.get("business_id"), float('inf')))

    headers = ["Name", "Type", "Chem_ID", "Gene_ID", "Action"]

    table_header = html.Thead(
        html.Tr(
            [
                html.Th(
                    h,
                    style={
                        "borderBottom": "2px solid #ddd",
                        "padding": "12px",
                        "textAlign": "left",
                        "backgroundColor": "#f4f4f4",
                        "fontFamily": "Arial",
                    },
                )
                for h in headers
            ]
        )
    )

    type_colors = {
        "reaction": "#FFEAD2",
        "protein": "#E2F0D9",
        "chemical": "#D9E1F2",
    }

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
                style={
                    "backgroundColor": row_bg,
                    "fontFamily": "Arial",
                    "fontSize": "14px",
                    "color": "#222222"
                },
            )
        )

    return html.Table(
        [table_header, html.Tbody(table_rows)],
        style={"width": "100%", "borderCollapse": "collapse", "marginTop": "20px"},
    )


# --- 2. DASH APPLICATION LAYOUT ---
app = Dash(__name__)

kpi_box_style = {
    "padding": "10px 20px",
    "border": "1px solid #dcdde1",
    "borderRadius": "6px",
    "backgroundColor": "#f5f6fa",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
    "textAlign": "center",
    "minWidth": "180px",
    "maxWidth": "320px",
    "fontFamily": "Arial",
    "margin": "0 10px"
}

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(
            [
                html.H1(
                    TITLE,
                    style={"margin": "0 0 20px 0", "fontFamily": "Arial", "textAlign": "center"},
                ),
                html.Div(
                    [
                        html.Div(id="kpi-source-id", style=kpi_box_style),
                        html.Div(id="kpi-target-id", style=kpi_box_style),
                        html.Div(id="kpi-hops", style=kpi_box_style),
                        html.Div(
                            [
                                html.Div("ORGANISM",
                                         style={"fontSize": "11px", "fontWeight": "bold", "color": "#e67e22",
                                                "letterSpacing": "1.2px"}),
                                html.Div(str(selected_organism),
                                         style={"fontSize": "16px", "fontWeight": "bold", "color": "#2c3e50",
                                                "marginTop": "4px"})
                            ],
                            style={**kpi_box_style, "backgroundColor": "#fef9f3", "border": "1px solid #fbe3cc"}
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "row",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "width": "100%",
                        "flexWrap": "wrap",
                        "gap": "10px"
                    }
                )
            ],
            style={
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "center",
                "alignItems": "center",
                "width": "100%",
                "marginBottom": "25px",
                "padding": "10px 10px 0 10px",
                "boxSizing": "border-box"
            }
        ),

        html.Div(
            id="cytoscape-wrapper",
            title="",
            children=[
                cyto.Cytoscape(
                    id="cytoscape-neo4j",
                    elements=[],
                    layout={"name": "cose"},
                    style={
                        "width": "100%",
                        "height": "600px",
                        "border": "1px solid #ccc",
                    },
                    stylesheet=[
                        {
                            "selector": "node",
                            "style": {
                                "label": "data(label)",
                                "background-color": "#777777",
                                "color": "#333",
                                "font-size": "12px",
                                "text-valign": "center",
                                "text-halign": "right",
                            },
                        },
                        {
                            "selector": "node[type = 'chemical']",
                            "style": {"background-color": "#0074D9"},
                        },
                        {
                            "selector": "node[type = 'protein']",
                            "style": {"background-color": "#2ECC40"},
                        },
                        {
                            "selector": "node[type = 'reaction']",
                            "style": {"background-color": "#FF851B"},
                        },
                        {
                            "selector": "edge",
                            "style": {
                                "label": "data(label)",
                                "line-color": "#FF4136",  # Vivid Red lines default
                                "target-arrow-color": "#FF4136",
                                "target-arrow-shape": "triangle",
                                "curve-style": "bezier",
                                "control-point-step-size": "40px",
                                "text-rotation": "autorotate",
                                "font-size": "10px",
                                "text-margin-y": "-10px",
                            },
                        },
                        # Base rule for normal HAS_AGENT or HAS_TARGET edges (Grey color)
                        {
                            "selector": "edge[label = 'HAS_AGENT'], edge[label = 'HAS_TARGET'], edge[label = 'TARGET_OF']",
                            "style": {
                                "line-color": "#999999",
                                "target-arrow-color": "#999999",
                            },
                        },
                        # --- EXCEPTION RULE FOR THE LAST EDGE IF IT IS HAS_TARGET OR TARGET_OF ---
                        # Inverts the default grey edge to red (or choose any inverted color hex like #000000)
                        {
                            "selector": "edge[label = 'HAS_TARGET'][?is_last], edge[label = 'TARGET_OF'][?is_last]",
                            "style": {
                                "line-color": "#FF4136",  # Inverted color choice
                                "target-arrow-color": "#FF4136",  # Inverted color choice
                            },
                        },
                    ],
                )
            ],
        ),
        html.Div(
            id="table-output-container",
            style={"width": "100%", "padding": "0 10px", "boxSizing": "border-box"},
        ),
    ]
)


@app.callback(
    [Output("cytoscape-neo4j", "elements"),
     Output("table-output-container", "children"),
     Output("kpi-source-id", "children"),
     Output("kpi-target-id", "children"),
     Output("kpi-hops", "children")],
    Input("url", "search")
)
def update_graph_and_table(search_query):
    target_ids = None
    source_id_val = "N/A"
    target_id_val = "N/A"
    hops_val = "N/A"

    if search_query:
        parsed_params = urllib.parse.parse_qs(search_query.lstrip("?"))

        if "source_id" in parsed_params:
            source_id_val = parsed_params["source_id"][0]

        if "target_id" in parsed_params:
            target_id_val = parsed_params["target_id"][0]

        if "hops" in parsed_params:
            hops_val = parsed_params["hops"][0]

        if "vertices_path" in parsed_params:
            target_ids = [idx.strip() for idx in parsed_params["vertices_path"][0].split(",") if idx.strip()]

    elements, source_name, target_name = fetch_graph_elements_and_entities(target_ids, source_id_val, target_id_val)

    table_layout = [
        html.H3("Represented elements", style={"fontFamily": "Arial", "marginTop": "40px"}),
        build_html_table(elements, target_ids=target_ids),
    ]

    source_kpi_children = [
        html.Div("SOURCE ENTITY",
                 style={"fontSize": "11px", "fontWeight": "bold", "color": "#2980b9", "letterSpacing": "1.2px"}),
        html.Div(str(source_name),
                 style={"fontSize": "14px", "fontWeight": "bold", "color": "#2c3e50", "marginTop": "4px",
                        "wordBreak": "break-all"})
    ]

    target_kpi_children = [
        html.Div("TARGET ENTITY",
                 style={"fontSize": "11px", "fontWeight": "bold", "color": "#27ae60", "letterSpacing": "1.2px"}),
        html.Div(str(target_name),
                 style={"fontSize": "14px", "fontWeight": "bold", "color": "#2c3e50", "marginTop": "4px",
                        "wordBreak": "break-all"})
    ]

    hops_kpi_children = [
        html.Div("HOPS",
                 style={"fontSize": "11px", "fontWeight": "bold", "color": "#9b59b6", "letterSpacing": "1.2px"}),
        html.Div(str(hops_val),
                 style={"fontSize": "16px", "fontWeight": "bold", "color": "#2c3e50", "marginTop": "4px"})
    ]

    return elements, table_layout, source_kpi_children, target_kpi_children, hops_kpi_children


@app.callback(
    Output("cytoscape-wrapper", "title"),
    Input("cytoscape-neo4j", "mouseoverNodeData"),
    prevent_initial_call=True,
)
def display_hover_data(node_data):
    if not node_data:
        return ""

    tooltip_lines = [
        f"Name: {node_data.get('name', 'N/A')}",
        f"Type: {node_data.get('type', 'Unknown').capitalize()}",
    ]

    if node_data.get("gene_ID") and node_data["gene_ID"] != "":
        tooltip_lines.append(f"Gene ID: {node_data['gene_ID']}")

    if node_data.get("chem_ID") and node_data["chem_ID"] != "":
        tooltip_lines.append(f"Chemical ID: {node_data['chem_ID']}")

    if node_data.get("action") and node_data["action"] != "":
        tooltip_lines.append(f"Action: {node_data['action']}")

    return "\n".join(tooltip_lines)


if __name__ == "__main__":
    app.run(debug=False, port=PORT)