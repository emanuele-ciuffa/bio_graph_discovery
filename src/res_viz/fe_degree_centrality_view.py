import os
import urllib.parse
from dash import Dash, dcc, html, dash_table, Input, Output
from neo4j import GraphDatabase  # Imperial driver instance
from src.utils.config_handler import Config_handler
from src.utils import db_info_utils

# Instantiate common config handler
common_config_handler = Config_handler("config-common.yml")

# Reading organism from DB
selected_organism = db_info_utils.get_db_organism()

# Instantiate dashboard config handler
dashboard_config_handler = Config_handler("config-dashboard.yml")

# Reading port for the SECOND level view from dashboard config file
PORT = dashboard_config_handler.read_property("degree_centrality.second_level_view.port")

# Reading title for the SECOND level view from dashboard config file
TITLE = dashboard_config_handler.read_property("degree_centrality.second_level_view.title")

# Instatiate config handler for retrieving neo4j parameters
config_handler_neo4j = Config_handler("config-neo4j.yml")

# --- Neo4j Configuration Fetch ---
NEO4J_URI = config_handler_neo4j.read_property("neo4j.uri")
NEO4J_USER = config_handler_neo4j.read_property("neo4j.user")
NEO4J_PASSWORD = config_handler_neo4j.read_property("neo4j.password")

# Initialize Neo4j Driver Connection Instance
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Initialize the Dash app
app = Dash(__name__)

# Design structural template layout equipped with URL listening capabilities
app.layout = html.Div(
    style={
        "width": "90%",
        "margin": "0 auto",
        "paddingTop": "40px",
        "paddingBottom": "60px",
        "fontFamily": "Segoe UI, Arial, sans-serif",
    },
    children=[
        # Dash component that tracks URL paths and request query parameters
        dcc.Location(id="url", refresh=False),

        # Dashboard Main Title Header
        html.H1(
            id="page-title",
            children=TITLE,
            style={
                "textAlign": "center",
                "color": "#2c3e50",
                "marginBottom": "45px",
            }
        ),

        # Dynamic container where row layouts will be structurally injected
        html.Div(id="kpi-rows-wrapper"),

        # Horizontal Separator
        html.Hr(style={"border": "0", "borderTop": "1px solid #e4e4e7", "margin": "40px 0"}),

        # Section header for relationships datatable
        html.H2(
            "Connected Network Relationships",
            style={"color": "#2c3e50", "marginBottom": "20px", "fontSize": "22px"}
        ),

        # Container to hold the dynamic relationship data table view
        html.Div(id="table-container")
    ]
)


# Helper function to generate clean metric layout components safely
def create_kpi_card(title, value, color="#333333"):
    return html.Div(
        style={
            "backgroundColor": "#ffffff",
            "boxShadow": "0px 4px 10px rgba(0, 0, 0, 0.05)",
            "padding": "20px",
            "borderRadius": "8px",
            "textAlign": "center",
            "border": "1px solid #e4e4e7",
            "flex": "1",
            "minWidth": "180px"
        },
        children=[
            html.H3(title, style={"margin": "0", "color": "#7f8c8d", "fontSize": "14px", "textTransform": "uppercase"}),
            html.P(str(value), style={"fontSize": "22px", "fontWeight": "bold", "color": color, "margin": "10px 0 0 0"})
        ]
    )


# Function to fetch target node specific structural keys (MESH ID & Gene ID)
def fetch_node_metadata(node_id):
    query = """
    MATCH (n)
    WHERE n.id = $target_id
    RETURN n.chem_ID AS mesh_id, n.gene_ID AS gene_id
    LIMIT 1
    """
    metadata = {"mesh_id": "N/A", "gene_id": "N/A"}
    try:
        with driver.session() as session:
            result = session.run(query, target_id=node_id)
            record = result.single()
            if record:
                metadata["mesh_id"] = record["mesh_id"] if record["mesh_id"] is not None else "N/A"
                metadata["gene_id"] = record["gene_id"] if record["gene_id"] is not None else "N/A"
    except Exception as e:
        print(f"Error querying Node metadata from Neo4j: {e}")
    return metadata


# Function to query Neo4j graph using the active request node identification safely
def fetch_neighborhood_relationships(node_id):
    query = """
    // First Pass: Find all outgoing relationships (Out-degree)
    MATCH (n)-[r]->(adjacent)
    WHERE n.id = $target_id
    RETURN 
        adjacent.name AS name,
        labels(adjacent)[0] AS type,
        adjacent.chem_ID AS mesh_id,
        adjacent.gene_ID AS gene_id,
        r.relationship AS relationship,
        r.type AS relation_type,
        'out-degree' AS degree_type

    UNION ALL

    // Second Pass: Find all incoming relationships (In-degree)
    MATCH (n)<-[r]-(adjacent)
    WHERE n.id = $target_id
    RETURN 
        adjacent.name AS name,
        labels(adjacent)[0] AS type,
        adjacent.chem_ID AS mesh_id,
        adjacent.gene_ID AS gene_id,
        r.relationship AS relationship,
        r.type AS relation_type,
        'in-degree' AS degree_type
    """
    records_data = []
    try:
        with driver.session() as session:
            result = session.run(query, target_id=node_id)
            for record in result:
                records_data.append({
                    "name": record["name"] if record["name"] is not None else "N/A",
                    "type": record["type"] if record["type"] is not None else "N/A",
                    "mesh_id": record["mesh_id"] if record["mesh_id"] is not None else "N/A",
                    "gene_id": record["gene_id"] if record["gene_id"] is not None else "N/A",
                    "relationship": record["relationship"] if record["relationship"] is not None else "N/A",
                    "relation_type": record["relation_type"] if record["relation_type"] is not None else "N/A",
                    "degree_type": record["degree_type"]
                })
    except Exception as e:
        print(f"Error querying Neo4j Database: {e}")
    return records_data


# Combined multi-output callback to construct layout and execute graph analytical queries dynamically
@app.callback(
    [Output("kpi-rows-wrapper", "children"),
     Output("table-container", "children")],
    Input("url", "search")
)
def update_page_content(search_string):
    # Fallback default responses if URL parameters are missing entirely
    if not search_string:
        error_msg = html.Div(
            "No entity data requested. Please navigate to this page from the primary Degree Centrality dashboard.",
            style={"textAlign": "center", "color": "#7f8c8d", "padding": "40px"}
        )
        return error_msg, html.Div()

    # Strip leading '?' if present and parse request query parameters into dict mapping strings
    parsed_params = urllib.parse.parse_qs(search_string.lstrip("?"))

    # Extract values gracefully using safe lookup fallbacks
    entity_id = parsed_params.get("id_node", ["N/A"])[0]
    entity_name = parsed_params.get("name", ["N/A"])[0]
    entity_type = parsed_params.get("type", ["N/A"])[0]
    in_degree = parsed_params.get("inDegree", ["0"])[0]
    out_degree = parsed_params.get("outDegree", ["0"])[0]
    degree = parsed_params.get("degree", ["0"])[0]
    degree_centrality = parsed_params.get("degree_centrality", ["0.00%"])[0]

    # Dynamically match colors scheme based on entity type values parsed
    type_lower = entity_type.lower()
    if "chemical" in type_lower:
        type_color = "#0074D9"
    elif "protein" in type_lower:
        type_color = "#2ECC40"
    elif "reaction" in type_lower:
        type_color = "#FF851B"
    else:
        type_color = "#7f8c8d"

    # Execute target node query execution to extract specific key indexes
    target_metadata = fetch_node_metadata(entity_id)

    # Row layouts wrapper block structures
    row_style = {
        "display": "flex",
        "justifyContent": "space-between",
        "flexWrap": "wrap",
        "gap": "20px",
        "marginBottom": "25px"
    }

    # Generate Row Layout Structures
    kpi_layout = html.Div([
        # Row 1 Layout Block (Selected Entity ID KPI removed)
        html.Div(
            style=row_style,
            children=[
                create_kpi_card("Entity Name", entity_name, "#2c3e50"),
                create_kpi_card("Entity Type", entity_type, type_color),
                create_kpi_card("MESH ID", target_metadata["mesh_id"],
                                "#0074D9" if target_metadata["mesh_id"] != "N/A" else "#7f8c8d"),
                create_kpi_card("Gene ID", target_metadata["gene_id"],
                                "#2ECC40" if target_metadata["gene_id"] != "N/A" else "#7f8c8d"),
                create_kpi_card("Organism", selected_organism, "#e67e22")
            ]
        ),

        # Row 2 Layout Block
        html.Div(
            style=row_style,
            children=[
                create_kpi_card("In-Degree",
                                f"{int(float(in_degree)):,}" if in_degree.isdigit() or in_degree.replace('.', '',
                                                                                                         1).isdigit() else in_degree,
                                "#2B8CBE"),
                create_kpi_card("Out-Degree",
                                f"{int(float(out_degree)):,}" if out_degree.isdigit() or out_degree.replace('.', '',
                                                                                                            1).isdigit() else out_degree,
                                "#FDB863"),
                create_kpi_card("Total Degree", f"{int(float(degree)):,}" if degree.isdigit() or degree.replace('.', '',
                                                                                                                1).isdigit() else degree,
                                "#333333"),
                create_kpi_card("Degree Centrality", degree_centrality, "#9b59b6")
            ]
        )
    ])

    # Fetch rows live from the local graph database environment based on active request id parameters
    neo4j_records = fetch_neighborhood_relationships(entity_id)

    # Render dynamic datatable layout showing network connectivity metrics
    table_layout = dash_table.DataTable(
        data=neo4j_records,
        columns=[
            {"name": "Adjacent Node Name", "id": "name", "type": "text"},
            {"name": "Node Type", "id": "type", "type": "text"},
            {"name": "MESH ID", "id": "mesh_id", "type": "text"},
            {"name": "Gene ID", "id": "gene_id", "type": "text"},
            {"name": "Relationship", "id": "relationship", "type": "text"},
            {"name": "Relation Type", "id": "relation_type", "type": "text"},
            {"name": "Direction (Degree Type)", "id": "degree_type", "type": "text"}
        ],
        sort_action="native",
        filter_action="native",
        page_action="native",
        filter_options={"case": "insensitive"},
        page_size=10,
        style_table={'overflowX': 'auto', 'boxShadow': '0px 0px 15px rgba(0,0,0,0.1)'},
        style_cell={
            'fontFamily': 'Arial, sans-serif',
            'padding': '12px',
            'textAlign': 'left',
            'minWidth': '120px',
            'backgroundColor': '#fafafa'
        },
        style_header={
            'backgroundColor': '#1f77b4',
            'color': 'white',
            'fontWeight': 'bold',
            'textTransform': 'capitalize'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f2f2f2',
            }
        ]
    )

    return kpi_layout, table_layout


if __name__ == "__main__":
    try:
        app.run(debug=False, port=PORT)
    finally:
        # Close the Neo4j driver context explicitly on termination
        driver.close()
