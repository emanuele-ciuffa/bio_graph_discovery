import base64
import os
from dash import Dash, dcc, html, dash_table
import pandas as pd
from neo4j import GraphDatabase
from src.utils import file_utils
from src.utils.config_handler import Config_handler
from src.utils import db_info_utils

# Reading organism from DB
selected_organism = db_info_utils.get_db_organism()

# Instantiate dashboard config handler
dashboard_config_handler = Config_handler("config-dashboard.yml")

# Reading title from dashboard config file
TITLE = dashboard_config_handler.read_property("link_prediction.first_level_view.title")

# Reading port from dashboard config file
PORT = dashboard_config_handler.read_property("link_prediction.first_level_view.port")

# Set the root link for the redirect when selecting the predicted link from the table for viewing details
REDIRECT_ROOT = f"http://127.0.0.1:{dashboard_config_handler.read_property('link_prediction.second_level_view.port')}/"

# -----------------------------------------------------------------------------
# DEFINE STATIC PATHS & NEO4J CONFIG
# -----------------------------------------------------------------------------
CSV_REPORT_PATH = file_utils.get_name_with_organism(r"..\..\out\link_prediction_out\link_prediction_report_path.csv", selected_organism) #r"..\..\out\link_prediction_out\link_prediction_report_path.csv"
CSV_PREDICTIONS_PATH = file_utils.get_name_with_organism(r"..\..\out\link_prediction_out\predictions.csv", selected_organism) # r"..\..\out\link_prediction_out\predictions_" + organism + ".csv"
IMAGE_PATH = r"..\..\out\link_prediction_out\metrics_roc_precision.png"

# Instatiate config handler for retrieving neo4j parameters
config_handler_neo4j = Config_handler("config-neo4j.yml")

# Reading references to connect to neo4j DB
URI = config_handler_neo4j.read_property("neo4j.uri")

USER = config_handler_neo4j.read_property("neo4j.user")
PASSWORD = config_handler_neo4j.read_property("neo4j.password")

# --- 1. NEO4J CONNECTION & DATA FETCHING ---
AUTH = (USER, PASSWORD)

# -----------------------------------------------------------------------------
# DATA EXTRACTION & NEO4J MAPPING (No Functions)
# -----------------------------------------------------------------------------
# Read the report CSV for KPIs
df_report = pd.read_csv(CSV_REPORT_PATH)
row = df_report.iloc[0]

organism = row["organism"]
roc_auc = f"{row['roc_auc']:.4f}"
avg_precision = f"{row['avg_precision']:.4f}"
num_nodes = f"{row['num_nodes']:,}"
num_train = f"{row['num_train_interactions']:,}"

# Read the predictions CSV for the DataTable
df_pred_raw = pd.read_csv(CSV_PREDICTIONS_PATH)

# Filter rows where 'Actual_Exist' is equal to 'no' OR 'yes' (case-insensitive for safety)
df_filtered = df_pred_raw[df_pred_raw["Actual_Exist"].str.lower().isin(["no", "yes"])].copy()

# ADAPTED: Map 'Actual_Exist' to 'TP' or 'FP' before dropping the column
import numpy as np
df_filtered["Classification"] = np.where(df_filtered["Actual_Exist"].str.lower() == "yes", "TP", "FP")

# Drop 'Actual_Exist' and 'relationship' columns so they are hidden from the user
df_pred = df_filtered.drop(columns=["Actual_Exist", "relationship"], errors="ignore")

# --- Identify CSV Structural Columns dynamically ---
numeric_cols = df_pred.select_dtypes(include=['number']).columns
score_col = numeric_cols[0] if len(numeric_cols) > 0 else df_pred.columns[-1]

remaining_cols = [c for c in df_pred.columns if c != score_col]
src_id_col = remaining_cols[0] if len(remaining_cols) > 0 else df_pred.columns[0]
dst_id_col = remaining_cols[1] if len(remaining_cols) > 1 else df_pred.columns[1]

# --- Neo4j Integration Layer ---
# Gather all unique IDs from both source and target positions
all_unique_ids = list(set(df_pred[src_id_col].dropna().tolist() + df_pred[dst_id_col].dropna().tolist()))

node_metadata = {}  # Format: { 'id_value': {'name': '...', 'label': '...'} }

# Query Neo4j 5 using modern elementId lookup matching strategy
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:
        # Match nodes whose ID property matches our list. Change 'id' to whatever property your graph uses (e.g., uid, name)
        cypher_query = """
        MATCH (n)
        WHERE n.id IN $id_list
        RETURN n.id AS id, n.name AS name, labels(n)[0] AS label
        """
        results = session.run(cypher_query, id_list=all_unique_ids)
        for record in results:
            node_metadata[record["id"]] = {
                "name": record["name"] if record["name"] else record["id"],
                "label": record["label"] if record["label"] else "Unknown"
            }

# Map metadata back to dataframe, default to ID values if not matched in DB
df_pred["Source Name"] = df_pred[src_id_col].apply(lambda x: node_metadata.get(x, {}).get("name", x))
df_pred["Source Label"] = df_pred[src_id_col].apply(lambda x: node_metadata.get(x, {}).get("label", "Unknown"))
df_pred["Target Name"] = df_pred[dst_id_col].apply(lambda x: node_metadata.get(x, {}).get("name", x))
df_pred["Target Label"] = df_pred[dst_id_col].apply(lambda x: node_metadata.get(x, {}).get("label", "Unknown"))

# --- Inject Dynamic Clickable Inspect Link using original IDs ---
tooltip_text = "click to display link prediction details"

# Dynamically route to local img/search.png relative to this script directory
current_script_dir = os.path.dirname(os.path.abspath(__file__))
local_search_icon_path = os.path.join(current_script_dir, "img", "search.png")

try:
    with open(local_search_icon_path, "rb") as f:
        encoded_icon = base64.b64encode(f.read()).decode("utf-8")
    magnifying_glass_url = f"data:image/png;base64,{encoded_icon}"
except Exception:
    # Safe fallback to standard remote icon if file access hits permissions issues or is missing
    magnifying_glass_url = "https://img.icons8.com/material-outlined/24/000000/search--v1.png"

# Inject explicit HTML tags into the column to handle Base64 URI renders inside the datatable securely
df_pred.insert(
    0,
    "Inspect",
    df_pred.apply(
        lambda r: f'<a href="{REDIRECT_ROOT}?source_id={r[src_id_col]}&target_id={r[dst_id_col]}&score={r[score_col]}" target="_blank" title="{tooltip_text}">'
                  f'<img src="{magnifying_glass_url}" alt="Inspect" style="height:20px; width:20px; vertical-align:middle;"/>'
                  f'</a>',
        axis=1
    )
)

# ADAPTED: Rearrange columns to place 'Classification' at the very end
final_display_cols = ["Inspect", "Source Name", "Source Label", "Target Name", "Target Label", score_col, "Classification"]
# Keep fallback layout protection if columns are missing
df_table_data = df_pred[[c for c in final_display_cols if c in df_pred.columns]]

# Encode local graph plot image
encoded_image = ""
if os.path.exists(IMAGE_PATH):
    with open(IMAGE_PATH, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# -----------------------------------------------------------------------------
# DASH APPLICATION & LAYOUT
# -----------------------------------------------------------------------------
app = Dash(__name__)

table_columns = []
for col_name in df_table_data.columns:
    if col_name == "Inspect":
        table_columns.append({
            "name": "",
            "id": col_name,
            "type": "text",
            "presentation": "markdown"
        })
    else:
        column_dict = {"name": col_name, "id": col_name}
        if pd.api.types.is_numeric_dtype(df_table_data[col_name]):
            column_dict["type"] = "numeric"
        else:
            column_dict["type"] = "text"
        table_columns.append(column_dict)

app.layout = html.Div(
    style={
        "width": "90%",
        "margin": "0 auto",
        "paddingTop": "20px",
        "paddingBottom": "40px",
        "fontFamily": "Segoe UI, Arial, sans-serif",
    },
    children=[
        html.H1(
            children=f"{TITLE}",
            style={
                "fontFamily": "sans-serif",
                "textAlign": "center",
                "color": "#2c3e50",
                "marginBottom": "45px",
            },
        ),

        # KPI Panel
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "marginBottom": "40px",
                "flexWrap": "wrap",
                "gap": "20px",
            },
            children=[
                html.Div(style={"backgroundColor": "white", "boxShadow": "0px 0px 15px rgba(0,0,0,0.05)", "padding": "20px", "borderRadius": "8px", "textAlign": "center", "flex": "1", "minWidth": "200px"},
                         children=[html.H3("ROC AUC", style={"margin": "0", "color": "#7f8c8d", "fontSize": "16px"}), html.P(roc_auc, style={"fontSize": "28px", "fontWeight": "bold", "color": "#2980b9", "margin": "10px 0 0 0"})]),
                html.Div(style={"backgroundColor": "white", "boxShadow": "0px 0px 15px rgba(0,0,0,0.05)", "padding": "20px", "borderRadius": "8px", "textAlign": "center", "flex": "1", "minWidth": "200px"},
                         children=[html.H3("Avg Precision", style={"margin": "0", "color": "#7f8c8d", "fontSize": "16px"}), html.P(avg_precision, style={"fontSize": "28px", "fontWeight": "bold", "color": "#27ae60", "margin": "10px 0 0 0"})]),
                html.Div(style={"backgroundColor": "white", "boxShadow": "0px 0px 15px rgba(0,0,0,0.05)", "padding": "20px", "borderRadius": "8px", "textAlign": "center", "flex": "1", "minWidth": "200px"},
                         children=[html.H3("Total Nodes", style={"margin": "0", "color": "#7f8c8d", "fontSize": "16px"}), html.P(num_nodes, style={"fontSize": "28px", "fontWeight": "bold", "color": "#8e44ad", "margin": "10px 0 0 0"})]),
                html.Div(style={"backgroundColor": "white", "boxShadow": "0px 0px 15px rgba(0,0,0,0.05)", "padding": "20px", "borderRadius": "8px", "textAlign": "center", "flex": "1", "minWidth": "200px"},
                         children=[html.H3("Train Interactions", style={"margin": "0", "color": "#7f8c8d", "fontSize": "16px"}), html.P(num_train, style={"fontSize": "28px", "fontWeight": "bold", "color": "#f39c12", "margin": "10px 0 0 0"})]),
                # Organism Card (Added directly to the top-right position within the panel)
                html.Div(style={"backgroundColor": "white", "boxShadow": "0px 0px 15px rgba(0,0,0,0.05)", "padding": "20px", "borderRadius": "8px", "textAlign": "center", "flex": "1", "minWidth": "200px"},
                         children=[html.H3("Organism", style={"margin": "0", "color": "#7f8c8d", "fontSize": "16px"}), html.P(str(selected_organism), style={"fontSize": "28px", "fontWeight": "bold", "color": "#e67e22", "margin": "10px 0 0 0"})]),
            ],
        ),

        # Plot Panel
        html.Div(
            style={"textAlign": "center", "backgroundColor": "white", "padding": "25px", "borderRadius": "8px", "boxShadow": "0px 0px 15px rgba(0,0,0,0.05)", "marginBottom": "40px"},
            children=[
                html.H2("Evaluation Plots", style={"color": "#34495e", "marginBottom": "20px", "fontSize": "20px", "fontFamily": "sans-serif"}),
                html.Img(src=f"data:image/png;base64,{encoded_image}" if encoded_image else "", style={"maxWidth": "100%", "height": "auto", "borderRadius": "4px"}),
            ],
        ),

        html.H2("Link Predictions Details (TP + FP)", style={"fontFamily": "sans-serif", "color": "#2c3e50", "marginBottom": "12px", "fontSize": "20px"}),

        # Table Layout Component
        dash_table.DataTable(
            id="predictions-table",
            columns=table_columns,
            data=df_table_data.to_dict("records"),
            sort_action="native",
            filter_action="native",
            page_action="native",
            page_current=0,
            filter_options={"case": "insensitive"},
            page_size=20,
            markdown_options={"link_target": "_blank", "html": True},
            style_table={"overflowX": "auto", "boxShadow": "0px 0px 15px rgba(0,0,0,0.1)"},
            style_cell={"fontFamily": "Arial, sans-serif", "padding": "12px", "textAlign": "left", "minWidth": "100px", "backgroundColor": "#fafafa"},
            style_cell_conditional=[
                {
                    "if": {"column_id": "Inspect"},
                    "width": "60px",
                    "minWidth": "60px",
                    "maxWidth": "60px",
                }
            ],
            style_header={"backgroundColor": "#1f77b4", "color": "white", "fontWeight": "bold", "textTransform": "capitalize"},
            style_header_conditional=[{"if": {"column_id": "Inspect"}, "pointerEvents": "none"}],
            style_data={"border": "1px solid #e4e4e7"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f2f2f2"},
                {"if": {"column_id": "Inspect"}, "textAlign": "center"}
            ],
            style_filter_conditional=[{"if": {"column_id": "Inspect"}, "visibility": "hidden", "pointerEvents": "none"}],
            css=[{"selector": 'tr th[data-dash-column="Inspect"] .sort', "rule": "display: none !important;"}]
        ),
    ],
)

if __name__ == "__main__":
    app.run(debug=False, port=PORT)