import base64
import os
import re
import pandas as pd
from dash import Dash, dcc, html, dash_table
from neo4j import GraphDatabase
from src.utils.config_handler import Config_handler
from src.utils import db_info_utils

# Reading organism from DB
selected_organism = db_info_utils.get_db_organism()

dashboard_config_handler = Config_handler("config-dashboard.yml")
PORT = dashboard_config_handler.read_property("shortest_path.first_level_view.port")
TITLE = dashboard_config_handler.read_property("shortest_path.first_level_view.title")
REDIRECT_ROOT = f"http://127.0.0.1:{dashboard_config_handler.read_property('shortest_path.second_level_view.port')}/"

config_handler_neo4j = Config_handler("config-neo4j.yml")
driver = GraphDatabase.driver(
    config_handler_neo4j.read_property("neo4j.uri"),
    auth=(config_handler_neo4j.read_property("neo4j.user"), config_handler_neo4j.read_property("neo4j.password"))
)

DIRECTORY_PATH = os.path.join("..", "..", "out", "graph_analysis_out")


# --- Helper Functions ---

def generate_shortest_path_data():
    plot_records = []
    if not os.path.exists(DIRECTORY_PATH):
        return pd.DataFrame(columns=["Entity ID", "Valid Hops Count", "Filename"])

    filename_pattern = re.compile(r"^shortest_path__(.+?)__(.+)\.csv$")
    normalized_config_org = str(selected_organism).strip().lower().replace(" ", "_")

    for filename in os.listdir(DIRECTORY_PATH):
        match = filename_pattern.match(filename)
        if match:
            entity_id = match.group(1).strip()
            file_organism = match.group(2).strip()
            if file_organism.lower().strip().replace(" ", "_") == normalized_config_org:
                file_path = os.path.join(DIRECTORY_PATH, filename)
                try:
                    df_file = pd.read_csv(file_path, sep=";")
                    hop_col = [c for c in df_file.columns if c.lower().strip() in ['hop', 'hops']]
                    valid_hops_count = int(
                        (pd.to_numeric(df_file[hop_col[0]], errors='coerce') > 0).sum()) if hop_col else 0
                    plot_records.append(
                        {"Entity ID": entity_id, "Valid Hops Count": valid_hops_count, "Filename": filename})
                except Exception as e:
                    print(f"Error parsing {filename}: {e}")
    return pd.DataFrame(plot_records)


def process_shortest_path_file(filename, entity_id):
    """Parses CSV and fetches metadata using the requested query for KPIs."""
    file_path = os.path.join(DIRECTORY_PATH, filename)
    try:
        df_target = pd.read_csv(file_path, sep=";")
        df_target = df_target.loc[:, ~df_target.columns.str.contains('^Unnamed')]

        id_candidates = [c for c in df_target.columns if c.lower().strip() in ['id', 'node_id', 'entity_id']]
        target_id_col = id_candidates[0] if id_candidates else df_target.columns[0]
        hop_col = [c for c in df_target.columns if c.lower().strip() in ['hop', 'hops']][0]

        df_target[hop_col] = pd.to_numeric(df_target[hop_col], errors='coerce')
        df_target = df_target[df_target[hop_col] > 0]

        with driver.session() as session:
            kpi_query = """
            MATCH (n)
            WHERE n.id = $node_id
            RETURN n.name AS name, labels(n)[0] AS label
            """
            res_kpi = session.run(kpi_query, node_id=entity_id)
            kpi_record = res_kpi.single()
            kpi_name = kpi_record["name"] if kpi_record else entity_id
            kpi_label = kpi_record["label"] if kpi_record else "Unknown"

            table_ids = df_target[target_id_col].unique().tolist()
            res_table = session.run(
                "MATCH (n) WHERE n.id IN $id_list RETURN n.id AS id, n.name AS name",
                id_list=table_ids
            )
            node_map = {rec["id"]: rec["name"] for rec in res_table}

        # Format Data
        df_target["Inspect"] = df_target.apply(
            lambda
                row: f'<a href="{REDIRECT_ROOT}?source_id={row[target_id_col]}&target_id={entity_id}&hops={row[hop_col]}&vertices_path={str(row["vertices_path"]).replace(" ", "")}" target="_blank"><img src="{magnifying_glass_url}" alt="Inspect" style="height:20px; width:20px; vertical-align:middle;"/></a>',
            axis=1
        )
        df_target["Entity Name"] = df_target[target_id_col].apply(lambda x: node_map.get(x, x))

        df_display = df_target[["Inspect", "Entity Name", hop_col]].sort_values(by=hop_col, ascending=True)

        columns = [
            {"name": "", "id": "Inspect", "presentation": "markdown"} if c == "Inspect"
            else {"name": c, "id": c, "type": "numeric" if c == hop_col else "text"}
            for c in df_display.columns
        ]

        return {
            "name": kpi_name,
            "type": kpi_label,
            "columns": columns,
            "data": df_display.to_dict("records"),
            "title": f"Shortest Path Details for: {kpi_name} (Filtered: Hops > 0)"
        }
    except Exception as e:
        print(f"Error: {e}")
        return None


# --- App Initialization ---

current_script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
with open(os.path.join(current_script_dir, "img", "search.png"), "rb") as f:
    magnifying_glass_url = f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

df_hops = generate_shortest_path_data()
initial_file_data = process_shortest_path_file(df_hops.iloc[0]["Filename"],
                                               df_hops.iloc[0]["Entity ID"]) if not df_hops.empty else None

app = Dash(__name__)

# --- TARGETED CSS INJECTION TO VAPORIZE SORT ARROWS AND FILTER INPUT FROM "Inspect" ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* 1. Vaporize the sort pink/blue arrows completely on the Inspect column header */
            th[data-dash-column="Inspect"] .column-actions,
            th[data-dash-column="Inspect"] .sort-user-select {
                display: none !important;
            }
            th[data-dash-column="Inspect"] .column-header-name {
                margin-right: 0px !important;
            }

            /* 2. Erase the filter box input field ("Aa filter") completely for the Inspect column */
            tr.dash-filter-row td[data-dash-column="Inspect"] input {
                display: none !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

kpi_card_style = {"backgroundColor": "white", "boxShadow": "0px 0px 15px rgba(0,0,0,0.05)", "padding": "20px",
                  "borderRadius": "8px", "textAlign": "center", "flex": "0 1 300px", "margin": "10px"}

app.layout = html.Div(
    style={"width": "90%", "margin": "0 auto", "paddingTop": "20px", "fontFamily": "Segoe UI, Arial, sans-serif"},
    children=[
        html.H1(TITLE, style={"textAlign": "center", "color": "#2c3e50", "marginBottom": "45px"}),

        # KPI Summary
        html.Div(
            style={"display": "flex", "justifyContent": "center", "marginBottom": "40px", "flexWrap": "wrap",
                   "gap": "20px"},
            children=[
                html.Div(style=kpi_card_style,
                         children=[html.H3("Entity name", style={"margin": "0", "color": "#7f8c8d"}),
                                   html.P(initial_file_data['name'] if initial_file_data else "N/A",
                                          id="kpi-entity-name",
                                          style={"fontSize": "22px", "fontWeight": "bold", "color": "#1f77b4"})]),
                html.Div(style=kpi_card_style, children=[html.H3("Type", style={"margin": "0", "color": "#7f8c8d"}),
                                                         html.P(
                                                             initial_file_data['type'] if initial_file_data else "N/A",
                                                             id="kpi-entity-type",
                                                             style={"fontSize": "22px", "fontWeight": "bold",
                                                                    "color": "#9b59b6"})]),
                html.Div(style=kpi_card_style, children=[html.H3("Organism", style={"margin": "0", "color": "#7f8c8d"}),
                                                         html.P(selected_organism,
                                                                style={"fontSize": "28px", "fontWeight": "bold",
                                                                       "color": "#e67e22"})]),
            ]
        ),

        # Data Table
        html.Div(
            id="table-container",
            style={"display": "block", "marginBottom": "40px"},
            children=[
                html.H2(initial_file_data['title'] if initial_file_data else "No data available", id="table-title",
                        style={"color": "#2c3e50", "fontSize": "20px"}),
                dash_table.DataTable(
                    id="csv-details-table",
                    columns=initial_file_data['columns'] if initial_file_data else [],
                    data=initial_file_data['data'] if initial_file_data else [],
                    sort_action="native",
                    filter_action="native",
                    page_action="native",
                    filter_options={"case": "insensitive"},
                    page_size=20,
                    markdown_options={"link_target": "_blank", "html": True},
                    style_table={"overflowX": "auto", "boxShadow": "0px 0px 15px rgba(0,0,0,0.1)"},
                    style_cell={"fontFamily": "Arial, sans-serif", "padding": "12px", "textAlign": "left",
                                "minWidth": "100px", "backgroundColor": "#fafafa"},
                    style_cell_conditional=[
                        {
                            "if": {"column_id": "Inspect"},
                            "width": "60px",
                            "minWidth": "60px",
                            "maxWidth": "60px",
                        }
                    ],
                    style_header={"backgroundColor": "#1f77b4", "color": "white", "fontWeight": "bold",
                                  "textTransform": "capitalize"},

                    # Core security backup rules to completely block text selections/clicks on the first row column
                    style_header_conditional=[
                        {
                            "if": {"column_id": "Inspect"},
                            "pointerEvents": "none",
                        }
                    ],
                    style_filter_conditional=[
                        {
                            "if": {"column_id": "Inspect"},
                            "pointerEvents": "none",
                        }
                    ],

                    style_data={"border": "1px solid #e4e4e7"},
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#f2f2f2"},
                        {"if": {"column_id": "Inspect"}, "textAlign": "center"}
                    ]
                )
            ]
        ),
    ]
)

if __name__ == "__main__":
    try:
        app.run(debug=False, port=PORT)
    finally:
        driver.close()