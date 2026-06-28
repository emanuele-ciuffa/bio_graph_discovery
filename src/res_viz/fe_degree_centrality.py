import base64
import os
import urllib.parse
from dash import Dash, html, dash_table
import pandas as pd
from dash import dcc, Input, Output
import plotly.express as px
from src.utils import file_utils
from src.utils.config_handler import Config_handler
from dash.dash_table.Format import Format, Symbol, Scheme  # Imported Scheme tool
from src.utils import db_info_utils

# Instantiate common config handler
common_config_handler = Config_handler("config-common.yml")

# Reading organism from DB
selected_organism = db_info_utils.get_db_organism()

# Instantiate dashboard config handler
dashboard_config_handler = Config_handler("config-dashboard.yml")

# Reading port from dashboard config file
PORT = dashboard_config_handler.read_property("degree_centrality.first_level_view.port")

# Set the root link for the redirect when selecting the node from the table for viewing the details
REDIRECT_ROOT = f"http://127.0.0.1:{dashboard_config_handler.read_property('degree_centrality.second_level_view.port')}/"

# Reading title from dashboard config file
TITLE = dashboard_config_handler.read_property("degree_centrality.first_level_view.title")

CSV_PATH = file_utils.get_name_with_organism(r"..\..\out\graph_analysis_out\degree_centrality.csv", selected_organism)

# Read your network data from the CSV file
df = pd.read_csv(CSV_PATH)

# Clean up the unnamed index column if it exists in the CSV
if df.columns[0].startswith('Unnamed') or df.columns[0] == '':
    df = df.drop(df.columns[0], axis=1)

# --- KPI Calculations ---
chemical_count = len(df[df['type'].str.lower() == 'chemical'])
protein_count = len(df[df['type'].str.lower() == 'protein'])
reaction_count = len(df[df['type'].str.lower() == 'reaction'])
total_count = len(df)

##### Bar plot style ###################################################################################################
df['degree'] = df['degree'].astype(int)
top_nodes = df.nlargest(15, 'degree')
fig_bar = px.bar(top_nodes,
                 x='name',
                 y=['inDegree', 'outDegree'],
                 title="Top 15 Most Connected Entities",
                 color_discrete_map={
                     'inDegree': '#2B8CBE',
                     'outDegree': '#FDB863'
                 },
                 labels={
                     'name': 'Biological Entity',
                     'value': 'Degree',
                     'variable': 'Degree Direction',
                 },
                 custom_data=['degree']
                 )

# Rename legend entries
fig_bar.for_each_trace(
    lambda t: t.update(
        name='In-Degree' if t.name == 'inDegree' else 'Out-Degree'
    )
)

# Custom tooltip for bar plot
fig_bar.for_each_trace(
    lambda t: t.update(
        hovertemplate=
        "<b>Degree direction</b>: " + t.name +
        "<br><b>Biological Entity</b>: %{x}" +
        "<br><b>Out-degree</b>: %{y}" +
        "<br><b>Degree</b>: %{customdata[0]}" +
        "<extra></extra>"
    )
)

# --- MOVE LEGEND ABOVE (Bar Chart) ---
fig_bar.update_layout(
    legend=dict(
        orientation="h",  # Horizontal orientation
        yanchor="bottom",  # Anchor the bottom of the legend box
        y=1.02,  # Push it just above the top plot boundary
        xanchor="center",  # Anchor the center of the legend box
        x=0.5  # Center it horizontally
    ),
    xaxis=dict(
        tickangle=-45,  # Tilt labels diagonally at -45 degrees
        tickmode='linear'  # Force Plotly to display every label
    )
)

##### Scatter plot style ###############################################################################################
fig_scatter = px.scatter(
    df,
    x='inDegree',
    y='outDegree',
    color='type',
    title="In-Degree vs Out-Degree Distribution",
    color_discrete_map={
        'chemical': '#0074D9',
        'protein': '#2ECC40',
        'reaction': '#FF851B'
    },
    labels={
        'inDegree': 'In-Degree',
        'outDegree': 'Out-Degree',
        'type': 'Entity Type'
    },
    hover_data={
        'name': True,
        'inDegree': True,
        'outDegree': True,
        'type': True
    }
)

# Custom tooltip for scatter plot
fig_scatter.update_traces(
    hovertemplate="<b>%{customdata[0]}</b><br>"
                  "Type: %{customdata[1]}<br>"
                  "In-Degree: %{x}<br>"
                  "Out-Degree: %{y}<extra></extra>"
)

# --- MOVE LEGEND ABOVE (Scatter Plot) ---
fig_scatter.update_layout(
    legend=dict(
        orientation="h",  # Horizontal orientation
        yanchor="bottom",  # Anchor the bottom of the legend box
        y=1.02,  # Push it just above the top plot boundary
        xanchor="center",  # Anchor the center of the legend box
        x=0.5  # Center it horizontally
    )
)

########################################################################################################################

# Construct a temporary column containing Markdown syntax for a clickable magnifying glass icon
df_table = df.copy()

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


# Helper logic to dynamically build URL request parameters from explicit table rows specified
def generate_inspect_link(row):
    # Extract only the explicit row properties requested for parameters
    params = {
        'id_node': row.get('id'),
        'name': row.get('name'),
        'inDegree': row.get('inDegree'),
        'outDegree': row.get('outDegree'),
        'type': row.get('type'),
        'degree': row.get('degree'),
        'degree_centrality': row.get('degree_centrality')
    }

    # Safely stringify and URL-encode the parameters string
    query_string = urllib.parse.urlencode(params)
    full_url = f"{REDIRECT_ROOT}?{query_string}"

    return f'<a href="{full_url}" target="_blank" title="click to display node details">' \
           f'<img src="{magnifying_glass_url}" alt="Explore" style="height:20px; width:20px; vertical-align:middle;"/>' \
           f'</a>'


# Inject explicit HTML strings into the column containing your mapped parameters row properties
df_table.insert(0, 'Explore', df_table.apply(generate_inspect_link, axis=1))

# Initialize the Dash app
app = Dash(__name__)

# Design the layout using Dash DataTable
app.layout = html.Div([
    html.H1(TITLE,
            style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'margin': '20px'}),

    # --- KPI Section Block ---
    html.Div([
        # Chemicals Card
        html.Div([
            html.H3("Chemicals", style={'margin': '0', 'fontSize': '16px', 'color': '#555'}),
            html.P(f"{chemical_count:,}",
                   style={'margin': '5px 0 0 0', 'fontSize': '28px', 'fontWeight': 'bold', 'color': '#0074D9'})
        ], style={'flex': '1', 'margin': '0 10px', 'padding': '15px', 'backgroundColor': '#fff', 'borderRadius': '8px',
                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'textAlign': 'center'}),

        # Proteins Card
        html.Div([
            html.H3("Proteins", style={'margin': '0', 'fontSize': '16px', 'color': '#555'}),
            html.P(f"{protein_count:,}",
                   style={'margin': '5px 0 0 0', 'fontSize': '28px', 'fontWeight': 'bold', 'color': '#2ECC40'})
        ], style={'flex': '1', 'margin': '0 10px', 'padding': '15px', 'backgroundColor': '#fff', 'borderRadius': '8px',
                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'textAlign': 'center'}),

        # Reactions Card
        html.Div([
            html.H3("Reactions", style={'margin': '0', 'fontSize': '16px', 'color': '#555'}),
            html.P(f"{reaction_count:,}",
                   style={'margin': '5px 0 0 0', 'fontSize': '28px', 'fontWeight': 'bold', 'color': '#FF851B'})
        ], style={'flex': '1', 'margin': '0 10px', 'padding': '15px', 'backgroundColor': '#fff', 'borderRadius': '8px',
                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'textAlign': 'center'}),

        # Total Entities Card
        html.Div([
            html.H3("Total Entities", style={'margin': '0', 'fontSize': '16px', 'color': '#555'}),
            html.P(f"{total_count:,}",
                   style={'margin': '5px 0 0 0', 'fontSize': '28px', 'fontWeight': 'bold', 'color': '#333'})
        ], style={'flex': '1', 'margin': '0 10px', 'padding': '15px', 'backgroundColor': '#f8f9fa',
                  'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.15)', 'border': '1px solid #ddd',
                  'textAlign': 'center'}),

        # Organism Card (Added to the far right)
        html.Div([
            html.H3("Organism", style={'margin': '0', 'fontSize': '16px', 'color': '#555'}),
            html.P(str(selected_organism),
                   style={'margin': '5px 0 0 0', 'fontSize': '28px', 'fontWeight': 'bold', 'color': '#e67e22'})
        ], style={'flex': '1', 'margin': '0 10px', 'padding': '15px', 'backgroundColor': '#fff',
                  'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.15)', 'border': '1px solid #ddd',
                  'textAlign': 'center'})

    ], style={'display': 'flex', 'justifyContent': 'space-between', 'width': '90%', 'margin': '0 auto 30px auto',
              'fontFamily': 'Arial, sans-serif'}),

    ########################################
    # Charts Section
    html.Div([
        html.Div([
            dcc.Graph(id='bar-chart', figure=fig_bar)
        ], style={'flex': '1', 'padding': '0 10px'}),

        # Scatter plot in-degree vs out-degree distribution
        html.Div([
            dcc.Graph(id='scatter-plot', figure=fig_scatter)
        ], style={'flex': '1', 'padding': '0 10px'}),
    ], style={'display': 'flex', 'width': '90%', 'margin': '0 auto 50px auto'}),

    ######################

    html.Div([
        dash_table.DataTable(
            data=df_table.to_dict('records'),
            columns=[
                {
                    "name": "",
                    "id": i,
                    "type": "text",
                    "presentation": "markdown",
                    "sortable": False,
                    "filterable": False
                } if i == 'Explore' else (
                    {
                        "name": i,
                        "id": i,
                        "type": "numeric",
                        # Sets fixed-point scheme decimal layout (.3f rounding strategy)
                        "format": Format(precision=3, scheme=Scheme.fixed).symbol(Symbol.yes).symbol_suffix('%')
                    } if i == 'degree_centrality' else {
                        "name": i,
                        "id": i,
                        "type": "numeric" if df_table[i].dtype in ["int64", "float64"] else "text"
                    }
                )
                for i in df_table.columns if i != 'id'
            ],
            sort_action="native",
            filter_action="native",
            page_action="native",
            filter_options={"case": "insensitive"},
            page_size=10,
            markdown_options={"html": True},
            style_table={'overflowX': 'auto', 'boxShadow': '0px 0px 15px rgba(0,0,0,0.1)'},
            style_cell={
                'fontFamily': 'Arial, sans-serif',
                'padding': '12px',
                'textAlign': 'left',
                'minWidth': '100px', 'backgroundColor': '#fafafa'
            },
            style_cell_conditional=[
                {
                    'if': {'column_id': 'Explore'},
                    'width': '50px',
                    'minWidth': '50px',
                    'maxWidth': '50px',
                    'textAlign': 'center'
                }
            ],
            style_header={
                'backgroundColor': '#1f77b4',
                'color': 'white',
                'fontWeight': 'bold',
                'textTransform': 'capitalize'
            },
            style_header_conditional=[{"if": {"column_id": "Explore"}, "pointerEvents": "none"}],
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f2f2f2',
                },
                {"if": {"column_id": "Explore"}, "textAlign": "center"}
            ],
            style_filter_conditional=[
                {"if": {"column_id": "Explore"}, "visibility": "hidden", "pointerEvents": "none"}],
            css=[{"selector": 'tr th[data-dash-column="Explore"] .sort', "rule": "display: none !important;"}]
        )
    ], style={'width': '90%', 'margin': '0 auto'})
])

# Run the local server
if __name__ == '__main__':
    app.run(debug=False, port=PORT)