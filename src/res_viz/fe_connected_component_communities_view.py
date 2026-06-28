import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table, Input, Output
from urllib.parse import parse_qs
from src.utils import file_utils
from src.utils.config_handler import Config_handler

from src.utils import db_info_utils

# Reading organism from DB
selected_organism = db_info_utils.get_db_organism()

# Instantiate dashboard config handler
dashboard_config_handler = Config_handler("config-dashboard.yml")

# Reading title from dashboard config file
TITLE = dashboard_config_handler.read_property("connected_components.second_level_view.title")

# Reading port from dashboard config file
PORT = dashboard_config_handler.read_property("connected_components.second_level_view.port")

# 1. Initialize the Dash app
app = Dash(__name__)

# 2. Define your static CSV path
CSV_PATH = file_utils.get_name_with_organism(r"..\..\out\graph_analysis_out\communities.csv", selected_organism)

# Global data loading
try:
    df_raw = pd.read_csv(CSV_PATH)
except Exception:
    # Fallback mockup simulating your real CSV structure with missing community entries
    print("Error")

# CLEANING DATA: Handle missing values and convert large IDs safely to clean strings
if 'community' in df_raw.columns:
    # Fill NaN values with a readable placeholder
    df_raw['community'] = df_raw['community'].fillna("Unassigned")
    # Clean up floating point strings like '1234.0' back to '1234'
    df_raw['community'] = df_raw['community'].apply(lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x))

# --- Color Palette Configuration ---
colors_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24


# -------------------------------------------------------------------------
# Layout Matrix (Empty Placeholders for Callbacks)
# -------------------------------------------------------------------------
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    # Top-Centered Page Title
    html.H1(id='page-title', children="Communities", style={'fontFamily': 'sans-serif', 'marginBottom': '25px', 'textAlign': 'center'}),

    # ---- KPI Cards Container (Centered and Clustered) ----
    html.Div([
        # Card 1: Related Component Name
        html.Div([
            html.H4("Related Component", style={"margin": "0", "fontSize": "12px", "color": "#7f8c8d", "textTransform": "uppercase", "letterSpacing": "1px"}),
            html.H2(id="kpi-component-name", style={"margin": "5px 0 0 0", "fontSize": "24px", "color": "#2c3e50", "fontWeight": "bold"})
        ], style={"background": "#ffffff", "padding": "15px 25px", "borderRadius": "8px", "boxShadow": "0 4px 15px rgba(0, 0, 0, 0.05)", "borderLeft": "5px solid #1f77b4", "minWidth": "220px"}),

        # Card 2: Number of Communities
        html.Div([
            html.H4("Communities Count", style={"margin": "0", "fontSize": "12px", "color": "#7f8c8d", "textTransform": "uppercase", "letterSpacing": "1px"}),
            html.H2(id="kpi-communities-count", style={"margin": "5px 0 0 0", "fontSize": "28px", "color": "#2c3e50", "fontWeight": "bold"})
        ], style={"background": "#ffffff", "padding": "15px 25px", "borderRadius": "8px", "boxShadow": "0 4px 15px rgba(0, 0, 0, 0.05)", "borderLeft": "5px solid #2ecc71", "minWidth": "220px"}),

        # Card 3: Selected Organism (Brought close to center)
        html.Div([
            html.H4("Organism", style={"margin": "0", "fontSize": "12px", "color": "#7f8c8d", "textTransform": "uppercase", "letterSpacing": "1px"}),
            html.H2(str(selected_organism), style={"margin": "5px 0 0 0", "fontSize": "28px", "color": "#2c3e50", "fontWeight": "bold"})
        ], style={"background": "#ffffff", "padding": "15px 25px", "borderRadius": "8px", "boxShadow": "0 4px 15px rgba(0, 0, 0, 0.05)", "borderLeft": "5px solid #e67e22", "minWidth": "220px"})
    ], style={"display": "flex", "justifyContent": "center", "gap": "20px", "marginBottom": "30px", "fontFamily": "Arial, sans-serif"}),

    # Bar Chart Container
    html.Div([
        dcc.Graph(id='bar-plot')
    ], style={'marginBottom': '25px', 'boxShadow': '0px 0px 15px rgba(0,0,0,0.05)', 'padding': '15px',
              'backgroundColor': '#fff', 'borderRadius': '8px'}),

    # Middle Grid Panel Layout
    html.Div([
        # Left Panel - Plotly Scatter Chart
        html.Div([
            dcc.Graph(id='scatter-plot', config={'displayModeBar': True})
        ], style={'width': '73%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Right Panel - Structural Community Table
        html.Div(id='summary-table-container', style={'width': '24%', 'display': 'inline-block', 'float': 'right', 'paddingLeft': '15px'})

    ], style={'marginBottom': '40px', 'boxShadow': '0px 0px 15px rgba(0,0,0,0.05)', 'padding': '20px',
              'backgroundColor': '#fff', 'borderRadius': '8px', 'overflow': 'hidden'}),

    # Detailed Data Table Title & Element
    html.H2("Nodes for community", style={
        'fontFamily': 'sans-serif',
        'color': '#2c3e50',
        'marginBottom': '12px',
        'fontSize': '20px'
    }),

    dash_table.DataTable(
        id='nodes-table',
        sort_action="native",
        filter_action="native",
        page_action="native",
        filter_options={"case": "insensitive"},
        page_size=20,
        style_table={
            'overflowX': 'auto',
            'boxShadow': '0px 0px 15px rgba(0,0,0,0.1)'
        },
        style_cell={
            'fontFamily': 'Arial, sans-serif',
            'padding': '12px',
            'textAlign': 'left',
            'minWidth': '100px',
            'backgroundColor': '#fafafa'
        },
        style_header={
            'backgroundColor': '#1f77b4',
            'color': 'white',
            'fontWeight': 'bold',
            'textTransform': 'capitalize'
        },
        style_data={
            'border': '1px solid #e4e4e7'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f2f2f2',
            }
        ]
    )
], style={'width': '90%', 'margin': '0 auto', 'paddingTop': '20px', 'paddingBottom': '40px'})


# -------------------------------------------------------------------------
# DYNAMIC URL PARAMETER CALLBACK
# -------------------------------------------------------------------------
@app.callback(
    [
        Output('page-title', 'children'),
        Output('bar-plot', 'figure'),
        Output('scatter-plot', 'figure'),
        Output('summary-table-container', 'children'),
        Output('nodes-table', 'columns'),
        Output('nodes-table', 'data'),
        Output('kpi-component-name', 'children'),
        Output('kpi-communities-count', 'children')
    ],
    [Input('url', 'search')]
)
def update_dashboard(search_string):
    component_id = None
    if search_string:
        parsed_params = parse_qs(search_string.lstrip('?'))
        if 'component_id' in parsed_params:
            try:
                component_id = int(parsed_params['component_id'][0])
            except ValueError:
                component_id = parsed_params['component_id'][0]

    # Filter data dynamically
    if component_id is not None and 'component_id' in df_raw.columns:
        df_filtered = df_raw[df_raw['component_id'] == component_id].copy()
        title_text = f"{TITLE}"
        kpi_comp_text = component_id
    else:
        df_filtered = df_raw.copy()
        title_text = f"{TITLE} - All Components"
        kpi_comp_text = "All Components"

    # Calculate total unique communities for the active filtered component selection
    num_communities = df_filtered['community'].nunique() if 'community' in df_filtered.columns else 0

    df_display = df_filtered[['id', 'name', 'type', 'community']].copy()
    total_nodes = len(df_display)

    # Re-calculate value distribution safely using string indices
    comm_counts = df_display['community'].value_counts().to_dict()
    sorted_comms = sorted(comm_counts.items(), key=lambda x: x[1], reverse=True)

    TOP_N = 15
    top_comms_data = sorted_comms[:TOP_N]
    remaining_comms_data = sorted_comms[TOP_N:]
    others_count = sum(count for _, count in remaining_comms_data)

    top_community_ids = [comm_id for comm_id, _ in top_comms_data]
    community_colors = {comm_id: colors_palette[idx % len(colors_palette)] for idx, comm_id in enumerate(top_community_ids)}
    community_colors['Others'] = '#a1a1aa'
    community_colors['Unassigned'] = '#cbd5e1' # Light grey accent for unassigned elements

    # -- Bar Chart Plotting --
    bar_x = [f"Comm {comm_id}" if comm_id != "Unassigned" else "Unassigned" for comm_id, _ in top_comms_data]
    bar_y = [count for _, count in top_comms_data]
    bar_marker_colors = [community_colors[comm_id] for comm_id, _ in top_comms_data]

    if others_count > 0:
        bar_x.append("Others")
        bar_y.append(others_count)
        bar_marker_colors.append(community_colors['Others'])

    bar_fig = go.Figure(data=[
        go.Bar(
            x=bar_x, y=bar_y,
            marker_color=bar_marker_colors,
            text=bar_y, textposition='auto',
            hovertemplate="<b>%{x}</b><br>Size: %{y} nodes<extra></extra>"
        )
    ])
    bar_fig.update_layout(
        title=dict(text=f"Node Count Distribution (Top {TOP_N} & Others - Log Scale)", font=dict(family="Arial, sans-serif", size=16, color="#2c3e50")),
        xaxis=dict(title="Community Identifier", type='category'),
        yaxis=dict(title="Total Node Count (Log Scale)", type='log', showgrid=True, gridcolor="#e2e8f0"),
        plot_bgcolor="white", margin=dict(l=60, r=20, t=50, b=50), height=330
    )

    # -- Proportional Scatter Architecture Setup --
    np.random.seed(42)
    scatter_fig = go.Figure()
    TARGET_TOTAL_POINTS = 2000

    for i, comm_id in enumerate(top_community_ids):
        actual_count = comm_counts.get(comm_id, 0)
        proportion = (actual_count / total_nodes) if total_nodes > 0 else 0
        allocated_points = max(15, int(TARGET_TOTAL_POINTS * proportion))

        center_x = (i % 4) * 3.5 + np.random.uniform(-0.2, 0.2)
        center_y = (i // 4) * 3.0 + np.random.uniform(-0.2, 0.2)
        cluster_spread = 0.35 + (proportion * 0.4)

        x_comm = np.random.normal(loc=center_x, scale=cluster_spread, size=allocated_points)
        y_comm = np.random.normal(loc=center_y, scale=cluster_spread, size=allocated_points)

        label_name = f"Community {comm_id}" if comm_id != "Unassigned" else "Unassigned Nodes"

        scatter_fig.add_trace(go.Scatter(
            x=x_comm, y=y_comm, mode='markers', name=label_name,
            text=[f"{label_name}<br>Real Size: {actual_count} nodes ({proportion * 100:.1f}%)" for _ in range(allocated_points)],
            hoverinfo="text",
            marker=dict(color=community_colors.get(comm_id), size=8, line=dict(width=0.4, color='rgba(255,255,255,0.6)'))
        ))
    scatter_fig.update_layout(
        title=dict(text=f"Simulated Architecture: Top {TOP_N} Communities", font=dict(family="Arial, sans-serif", size=16, color="#2c3e50")),
        xaxis=dict(title="Spatial Cluster Alignment (X)", showgrid=True, gridcolor="#e2e8f0", zeroline=False),
        yaxis=dict(title="Spatial Cluster Alignment (Y)", showgrid=True, gridcolor="#e2e8f0", zeroline=False),
        legend=dict(title=dict(text="Graph Clusters")),
        plot_bgcolor="white", margin=dict(l=60, r=20, t=50, b=50), hovermode="closest", height=450
    )

    # -- Sidebar Summary Generating Matrix --
    summary_rows = []
    displayed_limit = 5
    table_other_count = 0

    for idx, (comm_id, count) in enumerate(sorted_comms):
        rate = (count / total_nodes) * 100 if total_nodes > 0 else 0
        if idx < displayed_limit:
            label_text = f"Community {comm_id}" if comm_id != "Unassigned" else "Unassigned Nodes"
            summary_rows.append({"label": label_text, "count": count, "rate": f"{rate:.1f}%"})
        else:
            table_other_count += count

    if table_other_count > 0:
        table_other_rate = (table_other_count / total_nodes) * 100 if total_nodes > 0 else 0
        summary_rows.append({"label": "Other Communities", "count": table_other_count, "rate": f"{table_other_rate:.1f}%"})

    summary_table_element = html.Table([
        html.Thead(
            html.Tr([
                html.Th("Community", style={'textAlign': 'left', 'padding': '8px', 'borderBottom': '2px solid #ddd'}),
                html.Th("Count", style={'textAlign': 'right', 'padding': '8px', 'borderBottom': '2px solid #ddd'}),
                html.Th("Rate (%)", style={'textAlign': 'right', 'padding': '8px', 'borderBottom': '2px solid #ddd'}),
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(row["label"], style={'padding': '10px 8px', 'borderBottom': '1px solid #eee', 'fontWeight': 'bold' if any(x in row["label"] for x in ["Other", "Unassigned"]) else 'normal'}),
                html.Td(f"{row['count']}", style={'textAlign': 'right', 'padding': '10px 8px', 'borderBottom': '1px solid #eee'}),
                html.Td(row["rate"], style={'textAlign': 'right', 'padding': '10px 8px', 'borderBottom': '1px solid #eee'})
            ]) for row in summary_rows
        ])
    ], style={'width': '100%', 'fontFamily': 'Arial, sans-serif', 'fontSize': '13px', 'borderCollapse': 'collapse', 'marginTop': '45px', 'backgroundColor': '#f8fafc'})

    # -- Table Columns Processing --
    dt_columns = [
        {
            "name": i.capitalize(),
            "id": i,
            "type": "text",
            "filter_options": {"search": "default"}
        } for i in df_display.columns if i != 'id'  # Filters out the 'id' column from visual presentation
    ]
    dt_data = df_display.to_dict('records')

    return title_text, bar_fig, scatter_fig, summary_table_element, dt_columns, dt_data, kpi_comp_text, f"{num_communities:,}"


if __name__ == '__main__':
    app.run(debug=False, port=PORT)