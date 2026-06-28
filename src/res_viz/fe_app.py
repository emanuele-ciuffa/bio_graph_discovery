# pip install dash dash-bootstrap-components

import dash
from dash import html
import dash_bootstrap_components as dbc
from src.utils.config_handler import Config_handler
from src.utils import db_info_utils

# Reading organism from DB
selected_organism = db_info_utils.get_db_organism()

# Instantiate dashboard config handler
dashboard_config_handler = Config_handler("config-dashboard.yml")

# Reading title from dashboard config file for degree_centrality
degree_centrality_title = dashboard_config_handler.read_property("degree_centrality.first_level_view.title")

# Reading port from dashboard config file for degree_centrality
degree_centrality_port = dashboard_config_handler.read_property("degree_centrality.first_level_view.port")

# Set redirect url for degree_centrality
degree_centrality_url = f"http://127.0.0.1:{degree_centrality_port}"

# Reading title from dashboard config file for connected_components
connected_components_title = dashboard_config_handler.read_property("connected_components.first_level_view.title")

# Reading port from dashboard config file for connected_components
connected_components_port = dashboard_config_handler.read_property("connected_components.first_level_view.port")

# Set redirect url for connected components
connected_components_url = f"http://127.0.0.1:{connected_components_port}"

# Reading title from dashboard config file for shortest_path
shortest_path_title = dashboard_config_handler.read_property("shortest_path.first_level_view.title")

# Reading port from dashboard config file for shortest_path
shortest_path_port = dashboard_config_handler.read_property("shortest_path.first_level_view.port")

# Set redirect url for shortest_path
shortest_path_url = f"http://127.0.0.1:{shortest_path_port}"

# Reading title from dashboard config file for link_prediction
link_prediction_title = dashboard_config_handler.read_property("link_prediction.first_level_view.title")

# Reading port from dashboard config file for link_prediction
link_prediction_port = dashboard_config_handler.read_property("link_prediction.first_level_view.port")

# Set redirect url for connected components
link_prediction_url = f"http://127.0.0.1:{link_prediction_port}"

# Initialize the Dash app with a clean Bootstrap theme (LUX offers a modern look)
app = dash.Dash(
    __name__,
    assets_folder='img',
    external_stylesheets=[dbc.themes.LUX],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

app.title = "Graph Analysis for Chemical-Protein Interaction Discovery"

# Configuration for your dashboard links with your requested color map
DASHBOARDS = [
    {
        "title": degree_centrality_title,
        "url": degree_centrality_url,
        "description": "Analyze node importance based on the number of direct connections. Drill down to explore the node details.",
        "color": None,
        "image": "degree_centrality.png",
        "custom_bg": "#4fc47f"  # Green
    },
    {
        "title": connected_components_title,
        "url": connected_components_url,
        "description": "Discover isolated subgraphs and network connectivity structures. Drill down into any component to explore its internal communities and functional clusters.",
        "color": None,
        "image": "connected_components.png",
        "custom_bg": "#3f80dc"  # Blue
    },
    {
        "title": shortest_path_title,
        "url": shortest_path_url,
        "description": "Visualize the most efficient routes between nodes. Drill down to display the selected path.",
        "color": None,
        "image": "shortest_path.png",
        "custom_bg": "#f4b053"  # Orange
    },
    {
        "title": link_prediction_title,
        "url": link_prediction_url,
        "description": "Visualize the predicted links based on network topology. Drill down to display the selected predicted link.",
        "color": None,
        "image": "link_prediction.png",
        "custom_bg": "#9b51e0"  # Purple
    }
]

# Build the layout using a responsive grid
app.layout = dbc.Container([
    # Header Section
    dbc.Row([
        dbc.Col(
            html.Div([
                # Left Side: Single line Title Configuration and Subtext
                html.Div([
                    html.H1(
                        "Chemical-Protein Interaction Discovery",
                        className="mt-5 mb-2 font-weight-bold",
                        style={
                            "color": "#1e4620",
                            "fontSize": "2.2rem",
                            "whiteSpace": "nowrap"
                        }
                    ),
                    html.P([
                        "Select a specialized dashboard below to analyze network metrics.",
                        html.Br(),
                        "Available data for the organism: ",
                        html.Strong(selected_organism, style={"color": "#007bff"})
                    ], className="lead text-muted mb-0"),
                ], style={"flex": "1", "marginRight": "20px", "overflow": "hidden"}),

                # Right Side: Logo aligned with the title block
                html.Div(
                    html.Img(
                        src=app.get_asset_url('univaq_logo.gif'),
                        style={
                            "maxHeight": "110px",
                            "maxWidth": "100%",
                            "objectFit": "contain"
                        }
                    ),
                    className="mt-5"
                )
            ], className="d-flex justify-content-between align-items-start flex-column flex-md-row"),
            width=12
        ),

        # Divider line below the header elements
        dbc.Col(html.Hr(className="my-4"), width=12)
    ]),

    # Grid of Dashboard Cards
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H4(db["title"], className="card-title font-weight-bold"),
                    html.P(db["description"], className="card-text text-secondary mb-4"),

                    # The Button (perfectly aligned, size-matched, color-customized)
                    dbc.Button(
                        f"Open {db['title']}",
                        href=db["url"],
                        target="_blank",
                        color=db["color"],
                        className="w-100 mt-auto font-weight-bold d-flex align-items-center justify-content-center",
                        style={
                            "height": "65px",
                            "lineHeight": "1.2",
                            "fontSize": "0.95rem",
                            "backgroundColor": db["custom_bg"],
                            "borderColor": db["custom_bg"],
                            "color": "white"
                        }
                    ),

                    # The functional icon, perfectly centered and scaled, placed below the button
                    html.Div(
                        html.Img(
                            src=app.get_asset_url(db["image"]),
                            style={
                                "maxHeight": "150px",
                                "width": "100%",
                                "objectFit": "contain",
                                "marginTop": "20px"
                            }
                        ),
                        className="text-center w-100"
                    )

                ], className="d-flex flex-column align-items-center")  # Center the elements
            ], className="shadow-sm h-100 transition-hover"),
            xs=12, sm=6, md=6, lg=3, className="mb-4"
        ) for db in DASHBOARDS
    ], className="justify-content-center")
], fluid=False)

if __name__ == "__main__":
    # Runs the main page on default port 8050
    app.run(debug=False, port=8050)