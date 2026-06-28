import pandas as pd
import plotly.express as px
from dash import Dash, html, dash_table, dcc, Input, Output
from src.utils import file_utils
from src.utils.config_handler import Config_handler
from src.utils import db_info_utils

# Reading organism from DB
selected_organism = db_info_utils.get_db_organism()

# Instantiate dashboard config handler
dashboard_config_handler = Config_handler("config-dashboard.yml")

# Reading title from dashboard config file
TITLE = dashboard_config_handler.read_property("connected_components.first_level_view.title")

# Reading port from dashboard config file
PORT = dashboard_config_handler.read_property("connected_components.first_level_view.port")

# Set the root link for the redirect when selecting the component for exploring the related communities
REDIRECT_ROOT = f"http://127.0.0.1:{dashboard_config_handler.read_property('connected_components.second_level_view.port')}/"

# Initialize Dash app
app = Dash(__name__)

# CSV path
CSV_PATH = file_utils.get_name_with_organism(r"..\..\out\graph_analysis_out\connected_components.csv", selected_organism)

# Load data
df = pd.read_csv(CSV_PATH)

# ---- Prepare data for bar chart & KPIs ----
total_components = df["component_id"].nunique()

df_counts = (
    df.groupby("component_id")
    .size()
    .reset_index(name="element_count")
    .sort_values(by="element_count", ascending=False)
)

# Get the size of the largest component
largest_component_size = df_counts["element_count"].max()

# Convert to string for categorical axis
df_counts["component_id"] = df_counts["component_id"].astype(str)

# Create bar chart
fig = px.bar(
    df_counts,
    x="component_id",
    y="element_count",
    color="component_id",
    title="Size of Connected Components (Log Scale) ordered<br>- click on the barplot to select the component for exploring the related communities -",
    labels={
        "component_id": "Component ID",
        "element_count": "Number of Elements",
    },
    template="plotly_white",
    log_y=True,
)

fig.update_traces(
    marker_line_color="#2980b9",
    marker_line_width=1.5
)

fig.update_layout(
    title_font_size=18,
    title_x=0.5,
    xaxis=dict(type="category", categoryorder="total descending"),
    showlegend=True
)

# ---- App layout ----
app.layout = html.Div(
    [
        # Hidden dummy div needed as a target for the clientside callback output
        html.Div(id="dummy-output", style={"display": "none"}),

        html.H1(
            TITLE,
            style={"textAlign": "center", "fontFamily": "Arial, sans-serif", "color": "#2c3e50", "marginTop": "20px"}
        ),

        # ---- KPI Cards Container ----
        html.Div(
            [
                # Card 1: Total Components
                html.Div(
                    [
                        html.H4(
                            "Total Components",
                            style={"margin": "0", "fontSize": "12px", "color": "#7f8c8d", "textTransform": "uppercase",
                                   "letterSpacing": "1px"}
                        ),
                        html.H2(
                            f"{total_components:,}",
                            style={"margin": "5px 0 0 0", "fontSize": "28px", "color": "#2c3e50", "fontWeight": "bold"}
                        )
                    ],
                    style={
                        "background": "#ffffff",
                        "padding": "15px 25px",
                        "borderRadius": "8px",
                        "boxShadow": "0 4px 15px rgba(0, 0, 0, 0.05)",
                        "borderLeft": "5px solid #1f77b4",
                        "minWidth": "200px"
                    }
                ),

                # Card 2: Largest Component Size
                html.Div(
                    [
                        html.H4(
                            "Largest Component Size",
                            style={"margin": "0", "fontSize": "12px", "color": "#7f8c8d", "textTransform": "uppercase",
                                   "letterSpacing": "1px"}
                        ),
                        html.H2(
                            f"{largest_component_size:,}",
                            style={"margin": "5px 0 0 0", "fontSize": "28px", "color": "#2c3e50", "fontWeight": "bold"}
                        )
                    ],
                    style={
                        "background": "#ffffff",
                        "padding": "15px 25px",
                        "borderRadius": "8px",
                        "boxShadow": "0 4px 15px rgba(0, 0, 0, 0.05)",
                        "borderLeft": "5px solid #2ecc71",
                        "minWidth": "200px"
                    }
                ),

                # Card 3: Selected Organism (Top-Right position inside container)
                html.Div(
                    [
                        html.H4(
                            "Organism",
                            style={"margin": "0", "fontSize": "12px", "color": "#7f8c8d", "textTransform": "uppercase",
                                   "letterSpacing": "1px"}
                        ),
                        html.H2(
                            str(selected_organism),
                            style={"margin": "5px 0 0 0", "fontSize": "28px", "color": "#2c3e50", "fontWeight": "bold"}
                        )
                    ],
                    style={
                        "background": "#ffffff",
                        "padding": "15px 25px",
                        "borderRadius": "8px",
                        "boxShadow": "0 4px 15px rgba(0, 0, 0, 0.05)",
                        "borderLeft": "5px solid #e67e22",
                        "minWidth": "200px"
                    }
                )
            ],
            style={
                "display": "flex",
                "justifyContent": "center",
                "gap": "20px",
                "marginBottom": "30px",
                "fontFamily": "Arial, sans-serif"
            }
        ),

        # Graph
        dcc.Graph(id="component-bar-chart", figure=fig),

        # Data table
        dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[
                {
                    "name": i,
                    "id": i,
                    "type": "numeric" if df[i].dtype in ["int64", "float64"] else "text"
                }
                for i in df.columns if i != "id"  # Filters out the 'id' column from visual generation
            ],

            sort_action="native",
            filter_action="native",
            page_action="native",
            filter_options={"case": "insensitive"},
            page_size=10,

            style_table={
                "overflowX": "auto",
                "boxShadow": "0px 0px 15px rgba(0,0,0,0.1)",
                "marginTop": "20px"
            },

            style_cell={
                "fontFamily": "Arial, sans-serif",
                "padding": "12px",
                "textAlign": "left",
                "minWidth": "100px",
                "backgroundColor": "#fafafa",
            },

            style_header={
                "backgroundColor": "#1f77b4",
                "color": "white",
                "fontWeight": "bold",
                "textTransform": "capitalize",
            },

            style_data_conditional=[
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": "#f2f2f2",
                }
            ],
        ),
    ],
    style={"width": "90%", "margin": "0 auto"},
)

# ---- Clientside Callback to open a new tab immediately on click ----
app.clientside_callback(
    """
    function(clickData) {
        if (clickData && clickData.points && clickData.points.length > 0) {
            // Extract the component_id from the clicked bar
            const componentId = clickData.points[0].x;

            // Generate the target URL using the injected Python variable
            const targetUrl = "PLACEHOLDER_REDIRECT_ROOT?component_id=" + componentId;

            // Open in a new tab ('_blank')
            window.open(targetUrl, '_blank');
        }
        return window.dash_clientside.no_update;
    }
    """.replace("PLACEHOLDER_REDIRECT_ROOT", REDIRECT_ROOT),
    Output("dummy-output", "children"),
    Input("component-bar-chart", "clickData"),
    prevent_initial_call=True
)

# Run server
if __name__ == "__main__":
    app.run(debug=False, port=PORT)