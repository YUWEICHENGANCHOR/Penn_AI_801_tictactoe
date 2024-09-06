import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Load the data
file_path = 'game_metrics.csv'
data = pd.read_csv(file_path)

# Filter data for board_size = 5
board_size_5_data = data[data['board_size'] == 5]

# Create a numeric representation of winners
board_size_5_data['winner_numeric'] = board_size_5_data['winner'].map({'Draw': 0, 'Agent X': 1, 'Agent O': 2})

# Initialize the Dash app
app = dash.Dash(__name__)

# Create the app layout
app.layout = html.Div(children=[
    html.H1(children='Game Metrics Dashboard (Board Size 5)'),

    html.Div(children='''
        Distribution of game metrics for games with board size 5.
    '''),

    # Game Time Distribution
    dcc.Graph(
        id='game-time-distribution',
        figure=px.histogram(
            board_size_5_data,
            x='game_time',
            nbins=20,
            title='Game Time Distribution (Board Size 5)'
        )
    ),

    # Winner Distribution
    dcc.Graph(
        id='winner-distribution',
        figure=px.histogram(
            board_size_5_data,
            x='winner',
            title='Winner Distribution (Board Size 5)'
        )
    ),

    # Average Agent Time
    dcc.Graph(
        id='average-agent-time',
        figure=px.bar(
            board_size_5_data.melt(id_vars=['agent_x_name'], value_vars=['agent_x_time', 'agent_o_time'],
                                   var_name='Agent', value_name='Time'),
            x='agent_x_name',
            y='Time',
            color='Agent',
            title='Average Agent Time (Board Size 5)'
        )
    ),

    # Game Outcomes Over Time
    dcc.Graph(
        id='game-outcomes-over-time',
        figure=go.Figure([
            go.Scatter(
                x=board_size_5_data.index,
                y=board_size_5_data['winner_numeric'],
                mode='markers',
                marker=dict(color='blue', size=5),
                name='Game Outcome'
            ),
            go.Scatter(
                x=board_size_5_data.index,
                y=board_size_5_data['winner_numeric'].rolling(window=20).mean(),
                mode='lines',
                line=dict(color='red'),
                name='Rolling Average (window=20)'
            )
        ]).update_layout(
            title='Game Outcomes Over Time (Board Size 5)',
            xaxis_title='Game Index',
            yaxis=dict(
                title='Outcome',
                tickvals=[0, 1, 2],
                ticktext=['Draw', 'Agent X', 'Agent O']
            )
        )
    )
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
