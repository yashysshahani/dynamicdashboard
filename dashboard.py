import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

file_path = 'Placement_Data_Full_Class.csv'
placement_data = pd.read_csv(file_path)
placement_data['salary'].dropna()

num_rows = len(placement_data)
num_columns = len(placement_data.columns)

placement_data['ssc_p'] = pd.to_numeric(placement_data['ssc_p'], errors='coerce')
placement_data['hsc_p'] = pd.to_numeric(placement_data['hsc_p'], errors='coerce')
placement_data['degree_p'] = pd.to_numeric(placement_data['degree_p'], errors='coerce')
placement_data['mba_p'] = pd.to_numeric(placement_data['mba_p'], errors='coerce')
placement_data = placement_data.dropna(subset=['ssc_p', 'hsc_p', 'degree_p', 'mba_p'])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.PULSE])

app.layout = html.Div([
    html.H1("Interactive EDA Dashboard", style={'textAlign': 'center', 'color': 'blue'}),

    html.Div([
        html.H3(f"Number of Rows: {num_rows}"),
        html.H3(f"Number of Columns: {num_columns}")],
        style={'textAlign': 'center', 'marginBottom': '20px'}),
    html.Br(),
        
    html.H3("Placement Data"),

    html.Div([
        dcc.Graph(
            id='placement-dist',
            figure=px.pie(placement_data, names='status'),
        )
    ]),

    html.H3([
        "Scores by Education Level"
    ]),

    dbc.Row([
        dbc.Col([
        dcc.Graph(id='education-graph')
        ], width=6),

        dbc.Col([
            dcc.Graph(
                id="density-salary",
                figure=px.histogram(
                    placement_data,
                    x='salary',
                    title="Salary Distribution",
                    labels={"salary": "Salary"}
                ).update_layout(
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False),
                    plot_bgcolor='white'
                )
            )
        ], width=6),
    ], style={'marginTop': '20px'}),

    html.Div([
        dcc.Dropdown(
                id='education-filter',
                options=[{'label': 'Secondary', 'value': 'ssc_p'},
                        {'label': 'Higher Secondary', 'value': 'hsc_p'},
                        {'label': 'Degree', 'value': "degree_p"},
                        {'label': 'MBA', 'value': 'mba_p'}],
                placeholder='Select Education Level',
                style={'marginBottom': '20px'}
        )
    ]),

    html.H3("Gender, Work Experience, Employability Test"),
    html.Div([
        dcc.Tabs(id="gender-work-employability", value="gender", children=[
            dcc.Tab(label="Gender Distribution", value="gender"),
            dcc.Tab(label="Work Experience", value="work-exp"),
            dcc.Tab(label="Employability Test", value="emp-test")
        ])
    ]),
    html.Div(id='gender-work-employability-graph'),

    html.H3("Board and Specialization Distribution"),
    html.Div([
        dcc.Tabs(
            id="board-specialization-tabs",
            value="ssc_b",  # Default tab value
            children=[
                dcc.Tab(label="Secondary Board", value="ssc_b"),
                dcc.Tab(label="Higher Secondary Board", value="hsc_b"),
                dcc.Tab(label="Higher Secondary Specialisation", value="hsc_s"),
                dcc.Tab(label="Degree Chosen", value="degree_t"),
                dcc.Tab(label="MBA Specialisation", value="specialisation"),
            ]
        )
    ]),
    html.Div(id="board-specialization-graph")
    ])


@app.callback(
    Output('education-graph', 'figure'),
    Input('education-filter', 'value')
)
def update_education_graph(selected_education):
    if not selected_education:
        empty_fig = go.Figure().update_layout(
            title="Please select an education metric",
            xaxis_title="Metric",
            yaxis_title="Count"
        )
        return empty_fig
    
    filtered_data = placement_data.dropna(subset=[selected_education])
    filtered_data[selected_education] = pd.to_numeric(filtered_data[selected_education], errors='coerce')
    
    titles = {
        'ssc_p': 'Secondary Education',
        'hsc_p': 'Higher Secondary',
        'degree_p': 'Degree',
        'mba_p': 'MBA'
    }

    graph_title = titles.get(selected_education, selected_education.replace('_', ' ').title())

    education_fig = px.histogram(
        data_frame=filtered_data,
        x=selected_education,
        title=f'Scores Distributed by {graph_title}',
        labels={selected_education: f'Score in {graph_title}', 'count': 'Count'}
    ).update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor='white'
    )

    return education_fig

@app.callback(
    Output('gender-work-employability-graph', 'children'),
    Input('gender-work-employability', 'value')
)
def gwe_content(tab):
    if tab == 'gender':
        fig = px.bar(
            data_frame=placement_data,
            x='gender',
            title="Gender Distribution",
            labels={'gender': 'Gender', 'y': 'Count'}
        )
        fig.update_layout(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor='white'
        )
    elif tab == 'work-exp':
        fig = px.bar(
            data_frame=placement_data,
            x='workex',
            title="Work Experience Distribution",
            labels={'workex': 'Work Experience', 'y': 'Count'}
        )
        fig.update_layout(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor='white'
        )
    elif tab == 'emp-test':
        fig = px.histogram(
            data_frame=placement_data,
            x='etest_p',
            title="Employability Test Scores",
            labels={'etest_p': 'Employability Test Score', 'y': 'Count'}
        )
        fig.update_layout(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor='white'
        )
    else:
        fig = go.Figure().update_layout(
            title="Please select a valid tab",
            xaxis_title="",
            yaxis_title=""
        )

    return dcc.Graph(figure=fig)

@app.callback(
    Output("board-specialization-graph", "children"),
    Input("board-specialization-tabs", "value")
)
def update_board_specialization_graph(selected_category):
    titles = {
        'ssc_b': 'Secondary Board',
        'hsc_b': 'Higher Secondary Board',
        'hsc_s': 'Higher Secondary Specialisation',
        'degree_t': 'Degree Chosen',
        'specialisation': 'MBA Specialisation'
    }

    graph_title = titles.get(selected_category, selected_category.replace('_', ' ').title())

    fig = px.bar(
        data_frame=placement_data,
        x=selected_category,
        title=f"Distribution of {graph_title}",
        labels={selected_category: graph_title, 'count': 'Count'}
    )
    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor="white"
    )
    return dcc.Graph(figure=fig)

if __name__ == '__main__':
    app.run_server(debug=True)
