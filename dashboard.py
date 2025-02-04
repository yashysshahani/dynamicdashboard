import dash
from dash import dcc, html, Input, Output, State, dash_table, no_update
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import base64
import io

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

dataframe = pd.DataFrame()

def get_dtype(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    elif pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    elif pd.api.types.is_categorical_dtype(series) or (series.dtype == object):
        return 'categorical'
    else:
        return 'unknown'
    
def guess_chart_type(x_dtype: str, y_dtypes: list[str], multiple_y: bool) -> str:
    if multiple_y:
        if x_dtype in ['numeric', 'datetime']:
            if x_dtype == 'datetime':
                return 'line'
            else:
                return 'scatter'
        elif x_dtype == 'categorical':
            return 'bar'
        else:
            return 'scatter'
    y_dtype = y_dtypes[0]
    if x_dtype == 'datetime' and y_dtype == 'numeric':
        return 'line'
    if x_dtype == 'numeric' and y_dtype == 'numeric':
        return 'scatter'
    if x_dtype == 'numeric' and y_dtype == 'categorical':
        return 'bar'
    if x_dtype == 'categorical' and y_dtype == 'numeric':
        return 'bar'
    if x_dtype == 'categorical' and y_dtype == 'categorical':
        return 'heatmap'
    return 'scatter'

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Dashboard", className="text-center text-primary mb-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and Drop or ', html.A('Select a File')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center'
                }
            ),
            html.Div(id='file-info', className="mt-3")
        ], width=12)
    ], className="mb-4"),

    dcc.Store(id='uploaded-data'),

    dcc.Store(id='filtered-data'),

    dbc.Row([
        dbc.Col([
            dbc.Label("Filter Column"),
            dcc.Dropdown(id="filter-column-dropdown", placeholder='Select column')
        ], width=3),
        dbc.Col([
            dbc.Label("Filter Value"),
            dcc.Input(id='filter-value', type='text', placeholder='Value or substring', style={'width': '100%'})
        ], width=3),
        dbc.Col([
            dbc.Button("Apply Filter", id='filter-button', color='primary', className='mt-4')
        ], width=2)
    ], className='mb-4'),

    dbc.Row([
        dbc.Col([
            html.H5("Summary of Data", className='text-info mb-3'),
            html.Div(id='data-summary')
        ], width=12)
    ], className='mb-4'),

    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='attribute-dropdown', placeholder='Select an Attribute')
        ], width=4),
        dbc.Col([
            dcc.Dropdown(id='x-axis-dropdown', placeholder='Select X-Axis')
        ], width=4),
        dbc.Col([
            dcc.Dropdown(id='y-axis-dropdown', placeholder='Select Y-Axis (multi)', multi=True)
        ], width=4),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='chart-type-dropdown',
                options=[
                    {'label': 'Auto (default)', 'value': ''},
                    {'label': 'Scatter', 'value': 'scatter'},
                    {'label': 'Line', 'value': 'line'},
                    {'label': 'Bar', 'value': 'bar'},
                    {'label': 'Box', 'value': 'box'},
                    {'label': 'Violin', 'value': 'violin'},
                    {'label': 'Histogram', 'value': 'histogram'},
                    {'label': 'Heatmap (crosstab)', 'value': 'heatmap'}
                ],
                value='',
                placeholder='Select Chart Type'
            )
        ], width=6),
        dbc.Col([
            dcc.Checklist(
                id='trendline-toggle',
                options=[{'label': 'Show Trend Line', 'value': 'trend'}],
                value=[],
                className="mt-2"
            )
        ], width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([dcc.Graph(id='data-chart')], width=6),
        dbc.Col([dcc.Graph(id='custom-chart')], width=6)
    ])
], fluid=True)

@app.callback(
    [
        Output('file-info', 'children'),
        Output('attribute-dropdown', 'options'),
        Output('x-axis-dropdown', 'options'),
        Output('y-axis-dropdown', 'options'),
        Output('uploaded-data', 'data'),
        Output('filter-column-dropdown', 'options')
    ],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def upload_file(contents, filename):
    if not contents:
        return (
            html.H4("No file uploaded yet.", className="text-muted"),
            [], [], [], None, []
        )
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'csv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            raise ValueError("Unsupported file type")

        options = [{'label': col, 'value': col} for col in df.columns]
        return (
            html.H4(f"Uploaded File: {filename}", className="text-success"),
            options,
            options,
            options,
            df.to_dict('records'),
            options
        )
    except Exception as e:
        print(f"Error processing file: {e}")
        return (
            html.H4("Error reading file.", className="text-danger"),
            [], [], [], None, []
        )

@app.callback(
    Output('filtered-data', 'data'),
    Input('filter-button', 'n_clicks'),
    State('uploaded-data', 'data'),
    State('filter-column-dropdown', 'value'),
    State('filter-value', 'value'),
    prevent_initial_call=True
)
def apply_filter(n_clicks, raw_data, col, val):
    if not raw_data:
        return None
    df = pd.DataFrame(raw_data)
    if not col or not val:
        return df.to_dict('records')
    dtype = get_dtype(df[col])
    if dtype == 'numeric':
        try:
            threshold = float(val)
            df = df[df[col] >= threshold]
        except ValueError:
            pass
    else:
        df = df[df[col].astype(str).str.contains(val, case=False, na=False)]
    return df.to_dict('records')

@app.callback(
    Output('data-summary', 'children'),
    Input('filtered-data', 'data')
)
def update_summary(filtered_data):
    if not filtered_data:
        return html.Div("No data available.", className="text-muted")
    df = pd.DataFrame(filtered_data)
    if df.empty:
        return html.Div("No rows match the current filter.", className="text-warning")
    numeric_cols = df.select_dtypes(include='number').columns
    if numeric_cols.empty:
        return html.Div("No numeric columns to summarize.", className="text-info")
    desc = df[numeric_cols].describe()
    desc = desc.reset_index().rename(columns={'index': 'stat'})
    return dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in desc.columns],
        data=desc.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
        page_size=10
    )

@app.callback(
    Output('data-chart', 'figure'),
    Input('attribute-dropdown', 'value'),
    Input('filtered-data', 'data')
)
def data_distributions(value, filtered_data):
    if not filtered_data:
        return go.Figure().update_layout(title="No data to display")
    df = pd.DataFrame(filtered_data)
    if not value or value not in df.columns:
        return go.Figure().update_layout(title="Please select an attribute")
    data = df[value].dropna()
    if data.empty:
        return go.Figure().update_layout(title=f"No data in {value} after filter")
    if pd.api.types.is_numeric_dtype(data):
        fig = px.histogram(df, x=value, title=f"Distribution of {value}")
    else:
        fig = px.pie(df, names=value, title=f"Distribution of {value}")
    fig.update_layout(colorway=px.colors.qualitative.Plotly)
    return fig

@app.callback(
    Output('custom-chart', 'figure'),
    [
        Input('x-axis-dropdown', 'value'),
        Input('y-axis-dropdown', 'value'),
        Input('chart-type-dropdown', 'value'),
        Input('trendline-toggle', 'value'),
        Input('filtered-data', 'data')
    ]
)
def custom_chart(x, y_list, chart_type, trendline_toggle, filtered_data):
    if not filtered_data:
        return go.Figure().update_layout(title="No data to display")
    df = pd.DataFrame(filtered_data)
    if not x or not y_list:
        return go.Figure().update_layout(title="Please select both X and Y axes")
    if isinstance(y_list, str):
        y_list = [y_list]
    x_dtype = get_dtype(df[x])
    y_dtypes = [get_dtype(df[y]) for y in y_list]
    if not chart_type:
        chart_type = guess_chart_type(x_dtype, y_dtypes, multiple_y=(len(y_list) > 1))
    if chart_type == 'heatmap':
        if x_dtype == 'categorical' and all(d == 'categorical' for d in y_dtypes) and len(y_list) == 1:
            crosstab = pd.crosstab(df[x], df[y_list[0]])
            fig = px.imshow(crosstab, text_auto=True, aspect='auto')
            fig.update_layout(title=f"Heatmap of {x} vs. {y_list[0]}")
            return fig
        else:
            return go.Figure().update_layout(
                title="Heatmap requires a single categorical Y and categorical X"
            )
    fig = go.Figure()
    show_trendline = ('trend' in trendline_toggle)
    for y in y_list:
        if df[x].dropna().empty or df[y].dropna().empty:
            continue
        if chart_type == 'scatter':
            fig.add_trace(go.Scatter(x=df[x], y=df[y], mode='markers', name=y))
        elif chart_type == 'line':
            fig.add_trace(go.Scatter(x=df[x], y=df[y], mode='lines', name=y))
        elif chart_type == 'bar':
            fig.add_trace(go.Bar(x=df[x], y=df[y], name=y))
        elif chart_type == 'box':
            fig.add_trace(go.Box(x=df[x], y=df[y], name=y))
        elif chart_type == 'violin':
            fig.add_trace(go.Violin(x=df[x], y=df[y], box_visible=True, meanline_visible=True, name=y))
        elif chart_type == 'histogram':
            fig.add_trace(go.Histogram(x=df[x], name=y))
        else:
            fig.add_trace(go.Scatter(x=df[x], y=df[y], mode='markers', name=y))
        if show_trendline:
            if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
                valid_df = df[[x, y]].dropna()
                xs = valid_df[x].values
                ys = valid_df[y].values
                if len(xs) > 1:
                    m, b = np.polyfit(xs, ys, 1)
                    x_fit = np.linspace(xs.min(), xs.max(), 50)
                    y_fit = m * x_fit + b
                    fig.add_trace(
                        go.Scatter(
                            x=x_fit,
                            y=y_fit,
                            mode='lines',
                            name=f"{y} Trend",
                            line=dict(dash='dash')
                        )
                    )
    fig.update_layout(
        title=f"Custom Chart: {chart_type}",
        xaxis_title=x,
        yaxis_title=", ".join(y_list),
        barmode="group",
        colorway=px.colors.qualitative.Plotly
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
