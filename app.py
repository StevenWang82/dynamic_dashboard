import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from utils.data_processing import parse_upload_contents
from components.data_preview import generate_data_preview
from components.visualizations import (
    create_visualization_controls, 
    generate_plot,
    get_variable_type,
    get_chart_options
)
import pandas as pd
import io
from utils.date_processing import create_date_controls, convert_dates, generate_conversion_report

# 初始化 Dash 應用
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)

# 建立基本布局
app.layout = dbc.Container([
    # 標題列
    dbc.Row([
        dbc.Col(html.H1("數據分析儀表板", className="text-center mb-4"), width=12)
    ]),
    
    # 檔案上傳區域
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    '拖放或 ',
                    html.A('選擇檔案')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),
        ], width=12),
    ]),
    
    # 數據預覽區域
    dbc.Row([
        dbc.Col(id='output-data-upload', width=12)
    ]),

    # 視覺化分析區域
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("探索性數據分析", className="mt-4 mb-3"),
                html.Div(id='visualization-controls'),  # 視覺化控制面板
                html.Div(id='additional-controls'),     # 額外控制選項
                dcc.Graph(id='visualization-output')    # 圖表輸出區域
            ], id='analysis-section', style={'display': 'none'})  # 初始時隱藏
        ], width=12)
    ]),

    # 儲存數據的中間組件
    dcc.Store(id='stored-data')
], fluid=True)

@app.callback(
    [Output('output-data-upload', 'children'),
     Output('stored-data', 'data', allow_duplicate=True),  # 添加 allow_duplicate=True
     Output('analysis-section', 'style')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')],
    prevent_initial_call=True  # 防止初始觸發
)
def update_output(contents, filename):
    if contents is None:
        return html.Div(), None, {'display': 'none'}
    
    df, error = parse_upload_contents(contents, filename)
    if error:
        return html.Div([
            dbc.Alert(f"錯誤: {error}", color="danger")
        ]), None, {'display': 'none'}
    
    # 生成預覽
    preview = generate_data_preview(df)
    
    # 添加日期轉換控制元件
    date_controls = create_date_controls(df)
    
    # 將數據存儲為 JSON
    stored_data = df.to_json(date_format='iso', orient='split')
    
    return html.Div([
        preview,
        html.Hr(),
        date_controls
    ]), stored_data, {'display': 'block'}

@app.callback(
    Output('visualization-controls', 'children'),
    Input('stored-data', 'data')
)
def update_visualization_controls(stored_data):
    if stored_data is None:
        return html.Div()
    
    # 使用 StringIO 包裝 JSON 字符串
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    return create_visualization_controls(df)

@app.callback(
    Output('additional-controls', 'children'),
    [Input('visualization-controls', 'children'),  # 改為依賴 visualization-controls
     Input('stored-data', 'data')]
)
def update_additional_controls(viz_controls, stored_data):
    if not stored_data:
        return html.Div()
    
    try:
        # 使用 StringIO 包裝 JSON 字符串
        df = pd.read_json(io.StringIO(stored_data), orient='split')
        
        ctx = dash.callback_context
        if not ctx.triggered:
            return html.Div()
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'visualization-controls':
            return html.Div()  # 重置額外控制項
            
        return html.Div()  # 預設返回空的 Div
    
    except Exception as e:
        print(f"Error updating additional controls: {str(e)}")
        return html.Div()

# 添加控制次要變數啟用/禁用的回調
@app.callback(
    [Output('secondary-variable', 'disabled'),
     Output('chart-type', 'disabled'),
     Output('chart-type', 'options')],
    [Input('analysis-type', 'value'),
     Input('primary-variable', 'value'),
     Input('secondary-variable', 'value'),
     State('stored-data', 'data')]
)
def update_controls_state(analysis_type, primary_var, secondary_var, stored_data):
    if not stored_data or not analysis_type or not primary_var:
        return True, True, []
    
    try:
        df = pd.read_json(io.StringIO(stored_data), orient='split')
        
        # 獲取變數類型
        var_types = {
            'primary': get_variable_type(df, primary_var) if primary_var else None,
            'secondary': get_variable_type(df, secondary_var) if secondary_var else None
        }
        
        # 更新控制項狀態
        if analysis_type == 'univariate':
            chart_options = get_chart_options('univariate', var_types)
            return True, False, chart_options
        elif analysis_type == 'bivariate':
            if not secondary_var:
                return False, True, []
            chart_options = get_chart_options('bivariate', var_types)
            return False, False, chart_options
            
    except Exception as e:
        print(f"Error in update_controls_state: {str(e)}")
        
    return True, True, []



@app.callback(
    Output('visualization-output', 'figure'),
    [Input('analysis-type', 'value'),
     Input('primary-variable', 'value'),
     Input('secondary-variable', 'value'),
     Input('chart-type', 'value'),
     State('stored-data', 'data')]
)
def update_visualization(analysis_type, primary_var, secondary_var, chart_type, stored_data):
    # 創建一個空的圖表作為預設值
    empty_fig = {
        'data': [],
        'layout': {
            'title': '請選擇必要的變數和圖表類型',
            'annotations': [{
                'text': '請完成所有必要的選擇',
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'showarrow': False,
                'font': {'size': 20}
            }]
        }
    }
    
    # 檢查必要的選項是否已選擇
    if not all([stored_data, analysis_type, primary_var, chart_type]):
        return empty_fig
    
    # 對於雙變量分析，檢查是否選擇了次要變數
    if analysis_type == 'bivariate' and not secondary_var:
        return empty_fig
    
    try:
        # 解析數據
        df = pd.read_json(io.StringIO(stored_data), orient='split')
        
        # 檢查數據框
        if df.empty:
            return empty_fig
            
        # 生成圖表
        fig = generate_plot(
            df=df,
            analysis_type=analysis_type,
            primary_var=primary_var,
            chart_type=chart_type,
            secondary_var=secondary_var
        )
        
        # 如果沒有生成圖表，返回空圖表
        if not fig:
            return empty_fig
            
        return fig
    
    except Exception as e:
        print(f"Error in update_visualization: {str(e)}")
        return empty_fig

# 添加新的回調函數處理日期轉換
@app.callback(
    [Output('stored-data', 'data', allow_duplicate=True),  # 添加 allow_duplicate=True
     Output('date-conversion-status', 'children')],
    [Input('convert-dates-button', 'n_clicks')],
    [State('date-columns-checklist', 'value'),
     State('date-format-input', 'value'),
     State('stored-data', 'data')],
    prevent_initial_call=True  # 防止初始觸發
)
def process_date_conversion(n_clicks, date_columns, date_format, stored_data):
    if n_clicks is None or not date_columns:
        return dash.no_update, None  # 使用 dash.no_update 而不是 stored_data
        
    # 讀取數據
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    
    # 執行轉換
    df_converted, results = convert_dates(df, date_columns, date_format)
    
    # 生成報告
    report = generate_conversion_report(results)
    
    # 更新存儲的數據
    new_stored_data = df_converted.to_json(date_format='iso', orient='split')
    
    return new_stored_data, report

if __name__ == '__main__':
    app.run_server(debug=True)










