from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.nonparametric.kde import KDEUnivariate
import io

def get_variable_type(df, column):
    """判斷變數類型"""
    if pd.api.types.is_numeric_dtype(df[column]):
        return 'numeric'
    return 'categorical'

def get_chart_options(analysis_type, var_types):
    """根據分析類型和變量類型返回可用的圖表選項"""
    univariate_charts = {
        'numeric': [
            {'label': '直方圖+KDE', 'value': 'histogram_kde'}
        ],
        'categorical': [
            {'label': '柱狀圖', 'value': 'bar'},
            {'label': '棒棒糖圖', 'value': 'lollipop'},
            {'label': '圓餅圖', 'value': 'pie'},
            {'label': '甜甜圈圖', 'value': 'doughnut'},
            {'label': '樹狀圖', 'value': 'treemap'}
        ]
    }
    
    bivariate_charts = {
        'categorical_categorical': [
            {'label': '熱力圖', 'value': 'heatmap'},
            {'label': '堆疊長條圖', 'value': 'stacked_bar'},
            {'label': '並排長條圖', 'value': 'side_by_side_bar'},
            {'label': '堆疊百分比直方圖', 'value': 'stacked_histogram_percentage'}
        ],
        'numeric_numeric': [
            {'label': '散佈圖', 'value': 'scatter'}
        ],
        'categorical_numeric': [
            {'label': '箱形圖', 'value': 'box'},
            {'label': '小提琴圖', 'value': 'violin'},
            {'label': '分組散點圖', 'value': 'grouped_scatter'}
        ],
        'numeric_categorical': [
            {'label': '箱形圖', 'value': 'box'},
            {'label': '小提琴圖', 'value': 'violin'},
            {'label': '分組散點圖', 'value': 'grouped_scatter'}
        ]
    }
    
    if analysis_type == 'univariate':
        return univariate_charts[var_types['primary']]
    elif analysis_type == 'bivariate':
        combo_key = f"{var_types['primary']}_{var_types['secondary']}"
        return bivariate_charts.get(combo_key, [])
    return []

def create_visualization_controls(df):
    """創建視覺化控制面板"""
    if df is None or df.empty:
        return html.Div()
    
    # 預先計算每個欄位的類型
    column_types = {col: get_variable_type(df, col) for col in df.columns}
    
    # 為每個欄位添加類型標籤
    column_options = [
        {'label': f"{col} ({column_types[col]})", 'value': col}
        for col in df.columns
    ]
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("分析類型"),
                dcc.Dropdown(
                    id='analysis-type',
                    options=[
                        {'label': '單變量分析', 'value': 'univariate'},
                        {'label': '雙變量分析', 'value': 'bivariate'}
                    ],
                    value=None,
                    placeholder="請選擇分析類型",
                    clearable=False
                )
            ], width=12, className='mb-3'),
            
            dbc.Col([
                html.Label("主要變數"),
                dcc.Dropdown(
                    id='primary-variable',
                    options=column_options,
                    value=None,
                    placeholder="請選擇主要變數",
                    clearable=False
                )
            ], width=12, className='mb-3'),
            
            dbc.Col([
                html.Label("次要變數"),
                dcc.Dropdown(
                    id='secondary-variable',
                    options=column_options,
                    value=None,
                    placeholder="請選擇次要變數",
                    clearable=True,
                    disabled=True
                )
            ], width=12, className='mb-3'),
            
            dbc.Col([
                html.Label("圖表類型"),
                dcc.Dropdown(
                    id='chart-type',
                    options=[],  # 將由回調函數動態更新
                    value=None,
                    placeholder="請選擇圖表類型",
                    clearable=False,
                    disabled=True
                )
            ], width=12, className='mb-3')
        ])
    ])

def create_histogram_kde(df, column):
    """創建直方圖和KDE疊加圖"""
    # 計算KDE
    data = df[column].dropna()
    if data.empty:
        return go.Figure()

    kde = stats.gaussian_kde(data)
    x_range = np.linspace(min(data), max(data), 200)  # 增加平滑度
    kde_values = kde(x_range)

    # 創建圖表
    fig = go.Figure()

    # 添加直方圖
    fig.add_trace(go.Histogram(
        x=data,
        name='直方圖',
        nbinsx=30,
        histnorm='probability density',
        opacity=0.7  # 增加透明度以便更好地看到KDE曲線
    ))

    # 添加KDE曲線
    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde_values,
        name='密度曲線',
        line=dict(color='red', width=2)
    ))

    fig.update_layout(
        title=f'{column} 分布',
        xaxis_title=column,
        yaxis_title='密度',
        bargap=0.1,
        showlegend=True,
        barmode='overlay'  # 確保直方圖和KDE曲線重疊
    )

    return fig

def create_kde_for_categorical(values, weights):
    """使用 statsmodels 創建更精確的 KDE"""
    # 將類別索引轉換為數值
    x_points = np.arange(len(values))
    
    # 確保權重為正數且非零
    weights = np.maximum(weights, 1e-10)
    
    kde = KDEUnivariate(x_points)
    kde.fit(weights=weights, kernel='gaussian', fft=False, bw='scott')
    
    # 生成更密集的平滑曲線點
    x_kde = np.linspace(-0.5, len(values) - 0.5, 200)
    y_kde = kde.evaluate(x_kde)
    
    return x_kde, y_kde

def add_trendline(fig, df, x_col, y_col):
    """添加趨勢線到散佈圖"""
    # 移除遺漏值
    data = df[[x_col, y_col]].dropna()
    x = data[x_col].values
    y = data[y_col].values
    
    # 計算線性回歸
    coeffs = np.polyfit(x, y, 1)
    line = np.poly1d(coeffs)
    r_squared = np.corrcoef(x, y)[0, 1]**2
    
    # 生成趨勢線的 x 值
    x_trend = np.linspace(x.min(), x.max(), 100)
    
    # 添加趨勢線
    fig.add_trace(go.Scatter(
        x=x_trend,
        y=line(x_trend),
        mode='lines',
        name=f'趨勢線 (R² = {r_squared:.3f})',
        line=dict(color='red', dash='dash'),
        hovertemplate=(
            f'方程式: y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}<br>'
            f'R² = {r_squared:.3f}<extra></extra>'
        )
    ))
    
    return fig

def generate_plot(df, analysis_type, primary_var, chart_type, secondary_var=None):
    """根據選擇生成適當的圖表"""
    try:
        if df is None or df.empty or primary_var not in df.columns:
            return {}

        # 清理遺漏值
        df = df.dropna(subset=[primary_var])
        if secondary_var:
            df = df.dropna(subset=[secondary_var])

        # 獲取變數類型
        primary_type = get_variable_type(df, primary_var)
        secondary_type = get_variable_type(df, secondary_var) if secondary_var else None

        # 單變量分析
        if analysis_type == 'univariate':
            if primary_type == 'categorical':
                # 獲取唯一值數量
                unique_values = df[primary_var].nunique()
                value_counts = df[primary_var].value_counts()

                # 檢查唯一值數量是否超過20
                if unique_values > 20 and chart_type in ['pie', 'doughnut', 'treemap']:
                    # 創建提示圖表
                    fig = go.Figure()
                    fig.add_annotation(
                        text=f"該變數有 {unique_values} 個唯一值<br>因唯一值數量過多，無法顯示圖表",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=14)
                    )
                    fig.update_layout(
                        title=f'{primary_var} - 無法顯示圖表',
                        showlegend=False
                    )
                    return fig

                # 根據圖表類型生成視覺化
                if chart_type == 'bar':
                    fig = px.bar(df, x=primary_var)
                elif chart_type == 'lollipop':
                    fig = go.Figure()
                    
                    # 添加基準線
                    fig.add_trace(go.Scatter(
                        x=value_counts.index,
                        y=[0] * len(value_counts),
                        mode='lines',
                        line=dict(color='gray', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    # 添加垂直線
                    for idx in range(len(value_counts)):
                        fig.add_trace(go.Scatter(
                            x=[value_counts.index[idx], value_counts.index[idx]],
                            y=[0, value_counts.values[idx]],
                            mode='lines',
                            line=dict(color='gray', width=1),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    
                    # 添加圓點
                    fig.add_trace(go.Scatter(
                        x=value_counts.index,
                        y=value_counts.values,
                        mode='markers',
                        marker=dict(
                            size=8,
                            color='blue',
                            line=dict(color='darkblue', width=1)
                        ),
                        name='頻率'
                    ))
                    
                    fig.update_layout(
                        title=f'{primary_var} 分布',
                        xaxis_title=primary_var,
                        yaxis_title='頻率',
                        showlegend=False,
                        xaxis=dict(
                            showgrid=False,
                            showline=True,
                            linecolor='gray',
                            linewidth=1,
                            tickangle=45 if len(value_counts) > 5 else 0
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='lightgray',
                            showline=True,
                            linecolor='gray',
                            linewidth=1,
                            zeroline=True,
                            zerolinecolor='gray',
                            zerolinewidth=1,
                            rangemode='tozero'
                        )
                    )
                elif chart_type == 'pie':
                    fig = px.pie(df, names=primary_var)
                elif chart_type == 'doughnut':
                    fig = go.Figure(data=[go.Pie(
                        labels=value_counts.index,
                        values=value_counts.values,
                        hole=0.3
                    )])
                elif chart_type == 'treemap':
                    fig = px.treemap(df, path=[primary_var])
                else:
                    fig = px.bar(df, x=primary_var)  # 默認圖表
            else:
                # 修正：當 chart_type 為 'histogram_kde' 時，調用 create_histogram_kde
                if chart_type == 'histogram_kde':
                    fig = create_histogram_kde(df, primary_var)
                else:
                    fig = px.histogram(df, x=primary_var)

        # 雙變量分析
        elif analysis_type == 'bivariate' and secondary_var:
            # 檢查唯一值數量
            primary_unique = df[primary_var].nunique()
            secondary_unique = df[secondary_var].nunique()
            
            # 創建警告圖表的函數
            def create_warning_figure(var_name, unique_count):
                fig = go.Figure()
                fig.add_annotation(
                    text=f"變數 '{var_name}' 有 {unique_count} 個唯一值<br>因唯一值數量過多，無法顯示圖表",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14)
                )
                fig.update_layout(
                    title=f'{primary_var} vs {secondary_var} - 無法顯示圖表',
                    showlegend=False
                )
                return fig
            
            # 檢查類別型變數的唯一值數量
            if primary_type == 'categorical' and primary_unique > 20:
                return create_warning_figure(primary_var, primary_unique)
            if secondary_type == 'categorical' and secondary_unique > 20:
                return create_warning_figure(secondary_var, secondary_unique)

            # 原有的圖表邏輯
            if primary_type == 'categorical' and secondary_type == 'categorical':
                cross_tab = pd.crosstab(df[primary_var], df[secondary_var])
                if chart_type == 'heatmap':
                    total = cross_tab.sum().sum()
                    percentage_matrix = (cross_tab / total) * 100
                    fig = go.Figure(data=go.Heatmap(
                        z=percentage_matrix.values,
                        x=percentage_matrix.columns,
                        y=percentage_matrix.index,
                        colorscale='Teal',
                        colorbar=dict(
                            title='百分比 (%)',
                            titleside='right',
                            tickformat='.1f'
                        ),
                        hoverongaps=False,
                        hovertemplate='%{y} - %{x}<br>百分比: %{z:.1f}%<extra></extra>'
                    ))
                    fig.update_layout(
                        title=f'{primary_var} vs {secondary_var} 熱力圖',
                        xaxis_title=secondary_var,
                        yaxis_title=primary_var,
                        xaxis_tickangle=-45
                    )
                elif chart_type == 'stacked_histogram_percentage':
                    percentage_df = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
                    fig = go.Figure()
                    for column in percentage_df.columns:
                        fig.add_trace(go.Bar(
                            name=str(column),
                            x=percentage_df.index,
                            y=percentage_df[column],
                            marker_line_width=0
                        ))
                    fig.update_layout(
                        barmode='stack',
                        title=f'{primary_var} vs {secondary_var} (堆疊百分比直方圖)',
                        xaxis_title=primary_var,
                        yaxis_title='百分比 (%)',
                        yaxis=dict(
                            tickformat='.1f',
                            range=[0, 100]
                        ),
                        showlegend=True,
                        legend_title=secondary_var,
                        xaxis_tickangle=-45 if len(percentage_df.index) > 5 else 0
                    )
                elif chart_type in ['stacked_bar', 'side_by_side_bar']:
                    fig = px.bar(cross_tab,
                                    barmode='stack' if chart_type == 'stacked_bar' else 'group',
                                    title=f'{primary_var} vs {secondary_var}')
                else:
                    fig = px.bar(cross_tab)  # default
            elif (primary_type == 'numeric' and secondary_type == 'numeric'):
                if chart_type == 'scatter':
                    # 創建基本散佈圖
                    fig = px.scatter(df, x=primary_var, y=secondary_var)
                    # 添加趨勢線
                    fig = add_trendline(fig, df, primary_var, secondary_var)
                    # 更新圖表標題以反映趨勢線
                    fig.update_layout(
                        title=f'{primary_var} vs {secondary_var} (含趨勢線)'
                    )
                else:
                    fig = px.scatter(df, x=primary_var, y=secondary_var) #default
            elif (primary_type == 'categorical' and secondary_type == 'numeric'):
                if chart_type == 'box':
                    fig = px.box(df, x=primary_var, y=secondary_var)
                elif chart_type == 'violin':
                    fig = px.violin(df, x=primary_var, y=secondary_var)
                elif chart_type == 'grouped_scatter':
                    # 為每個類別創建散點圖
                    fig = px.scatter(df, 
                                   x=df.groupby(primary_var).cumcount(),  # 使用序號作為x軸
                                   y=secondary_var,
                                   color=primary_var,
                                   title=f'{primary_var} vs {secondary_var} (分組散點圖)')
                    fig.update_layout(
                        xaxis_title='數據點序號',
                        yaxis_title=secondary_var,
                        showlegend=True
                    )
                else:
                    fig = px.box(df, x=primary_var, y=secondary_var) #default
            elif (primary_type == 'numeric' and secondary_type == 'categorical'):
                if chart_type == 'box':
                    fig = px.box(df, x=secondary_var, y=primary_var)
                elif chart_type == 'violin':
                    fig = px.violin(df, x=secondary_var, y=primary_var)
                elif chart_type == 'grouped_scatter':
                    # 為每個類別創建散點圖
                    fig = px.scatter(df, 
                                   x=df.groupby(secondary_var).cumcount(),  # 使用序號作為x軸
                                   y=primary_var,
                                   color=secondary_var,
                                   title=f'{primary_var} vs {secondary_var} (分組散點圖)')
                    fig.update_layout(
                        xaxis_title='數據點序號',
                        yaxis_title=primary_var,
                        showlegend=True
                    )
                else:
                    fig = px.box(df, x=secondary_var, y=primary_var) #default
            else:
                fig = px.scatter(df, x=primary_var, y=secondary_var)

        # 添加通用的圖表配置
        if 'fig' in locals():
            fig.update_layout(
                template='plotly_white',
                margin=dict(l=40, r=40, t=40, b=40),
                height=500,
                font=dict(size=12)
            )
            return fig
            
        return {}
    
    except Exception as e:
        print(f"Error in generate_plot: {str(e)}")
        return {}

# app.py
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from utils.data_processing import parse_upload_contents
from components.visualizations import (
    create_visualization_controls, 
    generate_plot,
    get_variable_type,
    get_chart_options
)
import pandas as pd
import io

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
                html.Div(id='additional-controls'),      # 額外控制選項
                dcc.Graph(id='visualization-output')     # 圖表輸出區域
            ], id='analysis-section', style={'display': 'none'})  # 初始時隱藏
        ], width=12)
    ]),

    # 儲存數據的中間組件
    dcc.Store(id='stored-data')
], fluid=True)

@app.callback(
    [Output('output-data-upload', 'children'),
     Output('stored-data', 'data'),
     Output('analysis-section', 'style')],  # 新增輸出
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
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
    
    # 將數據存儲為 JSON
    stored_data = df.to_json(date_format='iso', orient='split')
    
    # 顯示分析區域
    return preview, stored_data, {'display': 'block'}

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

if __name__ == '__main__':
    app.run_server(debug=True)
