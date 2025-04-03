from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats

def get_variable_type(df, column):
    """判斷變數類型"""
    if pd.api.types.is_numeric_dtype(df[column]):
        return 'numeric'
    return 'categorical'

def get_chart_options(analysis_type, var_types):
    """根據分析類型和變量類型返回可用的圖表選項"""
    univariate_charts = {
        'numeric': [
            {'label': '直方圖+KDE', 'value': 'histogram_kde'},
            # ... 其他數值型單變量圖表選項 ...
        ],
        'categorical': [
            {'label': '條形圖', 'value': 'bar'},
            {'label': '圓餅圖', 'value': 'pie'},
            {'label': '鬆餅圖', 'value': 'waffle'}
        ]
    }
    
    bivariate_charts = {
        'categorical_categorical': [
            {'label': '熱力圖', 'value': 'heatmap'},
            {'label': '堆疊長條圖', 'value': 'stacked_bar'},
            {'label': '並排長條圖', 'value': 'side_by_side_bar'},
            {'label': '堆疊百分比直方圖', 'value': 'stacked_histogram_percentage'}
        ]
        # ... 其他變量組合的圖表選項 ...
    }
    
    if analysis_type == 'univariate':
        return univariate_charts[var_types['primary']]
    elif analysis_type == 'bivariate':
        combo_key = f"{var_types['primary']}_{var_types['secondary']}"
        return bivariate_charts.get(combo_key, [])

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
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(min(data), max(data), 100)
    kde_values = kde(x_range)

    # 創建圖表
    fig = go.Figure()

    # 添加直方圖
    fig.add_trace(go.Histogram(
        x=data,
        name='直方圖',
        nbinsx=30,
        histnorm='probability density'
    ))

    # 添加KDE曲線
    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde_values,
        name='密度曲線',
        line=dict(color='red')
    ))

    fig.update_layout(
        title=f'{column} 分布',
        xaxis_title=column,
        yaxis_title='密度',
        bargap=0.1
    )

    return fig



def generate_plot(df, analysis_type, primary_var, chart_type, secondary_var=None):
    """根據選擇生成適當的圖表"""
    try:
        if df is None or df.empty or primary_var not in df.columns:
            return {}

        # 獲取變數類型
        primary_type = get_variable_type(df, primary_var)
        secondary_type = get_variable_type(df, secondary_var) if secondary_var else None

        # 單變量分析
        if analysis_type == 'univariate':
            if primary_type == 'numeric':
                if chart_type == 'histogram_kde':
                    fig = create_histogram_kde(df, primary_var)
            
            elif primary_type == 'categorical':
                if chart_type == 'bar':
                    value_counts = df[primary_var].value_counts()
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f'{primary_var} 分布'
                    )
                elif chart_type == 'pie':
                    value_counts = df[primary_var].value_counts()
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f'{primary_var} 分布'
                    )

        # 雙變量分析
        elif analysis_type == 'bivariate' and secondary_var:
            if primary_type == 'categorical' and secondary_type == 'categorical':
                cross_tab = pd.crosstab(df[primary_var], df[secondary_var])
                
                if chart_type == 'heatmap':
                    # 計算百分比
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
