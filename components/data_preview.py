from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import pandas as pd

def generate_data_preview(df):
    """生成數據預覽組件"""
    if df is None:
        return html.Div()
    
    # 1. 基本資訊卡片
    info_card = dbc.Card([
        dbc.CardBody([
            html.H4("數據基本資訊", className="card-title"),
            html.P(f"資料列數: {len(df)}", className="card-text"),
            html.P(f"欄位數: {len(df.columns)}", className="card-text"),
            html.P(f"記憶體使用: {df.memory_usage().sum() / 1024**2:.2f} MB", className="card-text"),
        ])
    ], className="mb-3")

    # 2. 數據類型和缺失值資訊
    dtype_info = []
    for col in df.columns:
        missing = df[col].isna().sum()
        missing_pct = (missing / len(df)) * 100
        dtype_info.append({
            "欄位名稱": col,
            "數據類型": str(df[col].dtype),
            "缺失值數": missing,
            "缺失值比例": f"{missing_pct:.2f}%"
        })
    
    dtype_table = dash_table.DataTable(
        data=dtype_info,
        columns=[{"name": k, "id": k} for k in dtype_info[0].keys()],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        }
    )

    # 3. 數據預覽表格
    preview_table = dash_table.DataTable(
        data=df.head(5).to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        }
    )

    # 4. 數值型欄位的基本統計量
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        stats_df = df[numeric_cols].describe()
        stats_table = dash_table.DataTable(
            data=stats_df.round(2).to_dict('records'),
            columns=[{"name": i, "id": i} for i in stats_df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            }
        )
    else:
        stats_table = html.P("沒有數值型欄位")

    return html.Div([
        dbc.Tabs([
            dbc.Tab([
                html.Div([
                    info_card,
                    html.H5("數據預覽 (前5筆)", className="mt-3"),
                    preview_table
                ])
            ], label="數據預覽"),
            
            dbc.Tab([
                html.H5("欄位資訊", className="mt-3"),
                dtype_table
            ], label="欄位資訊"),
            
            dbc.Tab([
                html.H5("數值統計", className="mt-3"),
                stats_table
            ], label="統計資訊")
        ])
    ])