import pandas as pd
from dash import html, dcc
import dash_bootstrap_components as dbc

def detect_date_columns(df):
    """檢測可能是日期的欄位"""
    potential_date_cols = []
    
    for col in df.columns:
        # 檢查是否已經是datetime類型
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            potential_date_cols.append(col)
            continue
            
        # 如果是object類型，嘗試轉換看看
        if pd.api.types.is_object_dtype(df[col]):
            try:
                # 取樣前100筆資料進行測試
                sample = df[col].dropna().head(100)
                pd.to_datetime(sample, infer_format=True)
                potential_date_cols.append(col)
            except:
                continue
                
    return potential_date_cols

def create_date_controls(df):
    """創建日期轉換的模態對話框控制元件"""
    date_cols = detect_date_columns(df)
    
    return html.Div([
        # 觸發按鈕
        dbc.Button(
            "日期欄位轉換",
            id="open-date-modal",
            color="primary",
            className="mb-3"
        ),
        
        # 模態對話框
        dbc.Modal([
            dbc.ModalHeader("日期欄位轉換"),
            dbc.ModalBody([
                # 自動檢測結果顯示
                dbc.Alert(
                    "自動檢測到以下可能的日期欄位" if date_cols else "未自動檢測到日期欄位，請手動選擇需要轉換的欄位",
                    color="info" if date_cols else "warning",
                    className="mb-3"
                ),
                
                # 欄位選擇區域
                html.H6("選擇要轉換的欄位："),
                dcc.Checklist(
                    id='date-columns-checklist',
                    options=[{'label': col, 'value': col} for col in df.columns],
                    value=date_cols,
                    className='mb-3'
                ),
                
                # 日期格式輸入區域
                html.H6("日期格式設定"),
                dbc.Input(
                    id='date-format-input',
                    type='text',
                    placeholder='例如：%Y-%m-%d',
                    className='mb-2'
                ),
                html.Small(
                    [
                        "常用格式：",
                        html.Br(),
                        "%Y-%m-%d (2023-12-31)",
                        html.Br(),
                        "%Y/%m/%d (2023/12/31)",
                        html.Br(),
                        "%d/%m/%Y (31/12/2023)",
                        html.Br(),
                        "%Y%m%d (20231231)"
                    ],
                    className="text-muted mb-3"
                ),
                
                # 轉換狀態顯示
                html.Div(id='date-conversion-status')
            ]),
            dbc.ModalFooter([
                dbc.Button(
                    "轉換",
                    id="convert-dates-button",
                    color="primary",
                    className="me-2"
                ),
                dbc.Button(
                    "關閉",
                    id="close-date-modal",
                    color="secondary"
                )
            ])
        ], id="date-modal", size="lg")
    ])

def convert_dates(df, date_columns, date_format=None):
    """轉換日期欄位並生成衍生欄位"""
    df = df.copy()
    conversion_results = {
        'success': [],
        'failed': [],
        'derived_columns': []
    }
    
    for col in date_columns:
        try:
            # 如果提供了格式，先嘗試使用指定格式
            if date_format:
                try:
                    df[col] = pd.to_datetime(df[col], format=date_format)
                except ValueError:
                    # 如果指定格式失敗，嘗試自動推斷
                    df[col] = pd.to_datetime(df[col])
            else:
                # 直接嘗試自動推斷
                df[col] = pd.to_datetime(df[col])
            
            # 生成衍生欄位
            col_prefix = f"{col}_"
            
            # 年份
            df[col_prefix + 'year'] = df[col].dt.year
            conversion_results['derived_columns'].append(col_prefix + 'year')
            
            # 月份
            df[col_prefix + 'month'] = df[col].dt.month
            conversion_results['derived_columns'].append(col_prefix + 'month')
            
            # 日期
            df[col_prefix + 'day'] = df[col].dt.day
            conversion_results['derived_columns'].append(col_prefix + 'day')
            
            # 星期名稱
            df[col_prefix + 'weekname'] = df[col].dt.day_name()
            conversion_results['derived_columns'].append(col_prefix + 'weekname')
            
            conversion_results['success'].append(col)
            
        except Exception as e:
            conversion_results['failed'].append({
                'column': col,
                'error': str(e)
            })
            
    return df, conversion_results

def generate_conversion_report(results):
    """生成轉換結果報告"""
    report_elements = []
    
    if results['success']:
        report_elements.append(
            dbc.Alert(
                [
                    html.H6("成功轉換的欄位："),
                    html.Ul([html.Li(col) for col in results['success']])
                ],
                color="success",
                className="mb-2"
            )
        )
        
        report_elements.append(
            dbc.Alert(
                [
                    html.H6("新增的衍生欄位："),
                    html.Ul([html.Li(col) for col in results['derived_columns']])
                ],
                color="info",
                className="mb-2"
            )
        )
    
    if results['failed']:
        report_elements.append(
            dbc.Alert(
                [
                    html.H6("轉換失敗的欄位："),
                    html.Ul([
                        html.Li([
                            f"{fail['column']}: ",
                            html.Small(fail['error'])
                        ]) for fail in results['failed']
                    ])
                ],
                color="danger",
                className="mb-2"
            )
        )
    
    return html.Div(report_elements)


