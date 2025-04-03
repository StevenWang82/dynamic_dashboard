import pandas as pd
import base64
import io

def parse_upload_contents(contents, filename):
    """解析上傳的檔案內容"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return None, '目前只支援 CSV 檔案'
        
        return df, None
    except Exception as e:
        return None, f'處理檔案時發生錯誤: {str(e)}'