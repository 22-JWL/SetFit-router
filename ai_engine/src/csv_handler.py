import pandas as pd
import requests
import os
import sys

# 상위 폴더 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

class CSVHandler:
    def __init__(self):
        self.csv_path = settings.CSV_FILE_PATH
        self.rules = {}
        self._load_csv()

    def _load_csv(self):
        """CSV 파일을 로드하여 메모리에 Dict 형태로 저장"""
        if not os.path.exists(self.csv_path):
            print(f"[Warning] CSV file not found at: {self.csv_path}")
            return

        try:
            # CSV 읽기 (헤더가 있다고 가정)
            #df = pd.read_csv(self.csv_path)

            # 1. 먼저 UTF-8로 시도 (표준)
            try:
                df = pd.read_csv(self.csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                # 2. 실패 시 CP949로 재시도 (한국어 Windows 엑셀 저장 파일)
                print(f">>> [CSVHandler] 'utf-8' decode failed. Retrying with 'cp949'...")
                df = pd.read_csv(self.csv_path, encoding='cp949')
            
            # B열(Text) -> C열(API URL) 매핑
            # 엑셀의 B열은 인덱스 1, C열은 인덱스 2지만, pandas는 컬럼명으로 접근하거나 iloc 사용
            # 여기서는 안전하게 iloc(인덱스) 사용: iloc[:, 1]은 B열, iloc[:, 2]는 C열
            count = 0
            for _, row in df.iterrows():
                # 데이터가 있는지 안전하게 확인 (iloc 사용)
                if len(row) >= 3 and pd.notna(row.iloc[1]) and pd.notna(row.iloc[2]):
                    key = str(row.iloc[1]).strip() # B열 (User Text)
                    value = str(row.iloc[2]).strip() # C열 (API Endpoint)
                    
                    # 키가 비어있지 않을 때만 등록
                    if key: 
                        self.rules[key] = value
                        count += 1
            
            print(f">>> [CSVHandler] Loaded {len(self.rules)} rules from CSV.")
            print(f">>> [CSVHandler] Successfully loaded {count} rules from CSV.")
            
        except Exception as e:
            print(f"[Error] Failed to load CSV: {e}")

    def check_and_execute(self, query):
        """
        쿼리가 CSV 규칙에 있는지 확인하고, 있으면 API 호출 실행
        Returns:
            result (str or None): API 응답 결과 (없으면 None)
        """
        # 정확히 일치하는지 확인 (필요시 부분 일치로 변경 가능)
        api_suffix = self.rules.get(query.strip())
        
        if api_suffix:
            target_url = f"{settings.EXTERNAL_API_BASE_URL}{api_suffix}"
            print(f">>> [Rule Match] Query matched CSV! Calling: {target_url}")
            return "[API 연동 결과] (모의 응답) API 호출이 성공적으로 처리되었습니다."
            
            # try:
            #     # GET 요청 (필요시 POST로 변경)
            #     response = requests.get(target_url, timeout=5)
                
            #     if response.status_code == 200:
            #         return f"[API 연동 결과] {response.text}"
            #     else:
            #         return f"[API Error] {response.status_code}: {response.text}"
                    
            # except requests.RequestException as e:
            #     return f"[API Connection Error] Node.js 서버({settings.EXTERNAL_API_BASE_URL})에 연결할 수 없습니다."
        
        return None # 매칭되는 규칙 없음