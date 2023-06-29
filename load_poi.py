import pandas as pd
import requests

api_keys=["j8VUTaKCsy9YwKQvy2FVR2fuz1HOvKdX8cWJFwDu","e8wHh2tya84M88aReEpXCa5XTQf3xgo01aZG39k5",
         "w3oFktJUat2NPpDorwHZE7avbxcvdq0S7wx4vXfN","Tt3yyROHTM8op2hEyv1Z34AXC2x8KPbn1iuD5Hlc",
         "RNM43SFreC8YwWjFIAGHY4VIpOi6jDHG98AHf7rN"]
api_key_index=0

#data 불러오기 - 1. Sk Api
def load_poi():
    df = pd.DataFrame(columns=['ticker', 'y', 'ds'])
    def get_data_pois(date, df):
        # 장소 id 리스트
        ids = [187961, 5783805, 5799875, 384515]
        # SK api 홈페이지에서 호출링크 가져옴
        base_url = "https://apis.openapi.sk.com/puzzle/place/congestion/stat/raw/hourly/pois/"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "appKey":api_keys[api_key_index]
        }
        # API 응답 저장할 빈 딕셔너리 생성
        responses = {}

        # for문으로 id리스트 이용, API 호출하기
        # id를 키로하여 Json형식으로 받아와서 저장
        for place_id in ids:
            url = base_url + str(place_id)
            query_params = "?date=" + str(date)
            full_url = url + query_params
            try:
                response = requests.get(full_url, headers=headers)
                response.raise_for_status()  # 오류가 있는 경우 예외 발생
                responses[place_id] = response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error occurred: {e}")
                print("Switching to the next API key...")
                switch_to_next_key()
                return get_data_pois(date, df) 

        # for문 reponse 딕셔너리 항목에서 키값으로 데이터 추출하고 각 변수에 저장
        for place_id, response_data in responses.items():
            poi_id = response_data['contents']['poiId']
            poi_name = response_data['contents']['poiName']
            for item in response_data['contents']['raw']:
                congestion = item['congestion']
                congestion_level = item['congestionLevel']
                datetime = item['datetime']

                # 'df'에 새로운 데이터 추가하고 인덱스 재설정
                new = pd.DataFrame({
                    #'Id': poi_id,
                    'ticker': [poi_name],
                    'y': [congestion],
                    #'CongestionLevel': congestion_level,
                    'ds': [datetime]
                })
                df=pd.concat([df,new], ignore_index=True)

        return df
    def get_data_areas(date, df):
        ids = [9273,9270]
        base_url = "https://apis.openapi.sk.com/puzzle/place/congestion/stat/raw/hourly/areas/"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "appKey":api_keys[api_key_index]
        }

        responses = {}
        for areas_id in ids:
            url = base_url + str(areas_id)
            query_params = "?date=" + str(date)
            full_url = url + query_params
            try:
                response = requests.get(full_url, headers=headers)
                response.raise_for_status()  # 오류가 있는 경우 예외 발생
                responses[areas_id] = response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error occurred: {e}")
                print("Switching to the next API key...")
                switch_to_next_key()
                return get_data_areas(date, df) 

        for areas_id, response_data in responses.items():
            area_id = response_data['contents']['areaId']
            area_name = response_data['contents']['areaName']

            for item in response_data['contents']['raw']:
                congestion = item['congestion']
                congestion_level = item['congestionLevel']
                datetime = item['datetime']
                new = pd.DataFrame({
                    #'Id': area_id,
                    'ticker': [area_name],
                    'y': [congestion],
                    #'CongestionLevel': congestion_level,
                    'ds': [datetime]
                })
                df=pd.concat([df,new], ignore_index=True)

        return df
    def switch_to_next_key():
        # 다음 API 키로 전환하는 함수
        global api_key_index
        api_key_index = (api_key_index + 1) % len(api_keys)

    
    df = get_data_pois('ystday',df)
    df = get_data_areas('ystday',df)
    return df


if __name__=='__main__':
    poi=load_poi()
    poi