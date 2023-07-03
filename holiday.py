from datetime import datetime, timedelta
import pandas as pd
import requests, xmltodict

def holiday_api(year):
    max_retries=10
    retry_delay=3
    
    for attempt in range(max_retries):
        try:    
            decoding_key='6tblOMEsON8DV8mDJwrWHEDqFscUjGc0P1JLpq5QZE8Y/7jyE2piugAbGHDiy4oKYwbAaLiP+i9L1wb3HZ9VnQ=='
            url = 'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo'
            params ={'serviceKey' : decoding_key, 'pageNo' : '1', 'numOfRows' : '100', 'solYear' : str(year)}

            response = requests.get(url, params=params)

            holiday=xmltodict.parse(response.text)
            return holiday

        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Retrying ({attempt+1}/{max_retries}) in {retry_delay} seconds...")
            time.sleep(retry_delay)
        
        raise Exception("Exceeded maximum number of retries for train_weather()")

def make_holiday_df():
    year=datetime.today().year
    
    holiday = {
        'holiday': [],
        'ds': [],
        'lower_window':-1,
        'upper_window':0
    }
    #api 불러오기

    
    holiday_data=holiday_api(year)
    
    for item in holiday_data['response']['body']['items']['item']:
        if item['isHoliday'] == 'Y':
            holiday['holiday'].append(item['dateName'])
            holiday['ds'].append(pd.to_datetime(item['locdate'], format='%Y%m%d'))
    
    holiday=pd.DataFrame(holiday)
    
    #기타 행사(1) 할로윈데이
    # 10월 마지막 주 토요일 날짜 계산
    october_last_sat = pd.to_datetime(pd.date_range(start=f'{year}-10-01', end=f'{year}-10-31', freq='WOM-4SAT')[-1])
    
    holloween=pd.DataFrame({
        'holiday': 'Holloween',
        'ds': pd.to_datetime([october_last_sat]),
        'lower_window':-1,
        'upper_window':1
    })
    
    
    #합치기
    holiday_df=pd.concat([holiday, holloween])
    holiday_df.reset_index(inplace=True, drop=True)
    
    return holiday_df


if __name__=='__main__':
    holiday=make_holiday_df()