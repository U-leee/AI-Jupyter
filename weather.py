import time
import pandas as pd
from datetime import datetime, timedelta
import requests, xmltodict

#하루전까지 날씨 (train)
def train_weather():
    max_retries=10
    retry_delay=3
    
    for attempt in range(max_retries):
        try:
            df=pd.read_csv('train_weather.csv', encoding='ms949')
            df.ds=pd.to_datetime(df.ds)
            df=df.fillna(method='ffill') #결측값 채우기

            #하루전 날씨 업데이트
            date=((datetime.now())-timedelta(days=2)).strftime('%Y%m%d')
            new=test_weather(date)
            new=new[:24]

            df=pd.concat([df, new])
            df=df.drop_duplicates()
            df.to_csv('train_weather.csv', index=False)
            
            return df
        
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Retrying ({attempt+1}/{max_retries}) in {retry_delay} seconds...")
            time.sleep(retry_delay)
        
        raise Exception("Exceeded maximum number of retries for train_weather()")



def test_weather(date=((datetime.now())-timedelta(days=1)).strftime('%Y%m%d')): #t~t+2 (당일~2일후까지 총 3일간 시간대별 기온, 강수량)
    
    decoding_key='6tblOMEsON8DV8mDJwrWHEDqFscUjGc0P1JLpq5QZE8Y/7jyE2piugAbGHDiy4oKYwbAaLiP+i9L1wb3HZ9VnQ=='
    url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'
    
    params ={'serviceKey' : decoding_key, 'pageNo' : '1', 'numOfRows' : '1000', 'dataType' : 'XML', 
             'base_date' : date, 
             'base_time' : '2300', 
             'nx' : '62', 'ny' : '126' } #송파구: 62(nx), 126(ny)
    response = requests.get(url, params=params)
    todict=xmltodict.parse(response.text)
    
    #dataframe 생성
    df=pd.DataFrame(columns=['ds', 'category', 'value'])

    for item in todict['response']['body']['items']['item']:
        fcst_date=item['fcstDate']
        fcst_time=item['fcstTime']
        category=item['category']
        value=item['fcstValue']

        data=pd.DataFrame({'ds': [fcst_date+fcst_time], 'category':[category], 'value':[value]})
        df=pd.concat([df, data])

    #전처리
    df['ds']=pd.to_datetime(df['ds'], format='%Y%m%d%H%M')
    df=df.query('category=="TMP"|category=="PCP"')
    df=df.pivot('ds','category','value')
    df=df.reset_index()
    df.TMP=df.TMP.apply(float)
    df['PCP']=df['PCP'].apply(lambda x: x.split('~')[0].strip('mm'))
    df['PCP']=df['PCP'].replace({'강수없음':'0.0'}).apply(float)
    df=df.fillna(method='ffill') #혹시모를 결측값 처리
    df=df[['ds','TMP','PCP']] #열순서 변경
    df=df.iloc[:-1] #마지막 행 삭제
    
    return df


if __name__=='__main__':
    train=train_weather()
    test=test_weather()
