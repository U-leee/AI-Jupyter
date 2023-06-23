#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries
import pandas as pd
import numpy as np
import requests, xmltodict
from datetime import datetime, timedelta
from time import time
import glob

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'NanumGothic'
import seaborn as sns
color_pal=sns.color_palette()

from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly, add_changepoints_to_plot, plot_cross_validation_metric
from prophet.diagnostics import cross_validation, performance_metrics

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#실시간 data 불러오기 - 1. Sk Api
def rltm_poi():
    df = pd.DataFrame(columns=['ds','ticker', 'y', 'CongestionLevel'])
    def get_rltm_pois(df):
        # 장소 id 리스트
        ids = [6967166, 187961, 188485, 188592, 5783805, 5799875, 384515, 188633]
        # SK api 홈페이지에서 호출링크 가져옴
        base_url = "https://apis.openapi.sk.com/puzzle/place/congestion/rltm/pois/"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            #"appKey": "RNM43SFreC8YwWjFIAGHY4VIpOi6jDHG98AHf7rN", # 앱키1 - 나 
            #"appKey": "Tt3yyROHTM8op2hEyv1Z34AXC2x8KPbn1iuD5Hlc", # 앱키2 - 민
            #"appKey": "w3oFktJUat2NPpDorwHZE7avbxcvdq0S7wx4vXfN", # 앱키3 - 아
            #"appkey": "e8wHh2tya84M88aReEpXCa5XTQf3xgo01aZG39k5", # 앱키4 
            "appkey": "j8VUTaKCsy9YwKQvy2FVR2fuz1HOvKdX8cWJFwDu" #앱키5 - 미리내
        }
        # API 응답 저장할 빈 딕셔너리 생성
        responses = {}

        # for문으로 id리스트 이용, API 호출하기
        # id를 키로하여 Json형식으로 받아와서 저장
        for place_id in ids:
            url = base_url + str(place_id)
            #query_params = "?date=" + str(date)
            #full_url = url + query_params
            response = requests.get(url, headers=headers)
            responses[place_id] = response.json() 

        # for문 reponse 딕셔너리 항목에서 키값으로 데이터 추출하고 각 변수에 저장
        for place_id, response_data in responses.items():
            #poi_id = response_data['contents']['poiId']
            poi_name = response_data['contents']['poiName']
            for item in response_data['contents']['rltm']:
                congestion = item['congestion']
                congestion_level = item['congestionLevel']
                datetime = item['datetime']

                # 'df'에 새로운 데이터 추가하고 인덱스 재설정
                df = df.append({
                    #'Id': poi_id,
                    'ds': datetime,
                    'ticker': poi_name,
                    'y': congestion,
                    'CongestionLevel': congestion_level
                }, ignore_index=True)

        return df
    def get_rltm_areas(df):
        ids = [9273,9270]
        base_url = "https://apis.openapi.sk.com/puzzle/place/congestion/rltm/areas/"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            #"appKey": "RNM43SFreC8YwWjFIAGHY4VIpOi6jDHG98AHf7rN", # 앱키1 - 나 
            #"appKey": "Tt3yyROHTM8op2hEyv1Z34AXC2x8KPbn1iuD5Hlc", # 앱키2 - 민
            #"appKey": "w3oFktJUat2NPpDorwHZE7avbxcvdq0S7wx4vXfN", # 앱키3 - 아
            #"appkey": "e8wHh2tya84M88aReEpXCa5XTQf3xgo01aZG39k5", # 앱키4 
            "appkey": "j8VUTaKCsy9YwKQvy2FVR2fuz1HOvKdX8cWJFwDu" #앱키5 - 미리내
        }

        responses = {}
        for areas_id in ids:
            url = base_url + str(areas_id)
            #query_params = "?date=" + str(date)
            #full_url = url + query_params
            response = requests.get(url, headers=headers)
            responses[areas_id] = response.json() 


        for areas_id, response_data in responses.items():
            #area_id = response_data['contents']['areaId']
            area_name = response_data['contents']['areaName']

            congestion = response_data['contents']['rltm']['congestion']
            congestion_level = response_data['contents']['rltm']['congestionLevel']
            datetime = response_data['contents']['rltm']['datetime']
            df = df.append({
                #'Id': area_id,
                'ds': datetime,
                'ticker': area_name,
                'y': congestion,
                'CongestionLevel': congestion_level,
            }, ignore_index=True)

        return df
    
    df = get_rltm_pois(df)
    df = get_rltm_areas(df)
    df['ds']=pd.to_datetime(df['ds'], format='%Y%m%d%H%M%S')
    
    return df


# In[298]:


df=rltm_poi()
df


# In[3]:


#data 불러오기 - 1. Sk Api
def load_poi():
    df = pd.DataFrame(columns=['Id', 'Name', 'Congestion', 'CongestionLevel', 'Datetime'])
    def get_data_pois(date, df):
        # 장소 id 리스트
        ids = [6967166, 187961, 188485, 188592, 5783805, 5799875, 384515, 188633]
        # SK api 홈페이지에서 호출링크 가져옴
        base_url = "https://apis.openapi.sk.com/puzzle/place/congestion/stat/raw/hourly/pois/"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            #"appKey": "RNM43SFreC8YwWjFIAGHY4VIpOi6jDHG98AHf7rN", # 앱키1 - 나 
            #"appKey": "Tt3yyROHTM8op2hEyv1Z34AXC2x8KPbn1iuD5Hlc", # 앱키2 - 민
            #"appKey": "w3oFktJUat2NPpDorwHZE7avbxcvdq0S7wx4vXfN", # 앱키3 - 아
            #"appkey": "e8wHh2tya84M88aReEpXCa5XTQf3xgo01aZG39k5", # 앱키4 
            "appkey": "j8VUTaKCsy9YwKQvy2FVR2fuz1HOvKdX8cWJFwDu" #앱키5 - 미리내
        }
        # API 응답 저장할 빈 딕셔너리 생성
        responses = {}

        # for문으로 id리스트 이용, API 호출하기
        # id를 키로하여 Json형식으로 받아와서 저장
        for place_id in ids:
            url = base_url + str(place_id)
            query_params = "?date=" + str(date)
            full_url = url + query_params
            response = requests.get(full_url, headers=headers)
            responses[place_id] = response.json() 

        # for문 reponse 딕셔너리 항목에서 키값으로 데이터 추출하고 각 변수에 저장
        for place_id, response_data in responses.items():
            poi_id = response_data['contents']['poiId']
            poi_name = response_data['contents']['poiName']
            for item in response_data['contents']['raw']:
                congestion = item['congestion']
                congestion_level = item['congestionLevel']
                datetime = item['datetime']

                # 'df'에 새로운 데이터 추가하고 인덱스 재설정
                df = df.append({
                    'Id': poi_id,
                    'Name': poi_name,
                    'Congestion': congestion,
                    'CongestionLevel': congestion_level,
                    'Datetime': datetime
                }, ignore_index=True)

        return df
    def get_data_areas(date, df):
        ids = [9273,9270]
        base_url = "https://apis.openapi.sk.com/puzzle/place/congestion/stat/raw/hourly/areas/"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            #"appKey": "RNM43SFreC8YwWjFIAGHY4VIpOi6jDHG98AHf7rN", # 앱키1 - 나 
            #"appKey": "Tt3yyROHTM8op2hEyv1Z34AXC2x8KPbn1iuD5Hlc", # 앱키2 - 민
            #"appKey": "w3oFktJUat2NPpDorwHZE7avbxcvdq0S7wx4vXfN", # 앱키3 - 아
            #"appkey": "e8wHh2tya84M88aReEpXCa5XTQf3xgo01aZG39k5", # 앱키4 
            "appkey": "j8VUTaKCsy9YwKQvy2FVR2fuz1HOvKdX8cWJFwDu" #앱키5 - 미리내
        }

        responses = {}
        for areas_id in ids:
            url = base_url + str(areas_id)
            query_params = "?date=" + str(date)
            full_url = url + query_params
            response = requests.get(full_url, headers=headers)
            responses[areas_id] = response.json() 

        for areas_id, response_data in responses.items():
            area_id = response_data['contents']['areaId']
            area_name = response_data['contents']['areaName']

            for item in response_data['contents']['raw']:
                congestion = item['congestion']
                congestion_level = item['congestionLevel']
                datetime = item['datetime']
                df = df.append({
                    'Id': area_id,
                    'Name': area_name,
                    'Congestion': congestion,
                    'CongestionLevel': congestion_level,
                    'Datetime': datetime
                }, ignore_index=True)

        return df
    df = get_data_pois('ystday',df)
    df = get_data_areas('ystday',df)
    return df


# In[4]:


def raw_dataset():
    origin=pd.read_csv('congestion.csv') #2일전까지 데이터
    new=load_poi()                       #1일전(어제)데이터, api 불러오기
    mer_df=pd.concat([origin, new])
    mer_df=mer_df.drop_duplicates()      #중복제거
    mer_df.to_csv('congestion.csv', index=False) #저장
    return mer_df


# In[5]:


def preprocess_api():
    con=raw_dataset()
    con['Datetime']=pd.to_datetime(con['Datetime'], format='%Y%m%d%H%M%S') #type: datetime
    con=con.rename(columns={'Datetime':'ds','Congestion':'y','Name':'ticker'}) 
    con=con.fillna(0) #congestion 결측값(None=>0)
    con=con[['ds','ticker','y']]
    con=con.drop_duplicates()

    #롯데월드어드벤쳐 + 롯데월드잠실점 합치기
    lotte=con.query('ticker=="롯데월드어드벤쳐"')
    world=con.query('ticker=="롯데월드잠실점"')
    mer_lotte=pd.merge(lotte, world, on='ds')
    mer_lotte['y']=mer_lotte['y_x']+mer_lotte['y_y']
    mer_lotte['ticker']='롯데월드'
    mer_lotte=mer_lotte[['ds','ticker','y']]
    con=con.query('ticker!="롯데월드어드벤쳐"&ticker!="롯데월드잠실점"')
    con=pd.concat([con,mer_lotte])

    con_piv=con.pivot_table('y','ds','ticker')
    return con, con_piv


# In[7]:


#con, con_piv=preprocess_api()


# In[201]:


con_piv.corr()


# In[200]:


con_piv


# In[6]:


def holiday_api(year):
    import requests, xmltodict
    decoding_key='6tblOMEsON8DV8mDJwrWHEDqFscUjGc0P1JLpq5QZE8Y/7jyE2piugAbGHDiy4oKYwbAaLiP+i9L1wb3HZ9VnQ=='
    url = 'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo'
    params ={'serviceKey' : decoding_key, 'pageNo' : '1', 'numOfRows' : '100', 'solYear' : str(year)}

    response = requests.get(url, params=params)

    holiday=xmltodict.parse(response.text)
    return holiday


# In[7]:


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


# In[11]:


holiday_df=make_holiday_df()
holiday_df


# In[8]:


#하루전까지 날씨 (train)
def train_weather():
    df=pd.read_csv('train_weather.csv', encoding='ms949')
    df.ds=pd.to_datetime(df.ds)
    #df=df.rename(columns={'일시':'ds','기온(°C)':'TMP','강수량(mm)':'PCP'})
    #df=df[['ds','TMP','PCP']]
    df=df.fillna(method='ffill') #결측값 채우기
    
    #하루전 날씨 업데이트
    date=((datetime.now())-timedelta(days=2)).strftime('%Y%m%d')
    new=test_weather(date)
    new=new[:24]
    
    df=pd.concat([df, new])
    df=df.drop_duplicates()
    df.to_csv('train_weather.csv', index=False)
    return df


# In[9]:


def test_weather(date=((datetime.now())-timedelta(days=1)).strftime('%Y%m%d')): #t~t+2 (당일~2일후까지 총 3일간 시간대별 기온, 강수량)
    import requests, xmltodict
    from datetime import datetime, timedelta
    
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
    df['PCP']=df['PCP'].str.replace('mm','')
    df['PCP']=df['PCP'].replace({'강수없음':'0.0'}).apply(float)
    df=df.fillna(method='ffill') #혹시모를 결측값 처리
    df=df[['ds','TMP','PCP']] #열순서 변경
    df=df.iloc[:-1] #마지막 행 삭제
    
    return df


# In[18]:


df=test_weather()
df


# In[10]:


def create_features(mul):
    mul['hour']=mul['ds'].dt.hour

    # weekday 열 생성 (주중: 1, 주말: 0)
    mul['weekday']=mul['ds'].dt.dayofweek
    mul['weekday']=mul['weekday'].apply(lambda x: 0 if x >= 5 else 1)

    # season 열 생성
    mul['season']=mul['ds'].dt.month
    mul['season']=mul['season'].replace(12,0)
    mul['season']=pd.cut(mul['season'], [-1,2,5,8,11], labels=['Winter', 'Spring', 'Summer', 'Fall'])
    season_ohe=pd.get_dummies(mul['season'], prefix='season')
    mul=mul.join(season_ohe)

    #timebin 열 생성
    mul['timebin']=pd.cut(mul['hour'], bins=4, labels=False) #[(-0.023, 5.75] < (5.75, 11.5] < (11.5, 17.25] < (17.25, 23.0]]
    time_ohe=pd.get_dummies(mul['timebin'], prefix='tbin')
    mul=mul.join(time_ohe)
    # 결과 출력
    mul=mul.drop(columns=['hour','season','timebin'])
    
    return mul


# ## MODEL 1 - 날씨 포함 3일치 예측 (t~t+2)

# In[148]:


#feature engineering
#t~t+2, 3일치 다변량 예측
#logistic => 음수 제한 작동안함, 그냥 음수는 0으로 처리하게끔 코딩해야 할 듯, =>linear가 더 잘 나옴
#Train dataset 불러오기
#con, con_piv=preprocess_api()

train_wea=train_weather()
test_wea=test_weather()
#holiday_df=holiday_df
holiday_df=make_holiday_df()


#날씨변수 병합
mul=pd.merge(con, train_wea, on='ds') 

#날짜파생 변수 병합
mul=create_features(mul)
groups_by_ticker=mul.groupby('ticker')
groups_by_ticker.groups.keys()
ticker_list=list(groups_by_ticker.groups.keys())


start_time=time()

for_loop_forecast=pd.DataFrame()

for ticker in ticker_list:
    group=groups_by_ticker.get_group(ticker)
    
    m = Prophet(interval_width=0.8, 
                #changepoint_range=0.8, #default가 good
                #n_changepoints=20,
                seasonality_mode='multiplicative', #good
                holidays_prior_scale=15,
                holidays=holiday_df) 
   
    #변수 추가
    m.add_regressor('TMP', standardize=False)
    m.add_regressor('PCP', standardize=False)
    m.add_regressor('weekday', standardize=False)
    m.add_regressor('season_Winter', standardize=False)
    m.add_regressor('season_Spring', standardize=False)
    m.add_regressor('season_Summer', standardize=False)
    m.add_regressor('season_Fall', standardize=False)
    m.add_regressor('tbin_0', standardize=False)
    m.add_regressor('tbin_1', standardize=False)
    m.add_regressor('tbin_2', standardize=False)
    m.add_regressor('tbin_3', standardize=False)

    #주간 주기성 요소 추가 (*)
    m.add_seasonality(name='weekly', period=7, fourier_order=10)
    
    m.fit(group)
    
    #test dataset 불러오기
    test_w=pd.concat([train_wea, test_wea]) #날씨 병합 888+72
    future=m.make_future_dataframe(periods=3*24, freq='h')

    future=pd.merge(future, test_w[['ds','TMP','PCP']], on='ds', how='outer')
    future=create_features(future)
    forecast=m.predict(future)

    
    #시각화
    fig=m.plot(forecast)
    ax=fig.gca()
    ax.plot(group['ds'],group['y'],'b.')
    
    #performance=pd.merge(group, forecast[['ds','yhat','yhat_lower','yhat_upper']], on='ds')
    mae=mean_absolute_error(group['y'], forecast['yhat'][:-72])
    mape=mean_absolute_percentage_error(group['y'], forecast['yhat'][:-72])
    print(ticker,'MAE:', mae,'MAPE:',mape)
    
    forecast['ticker']=group['ticker'].iloc[0]
    forecast=forecast[['ds','ticker','yhat','yhat_upper','yhat_lower']]
    
    #전체 장소 합치기
    for_loop_forecast=pd.concat([for_loop_forecast, forecast])
    
    
print('Time:', time()-start_time)


# In[195]:


for_loop_forecast


# In[199]:


#3일치 예측값
#음수 값 -> 0으로 대체
for_loop_forecast['yhat']=for_loop_forecast.apply(lambda x: 1e-6 if x['yhat']<0 else x['yhat'], axis=1)
total_forecast=for_loop_forecast.copy()
predictions=pd.merge(mul,total_forecast, on=['ds','ticker'], how='outer')[['ds','ticker','y','yhat','yhat_upper','yhat_lower']]
predictions=predictions.iloc[-3*24*9:]
predictions


# In[192]:


#이상 탐지
results=pd.merge(mul,for_loop_forecast, on=['ds','ticker'], how='outer') 
results=results[['ds','ticker','y','yhat','yhat_upper','yhat_lower']] 

results['error']=results['y']-results['yhat']
results['uncertainty']=results['yhat_upper']-results['yhat_lower']

#숫자가 작아질 수록 이상치로 정의될 경우가 많아짐. 
#1: 평소보다 많을 때, -1: 평소보다 적을 때

results['anomaly'] = results.apply(lambda x: -1 if x['error'] < -1.5 * x['uncertainty'] 
                                   else (1 if x['error'] > 1.2 * x['uncertainty'] else 0), axis=1)
                                    #민감도를 사용자가 설정할 수 있게 할지? ex)민감: 0.7, 덜 민감:0.85 등.. 
def two_weeks_ago_mean(x):
    if (x.name-7*24*9 >= 0) & (x.name-7*2*24*9 >= 0):
        if (pd.notna(results.iloc[x.name-7*24*9]['y'])) & (pd.notna(results.iloc[x.name-7*2*24*9]['y'])):
            return ((results.iloc[x.name-7*24*9]['y'])+(results.iloc[x.name-7*2*24*9]['y']))/2
    else:
        None

results['avg_2w']=results.apply(two_weeks_ago_mean, axis=1) #최근 2주간 평균
results['anomaly2']=results.apply(lambda x: 1 if x['y'] > 2.0 * x['avg_2w'] else 0, axis=1)

results


# In[48]:


results.query('anomaly==1') #평소보다 많을 때


# In[53]:


results.query('anomaly==-1') #평소보다 적을 때


# In[54]:


#시각화
import plotly.graph_objects as go

groups_by_ticker=results.groupby('ticker')
groups_by_ticker.groups.keys()
ticker_list=list(groups_by_ticker.groups.keys())
for ticker in ticker_list:
    group=groups_by_ticker.get_group(ticker)
    # 데이터프레임의 시간 열을 x축으로 설정하여 그래프 생성
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=group['ds'], y=group['y'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=group['ds'], y=group['yhat'], mode='lines', name='Predicted'))
    fig.add_trace(go.Scatter(x=group[group['anomaly'] != 0]['ds'], y=group[group['anomaly'] != 0]['y'],
                             mode='markers', name='Anomaly', marker=dict(color='black', size=4)))

    fig.update_layout(title=f'{ticker}: Anomaly Detection', xaxis_title='Date', yaxis_title='Value')
    fig.show()


# In[193]:


import plotly.graph_objects as go

groups_by_ticker=results.groupby('ticker')
groups_by_ticker.groups.keys()
ticker_list=list(groups_by_ticker.groups.keys())
for ticker in ticker_list:
    group=groups_by_ticker.get_group(ticker)
    # 데이터프레임의 시간 열을 x축으로 설정하여 그래프 생성
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=group['ds'], y=group['y'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=group['ds'], y=group['yhat'], mode='lines', name='Predicted'))
    fig.add_trace(go.Scatter(x=group[group['anomaly2'] != 0]['ds'], y=group[group['anomaly2'] != 0]['y'],
                             mode='markers', name='Anomaly', marker=dict(color='black', size=4)))

    fig.update_layout(title=f'{ticker}: Anomaly Detection', xaxis_title='Date', yaxis_title='Value')
    fig.show()


# In[180]:


#10분 단위로 Resample
groups_by_ticker=results.groupby('ticker')

dfs_by_ticker=pd.DataFrame()
for ticker, group in groups_by_ticker:
    group=groups_by_ticker.get_group(ticker)
    group.set_index('ds',inplace=True)
    group=group.iloc[-3*24:] #오늘 날짜~3일 후까지
    
    resampled_group=group[['y','yhat','yhat_upper','yhat_lower','uncertainty']].resample('10T').interpolate(method='linear')
    resampled_group=resampled_group.reset_index()
    resampled_group['ticker']=group['ticker'].iloc[0]
    
    mer_df=pd.merge(group, resampled_group, on=['ds','ticker','y','yhat','yhat_upper','yhat_lower','uncertainty'], how='outer')
    mer_df=mer_df.sort_values(by='ds')
    dfs_by_ticker=pd.concat([dfs_by_ticker,mer_df])
dfs_by_ticker


# In[156]:


#실시간 데이터 불러오기 (호출 시간: 11:37 최소 20분~최대 1시간)
rltm=rltm_poi()
rltm


# In[ ]:


이상치
-실시간 호출 -> 범위(1~4단계) -> 4단계 (절대적인 값) 
1-적다
2-보통
3-많다
4-아주많다 

-예측치에서 벗어나는 경우(상대적인 탐지)
anomaly :-1 : 


# In[184]:


res=pd.merge(rltm, dfs_by_ticker, on=['ds','ticker'], how='inner')
res['error']=res['y_x']-res['yhat']
res['anomaly'] = res.apply(lambda x: -1 if x['error'] < -1.5 * x['uncertainty'] #적은 인구수
                                   else (1 if x['error'] > 1.2 * x['uncertainty'] #많은 인구수
                                         else 0), axis=1)


# In[185]:


res


# ## MODEL 2 - 날씨 제외 일주일 후까지 예측

# In[55]:


#feature engineering
#날씨 제외 날짜 파생변수로 다변량 예측, 7일후까지 (날씨가 큰 영향을 미치지 않았음)
#logistic => 음수 제한 작동안함, 그냥 음수는 0으로 처리하게끔 코딩해야 할 듯, =>linear가 더 잘 나옴

#Train dataset 불러오기
#con, con_piv=preprocess_api()
#con.columns=['ds','ticker','y']
#train_wea=
#train_wea=train_weather()
#holiday_df=holiday_df
holiday_df=make_holiday_df()
#test_wea=test_weather()
#날씨변수 병합
mul=pd.merge(con, train_wea, on='ds') 

#날짜파생 변수 병합
mul=create_features(mul)
groups_by_ticker=mul.groupby('ticker')
groups_by_ticker.groups.keys()
ticker_list=list(groups_by_ticker.groups.keys())


start_time=time()

for_loop_forecast=pd.DataFrame()

for ticker in ticker_list:
    group=groups_by_ticker.get_group(ticker)
    
    m = Prophet(interval_width=0.8, 
                #changepoint_range=0.2,
                seasonality_mode='multiplicative', #good
                holidays_prior_scale=15,
                holidays=holiday_df) 
   
    #변수 추가
    #m.add_regressor('TMP', standardize=False)
    #m.add_regressor('PCP', standardize=False)
    m.add_regressor('weekday', standardize=False)
    m.add_regressor('season_Winter', standardize=False)
    m.add_regressor('season_Spring', standardize=False)
    m.add_regressor('season_Summer', standardize=False)
    m.add_regressor('season_Fall', standardize=False)
    m.add_regressor('tbin_0', standardize=False)
    m.add_regressor('tbin_1', standardize=False)
    m.add_regressor('tbin_2', standardize=False)
    m.add_regressor('tbin_3', standardize=False)

    #주간 주기성 요소 추가 (*)
    m.add_seasonality(name='weekly', period=7, fourier_order=10)
    
    m.fit(group)
    
    #test dataset 불러오기
    #test_w=pd.concat([train_wea, test_wea]) #날씨 병합
    future=m.make_future_dataframe(periods=8*24, freq='h') #일주일 후까지 예측

    #future=pd.merge(future, test_w[['ds','TMP','PCP']], on='ds', how='outer')
    future=create_features(future)
    forecast=m.predict(future)

    
    #시각화
    fig=m.plot(forecast)
    ax=fig.gca()
    ax.plot(group['ds'],group['y'],'b.')
    
    #performance=pd.merge(group, forecast[['ds','yhat','yhat_lower','yhat_upper']], on='ds')
    mae=mean_absolute_error(group['y'], forecast['yhat'][:-8*24])
    mape=mean_absolute_percentage_error(group['y'], forecast['yhat'][:-8*24])
    print(ticker,'MAE:', mae,'MAPE:',mape)
    
    forecast['ticker']=group['ticker'].iloc[0]
    forecast=forecast[['ds','ticker','yhat','yhat_upper','yhat_lower']]
    
    #전체 장소 합치기
    for_loop_forecast=pd.concat([for_loop_forecast, forecast])
    
    
print('Time:', time()-start_time)


# ## 이상탐지  
# 이 과정을 함수화 할 수 없을까.. 0, 1, -1을 리턴하게끔..

# In[146]:


#음수 값 -> 0으로 대체
for_loop_forecast['yhat']=for_loop_forecast.apply(lambda x: 1e-6 if x['yhat']<0 else x['yhat'], axis=1)
results=for_loop_forecast.copy()

results=pd.merge(mul,for_loop_forecast, on=['ds','ticker'], how='outer') 
results=results[['ds','ticker','y','yhat','yhat_upper','yhat_lower']] 

results['error']=results['y']-results['yhat']
results['uncertainty']=results['yhat_upper']-results['yhat_lower']

#숫자가 작아질 수록 이상치로 정의될 경우가 많아짐. 
#1: 평소보다 많을 때, -1: 평소보다 적을 때

results['anomaly'] = results.apply(lambda x: -1 if x['error'] < -1.5 * x['uncertainty'] 
                                   else (1 if x['error'] > 1.2 * x['uncertainty'] else 0), axis=1)

def two_weeks_ago_mean(x):
    if (pd.notna(results.iloc[x.name-7*24*9]['y'])) & (pd.notna(results.iloc[x.name-7*2*24*9]['y'])):
        return ((results.iloc[x.name-7*24*9]['y'])+(results.iloc[x.name-7*2*24*9]['y']))/2
    else:
        None

results['avg_2w']=results.apply(two_weeks_ago_mean, axis=1) #최근 2주간 평균
results['anomaly2']=results.apply(lambda x: 1 if x['y'] > 2.0 * x['avg_2w'] else 0, axis=1)
results


# In[277]:


import plotly.graph_objects as go

groups_by_ticker=results.groupby('ticker')
groups_by_ticker.groups.keys()
ticker_list=list(groups_by_ticker.groups.keys())
for ticker in ticker_list:
    group=groups_by_ticker.get_group(ticker)
    # 데이터프레임의 시간 열을 x축으로 설정하여 그래프 생성
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=group['ds'], y=group['y'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=group['ds'], y=group['yhat'], mode='lines', name='Predicted'))
    fig.add_trace(go.Scatter(x=group[group['anomaly'] != 0]['ds'], y=group[group['anomaly'] != 0]['y'],
                             mode='markers', name='Anomaly', marker=dict(color='black', size=4)))

    fig.update_layout(title=f'{ticker}: Anomaly Detection', xaxis_title='Date', yaxis_title='Value')
    fig.show()


# In[147]:


import plotly.graph_objects as go

groups_by_ticker=results.groupby('ticker')
groups_by_ticker.groups.keys()
ticker_list=list(groups_by_ticker.groups.keys())
for ticker in ticker_list:
    group=groups_by_ticker.get_group(ticker)
    # 데이터프레임의 시간 열을 x축으로 설정하여 그래프 생성
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=group['ds'], y=group['y'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=group['ds'], y=group['yhat'], mode='lines', name='Predicted'))
    fig.add_trace(go.Scatter(x=group[group['anomaly2'] != 0]['ds'], y=group[group['anomaly2'] != 0]['y'],
                             mode='markers', name='Anomaly', marker=dict(color='black', size=4)))

    fig.update_layout(title=f'{ticker}: Anomaly Detection', xaxis_title='Date', yaxis_title='Value')
    fig.show()


# In[ ]:




