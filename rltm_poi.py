import time
import pandas as pd
import requests, xmltodict
import plotly.express as px
import plotly.graph_objects as go
from plot import save_fig_as_png, save_fig_as_html


#실시간 data 불러오기 - 1. Sk Api
api_keys=["w3oFktJUat2NPpDorwHZE7avbxcvdq0S7wx4vXfN","Tt3yyROHTM8op2hEyv1Z34AXC2x8KPbn1iuD5Hlc",
         "RNM43SFreC8YwWjFIAGHY4VIpOi6jDHG98AHf7rN","e8wHh2tya84M88aReEpXCa5XTQf3xgo01aZG39k5"]
api_key_index=0

def rltm_poi():
    df = pd.DataFrame(columns=['ds','ticker', 'y', 'CongestionLevel'])
    def get_rltm_pois(df):
        # 장소 id 리스트
        ids = [187961, 5783805, 5799875, 384515]
        # SK api 홈페이지에서 호출링크 가져옴
        base_url = "https://apis.openapi.sk.com/puzzle/place/congestion/rltm/pois/"
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
            #query_params = "?date=" + str(date)
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()  # 오류가 있는 경우 예외 발생
                responses[place_id] = response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error occurred: {e}")
                print("Switching to the next API key...")
                switch_to_next_key()
                return get_rltm_pois(df) 


        # for문 reponse 딕셔너리 항목에서 키값으로 데이터 추출하고 각 변수에 저장
        for place_id, response_data in responses.items():
            #poi_id = response_data['contents']['poiId']
            poi_name = response_data['contents']['poiName']
            for item in response_data['contents']['rltm']:
                congestion = item['congestion']
                congestion_level = item['congestionLevel']
                datetime = item['datetime']

                # 'df'에 새로운 데이터 추가하고 인덱스 재설정
                new = pd.DataFrame({
                    #'Id': poi_id,
                    'ds': [datetime],
                    'ticker': [poi_name],
                    'y': [congestion],
                    'CongestionLevel': [congestion_level]
                })
                df=pd.concat([df, new], ignore_index=True)

        return df
    def get_rltm_areas(df):
        ids = [9273,9270]
        base_url = "https://apis.openapi.sk.com/puzzle/place/congestion/rltm/areas/"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "appKey":api_keys[api_key_index]
        }

        responses = {}
        for areas_id in ids:
            url = base_url + str(areas_id)
            #query_params = "?date=" + str(date)
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()  # 오류가 있는 경우 예외 발생
                responses[areas_id] = response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error occurred: {e}")
                print("Switching to the next API key...")
                switch_to_next_key()
                return get_rltm_areas(df)  


        for areas_id, response_data in responses.items():
            #area_id = response_data['contents']['areaId']
            area_name = response_data['contents']['areaName']

            congestion = response_data['contents']['rltm']['congestion']
            congestion_level = response_data['contents']['rltm']['congestionLevel']
            datetime = response_data['contents']['rltm']['datetime']
            new = pd.DataFrame({
                #'Id': area_id,
                'ds': [datetime],
                'ticker': [area_name],
                'y': [congestion],
                'CongestionLevel': [congestion_level],
            })
            df=pd.concat([df, new], ignore_index=True)

        return df
    
    def switch_to_next_key():
        # 다음 API 키로 전환하는 함수
        global api_key_index
        api_key_index = (api_key_index + 1) % len(api_keys)

    
    df = get_rltm_pois(df)
    df = get_rltm_areas(df)
    df.ds=pd.to_datetime(df.ds)
    df=df.fillna(0)
    df['y_100m2']=df['y']*100
    
    return df


def append_rltm_data():
    df = pd.read_csv('rltm_data.csv')
    if len(df)<144:
        df_new=rltm_poi()
        df=pd.concat([df, df_new], ignore_index=True)
        df.ds=pd.to_datetime(df.ds)
        print("New data fetched and appended.")
        
        #레벨
        bins=[0,0.0175,0.035,0.21,0.4,float('inf')]
        labels=['여유','보통','조금 혼잡','혼잡','매우 혼잡']
        num_labels=[1,2,3,4,5]
        df['level']=pd.cut(df['y'], bins=bins, labels=labels)
        df['num_level']=pd.cut(df['y'], bins=bins, labels=num_labels)
        
        #시각화(png, html로 저장)
        visualize_rltm_data(df)

        df.to_csv('rltm_data.csv', index=False)
    
    else:
        df=df.iloc[0:0]
        df.to_csv('rltm_data.csv',index=False)
        append_rltm_data()
        



def visualize_rltm_data(rltm):
    # 예측데이터 
    total_pred = pd.read_csv("total_pred.csv")
    
    # 날짜시간 타입으로 변경
    total_pred['ds'] = pd.to_datetime(total_pred['ds'])
    total_pred=total_pred.set_index(total_pred['ds'], drop=True)
    
    # 100제곱미터당 0-9명 단위로 나타나도록 예측값 변경
    total_pred['pred_100m2'] = round(total_pred['yhat']*100,1) #소수점 첫째자리까지
    
    #레벨
    bins=[0,0.0175,0.035,0.21,0.4,float('inf')]
    labels=['여유','보통','조금 혼잡','혼잡','매우 혼잡']
    num_labels=[1,2,3,4,5]
    
    total_pred['level']=pd.cut(total_pred['yhat'], bins=bins, labels=labels)
    total_pred['num_level']=pd.cut(total_pred['yhat'], bins=bins, labels=num_labels)

    ticker_list=list(total_pred.ticker.unique())
    for ticker in ticker_list:
        pred=total_pred.groupby('ticker').get_group(ticker)
        rltm_df=rltm[rltm['ticker']==ticker]
        print(ticker)

        print(pred.index[0].date().strftime('%m월 %d일'))
        day_pred=pred[0:0+24][['pred_100m2','level','num_level']] #하루치(today) 예측값, index에는 시간
        #print(day_pred)
        time_values=day_pred.index.strftime('%H시')
        rltm_time_values=rltm_df.ds.dt.strftime('%H시')
        rltm_values=round(rltm_df['y_100m2'],1) #소수점 첫째자리까지
        
        fig=go.Figure()
        fig.add_trace(go.Scatter(
                x=time_values,
                y=day_pred['pred_100m2'],
                mode='lines+markers',
                marker={'symbol':'circle', 'size': 8},
                name='예측',
                #fill='tozeroy',  # 면적 채우기 설정
                #name=ticker,
                hovertemplate='%{x}: %{y}명<br>%{text}단계: %{customdata}',
                text=day_pred['num_level'],
                customdata=day_pred['level']
            ))

       
        fig.add_trace(go.Bar(
                x=rltm_time_values,
                y=rltm_values,
                name='실시간',
                hovertemplate='%{y}명<br>%{text}단계: %{customdata}',
                text=rltm_df['num_level'],
                customdata=rltm_df['level'],
            ))


        fig.update_layout(
            title={'text': ticker, 'x': 0.5},
            xaxis={'title': None, 'tickformat': '%H시',
                    'tickmode': 'array',  # 눈금을 배열 모드로 설정
                    'tickvals': time_values[::3],  # 3시간 간격으로 눈금 값 설정
                    'ticktext': time_values[::3],  # 눈금에 표시될 텍스트 설정
                    'tickangle': 0,  # 눈금 텍스트의 회전 각도 설정
                  },
            yaxis={'title': '혼잡도(명/100㎡)', 'showgrid':True, 'gridcolor':'lightgray'},
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
        )
        #fig.show()
        
        #이미지로 저장
        image_filename = f"{ticker}_{pred.index[0].date().strftime('%m%d')}.png"
        save_fig_as_png(fig, image_filename)
        print(f"Image saved: {image_filename}")

        #HTML로 저장
        html_filename = f"{ticker}_{pred.index[0].date().strftime('%m%d')}.html"
        save_fig_as_html(fig, html_filename)
        print(f"Html saved: {html_filename}")
        

if __name__=='__main__':
    df=rltm_poi()
    rltm=append_rltm_data()
    #print(rltm)