import pandas as pd
from load_poi import load_poi

def train_dataset():
    origin=pd.read_csv('congestion3.csv') #2일전까지 데이터
    new=load_poi()                       #1일전(어제)데이터, api 불러오기
    mer_df=pd.concat([origin, new])
    mer_df.ds=pd.to_datetime(mer_df.ds)
    mer_df=mer_df.fillna(0)
    mer_df=mer_df[['ds','ticker','y']]
    mer_df=mer_df.drop_duplicates() 
    mer_df.to_csv('congestion3.csv', index=False) #저장
    
    return mer_df

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

if __name__=='__main__':
    df=train_dataset()
    df