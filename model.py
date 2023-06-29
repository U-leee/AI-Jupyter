import pandas as pd
from time import time
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from data import train_dataset, create_features
from weather import train_weather, test_weather
from holiday import make_holiday_df

class Model():
    def __init__(self):
        self.con=train_dataset()
        self.train_wea=train_weather()
        self.test_wea=test_weather()
        self.holiday_df=make_holiday_df()
        #self.con=create_features(self.con)
        #self.mul=pd.merge(self.con, self.train_wea, on='ds')

        #self.groups_by_ticker=self.mul.groupby('ticker')
        #self.ticker_list=list(self.groups_by_ticker.groups.keys())
        #self.for_loop_forecast=pd.DataFrame()

    def model1(self):
        #start_time=time()
        print(self.con)
        mul=pd.merge(self.con, self.train_wea, on='ds')
        mul=create_features(mul)
        groups_by_ticker=mul.groupby('ticker')
        ticker_list=list(groups_by_ticker.groups.keys())
        
        for_loop_forecast=pd.DataFrame()

        for ticker in ticker_list:
            group=groups_by_ticker.get_group(ticker)
            print(len(group))
            m = Prophet(interval_width=0.8, 
                        seasonality_mode='multiplicative', 
                        holidays_prior_scale=15,
                        holidays=self.holiday_df) 

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
            test_w=pd.concat([self.train_wea, self.test_wea]) #날씨 병합 888+72
            future=m.make_future_dataframe(periods=3*24, freq='h')
            future=create_features(future)
            future=pd.merge(future, test_w[['ds','TMP','PCP']], on='ds', how='outer')

            forecast=m.predict(future)


            #시각화
            fig=m.plot(forecast)
            ax=fig.gca()
            ax.plot(group['ds'],group['y'],'b.')

            comp_plot=m.plot_components(forecast)
            plt.show()

            #performance=pd.merge(group, forecast[['ds','yhat','yhat_lower','yhat_upper']], on='ds')
            mae=mean_absolute_error(group['y'], forecast['yhat'][:-72])
            mape=mean_absolute_percentage_error(group['y'], forecast['yhat'][:-72])
            print(ticker,'MAE:', mae,'MAPE:',mape)

            forecast['ticker']=group['ticker'].iloc[0]
            forecast=forecast[['ds','ticker','yhat','yhat_upper','yhat_lower']]

            #전체 장소 합치기
            for_loop_forecast=pd.concat([for_loop_forecast, forecast])


        #print('Time:', time()-start_time)
        pred=self.postprocessing(for_loop_forecast, mul)
        pred=pred.iloc[-3*24*6:]

        return pred


    def model2(self):
        #start_time=time()
        
        mul=pd.merge(self.con, self.train_wea, on='ds')
        mul=create_features(mul)
        groups_by_ticker=mul.groupby('ticker')
        ticker_list=list(groups_by_ticker.groups.keys())
        
        for_loop_forecast=pd.DataFrame()

        for ticker in ticker_list:
            group=groups_by_ticker.get_group(ticker)

            model = Prophet(interval_width=0.8, 
                        seasonality_mode='multiplicative', #good
                        holidays_prior_scale=15,
                        holidays=self.holiday_df) 

            #변수 추가
            #m.add_regressor('TMP', standardize=False)
            #m.add_regressor('PCP', standardize=False)
            model.add_regressor('weekday', standardize=False)
            model.add_regressor('season_Winter', standardize=False)
            model.add_regressor('season_Spring', standardize=False)
            model.add_regressor('season_Summer', standardize=False)
            model.add_regressor('season_Fall', standardize=False)
            model.add_regressor('tbin_0', standardize=False)
            model.add_regressor('tbin_1', standardize=False)
            model.add_regressor('tbin_2', standardize=False)
            model.add_regressor('tbin_3', standardize=False)

            #주간 주기성 요소 추가 (*)
            model.add_seasonality(name='weekly', period=7, fourier_order=10)

            model.fit(group)

            #test dataset 불러오기
            #test_w=pd.concat([train_wea, test_wea]) #날씨 병합
            future2=model.make_future_dataframe(periods=8*24, freq='h') #일주일 후까지 예측

            #future=pd.merge(future, test_w[['ds','TMP','PCP']], on='ds', how='outer')
            future2=create_features(future2)
            forecast2=model.predict(future2)


            #시각화
            fig=model.plot(forecast2)
            ax=fig.gca()
            ax.plot(group['ds'],group['y'],'b.')

            comp_plot2=model.plot_components(forecast2)
            plt.show()

            #performance=pd.merge(group, forecast[['ds','yhat','yhat_lower','yhat_upper']], on='ds')
            mae=mean_absolute_error(group['y'], forecast2['yhat'][:-8*24])
            mape=mean_absolute_percentage_error(group['y'], forecast2['yhat'][:-8*24])
            print(ticker,'MAE:', mae,'MAPE:',mape)

            forecast2['ticker']=group['ticker'].iloc[0]
            forecast=forecast2[['ds','ticker','yhat','yhat_upper','yhat_lower']]

            #전체 장소 합치기
            for_loop_forecast=pd.concat([for_loop_forecast, forecast2])


        #print('Time:', time()-start_time)
        pred2=self.postprocessing(for_loop_forecast, mul)
        pred2=pred2.iloc[-5*24*6:]

        return pred2


    def postprocessing(self, for_loop_forecast, mul):
        #예측 음수 값 -> 0으로 대체
        for_loop_forecast['yhat']=for_loop_forecast.apply(lambda x: 1e-6 if x['yhat']<0 else x['yhat'], axis=1)
        for_loop_forecast['yhat_upper']=for_loop_forecast.apply(lambda x: 1e-6 if x['yhat_upper']<0 else x['yhat_upper'], axis=1)
        for_loop_forecast['yhat_lower']=for_loop_forecast.apply(lambda x: 1e-6 if x['yhat_lower']<0 else x['yhat_lower'], axis=1)
        for_loop_forecast.sort_values(by=['ds','ticker'], inplace=True)
        total_forecast=for_loop_forecast.copy()

        
        #실제값('y')열과 합치기
        predictions=pd.merge(mul,total_forecast, on=['ds','ticker'], how='outer')
        predictions=predictions[['ds','ticker','y','yhat','yhat_upper','yhat_lower']]
        predictions.sort_values(by=['ds','ticker'], inplace=True)

        return predictions

    def total_predictions(self):
        #예측값 병합(모델1, 2)
        pred=self.model1()
        pred2=self.model2()
        total_pred=pd.concat([pred, pred2])
        #예측값 저장
        total_pred.to_csv('total_pred.csv', index=False)

        return total_pred
    

if __name__=='__main__':
    start_time=time()
    model=Model()
    total_pred=model.total_predictions()
    print('Time:', time()-start_time)