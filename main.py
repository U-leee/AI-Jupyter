import time
import pandas as pd
from model import Model
from plot import visualize_future_data

if __name__=='__main__':
    #모델 실행 model.py -> total_pred.csv 생성됨
    start_time=time.time()
    model=Model()
    total_pred=model.total_predictions()
    print('Time:', time.time()-start_time)

    time.sleep(1)
    
    # 100제곱미터당 0-9명 단위로 나타나도록 예측값 변경
    total_pred['pred_100m2'] = total_pred['yhat']*100
    
    #레벨
    bins=[0,0.0175,0.035,0.21,0.4,float('inf')]
    labels=['여유','보통','조금 혼잡','혼잡','매우 혼잡']
    num_labels=[1,2,3,4,5]
    
    total_pred['level']=pd.cut(total_pred['yhat'], bins=bins, labels=labels)
    total_pred['num_level']=pd.cut(total_pred['yhat'], bins=bins, labels=num_labels)
    
    #t+1~ 그래프 시각화, html_files와 images 폴더에 저장
    visualize_future_data(total_pred)