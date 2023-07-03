import os
import shutil
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


def clear_image_folder():
    # 이미지 저장 폴더 경로
    save_dir = "images"

    # 폴더가 존재하면 비우기
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f"Cleared image folder: {save_dir}")

def clear_html_folder():
    # HTML 저장 폴더 경로
    html_dir = "html_files"

    # 폴더가 존재하면 비우기
    if os.path.exists(html_dir):
        shutil.rmtree(html_dir)
        print(f"Cleared image folder: {html_dir}")


def save_fig_as_png(fig, filename):
    # 이미지 저장 폴더 경로
    save_dir = "images"

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 이미지 파일 경로
    file_path = os.path.join(save_dir, filename)

    # 이미지로 저장
    fig.write_image(file_path, format="png")

    
def save_fig_as_html(fig, filename):
    # 이미지 저장 폴더 경로
    save_dir = "html_files"

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 이미지 파일 경로
    file_path = os.path.join(save_dir, filename)
    
    pio.write_html(fig, file=file_path)
    print(f"Figure saved as HTML: {file_path}")
    
   

def visualize_future_data(total_pred):
    
    # 날짜시간 타입으로 변경
    total_pred['ds'] = pd.to_datetime(total_pred['ds'])
    total_pred=total_pred.set_index(total_pred['ds'], drop=True)

    ticker_list=list(total_pred.ticker.unique())
    
    #이미지 저장 전에 폴더 비우기
    clear_image_folder()
    clear_html_folder()
    
    for ticker in ticker_list:
        pred=total_pred.groupby('ticker').get_group(ticker)
        print(ticker)
        for i in range(24,len(pred),24):
            print(pred.index[i].date().strftime('%m월 %d일'))
            day_pred=pred[i:i+24][['pred_100m2','level','num_level']] #하루치 예측값, index에는 시간
            #print(day_pred)
            time_values=day_pred.index.strftime('%H시')

            fig=go.Figure(go.Scatter(
                x=time_values,
                y=day_pred['pred_100m2'],
                mode='lines+markers',
                marker={'symbol':'circle', 'size': 8},
                fill='tozeroy',  # 면적 채우기 설정
                #name=ticker,
                hovertemplate='%{x}: %{y}명<br>%{text}단계: %{customdata}',
                text=day_pred['num_level'],
                customdata=day_pred['level']
            ))

            fig.update_layout(
                #title={'text': ticker, 'x': 0.5},
                xaxis={'title': None, 'tickformat': '%H시',
                    'tickmode': 'array',  # 눈금을 배열 모드로 설정
                    'tickvals': time_values[::3],  # 3시간 간격으로 눈금 값 설정
                    'ticktext': time_values[::3],  # 눈금에 표시될 텍스트 설정
                    'tickangle': 0,  # 눈금 텍스트의 회전 각도 설정
                      },
                yaxis={'title': '혼잡도(명/100㎡)', 'showgrid':True, 'gridcolor':'lightgray'},
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False,
            )
            #fig.show()
            
            #이미지로 저장
            image_filename = f"{ticker}_{pred.index[i].date().strftime('%m%d')}.png"
            save_fig_as_png(fig, image_filename)
            print(f"Image saved: {image_filename}")
            
            #HTML로 저장
            html_filename = f"{ticker}_{pred.index[i].date().strftime('%m%d')}.html"
            save_fig_as_html(fig, html_filename)
            print(f"Image saved: {html_filename}")


            
def visualize_areas(total_pred):
    total_pred['ds'] = pd.to_datetime(total_pred['ds'])
    total_pred=total_pred[['ds','ticker','pred_100m2']]
    df=total_pred.pivot_table('pred_100m2','ds','ticker')

    ticker_list=list(total_pred.ticker.unique())

    for day in range(0,len(df),24):
        print(df.index[day].date().strftime('%m월 %d일'))
        day_pred=df.iloc[day:day+24]
        #print(day_pred)
        time_values=day_pred.index.strftime('%H시')
        fig = make_subplots(rows=len(ticker_list)//2 + len(ticker_list) % 2, cols=2, subplot_titles=ticker_list)

        for i, ticker in enumerate(ticker_list):
            row = i // 2 + 1
            col = i % 2 + 1

            fig_area = go.Scatter(
                x=time_values,
                y=day_pred[ticker],
                mode='lines',
                fill='tozeroy',
                hovertemplate='%{x}: %{y}명',
            )
            fig.add_trace(fig_area, row=row, col=col)

            fig.update_xaxes(
                showgrid=False,
                gridcolor='lightgray',
                tickformat='%H시',
                tickmode='array',
                tickvals = time_values[::3],  # 3시간 간격으로 눈금 값 설정
                ticktext = time_values[::3],  # 눈금에 표시될 텍스트 설정
                row=row,
                col=col
            )
            y_min=df.min().min()
            y_max=df.max().max()
            y_range=[y_min, y_max]
            fig.update_yaxes(
                showgrid=True,
                gridcolor='lightgray',
                title='혼잡도(명/100㎡)',
                title_font={'size':9},
                #range=y_range,
                row=row,
                col=col
            )

            fig.update_layout(
                plot_bgcolor='white',
                showlegend=False
            )

            #fig.show()

        #이미지로 저장
        image_filename = f"areas_{df.index[day].date().strftime('%m%d')}.png"
        save_fig_as_png(fig, image_filename)
        print(f"Image saved: {image_filename}")

        #HTML로 저장
        html_filename = f"areas_{df.index[day].date().strftime('%m%d')}.html"
        save_fig_as_html(fig, html_filename)
        print(f"Image saved: {html_filename}")
    
           
            
if __name__=='__main__':
    visualize_future_data(total_pred)
    visualize_areas(total_pred)
