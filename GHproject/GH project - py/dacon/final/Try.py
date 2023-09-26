import os
def delete_files_in_directory(directory):
    files_in_directory = os.listdir(directory)
    for file in files_in_directory:
        file_path = os.path.join(directory, file)  # 파일의 전체 경로를 얻습니다.
        if os.path.isfile(file_path):  # 파일인 경우만 삭제합니다.
            os.remove(file_path)
delete_files_in_directory('dacon/final/Loaded data/')
delete_files_in_directory('dacon/final/Model result/')


########################################################
# 0. 여기만 입력하세요.
ticker = 'nvda' # 소문자로 입력해야 합니다 아니면 FS 뽑을때 오류
start_date = '2013-09-28'
end_date = '2023-09-09'
########################################################
'''
사용 설명서

Try.py 를 실행하게되면 예측을 하려는 나스닥 상장회사에 대한 Ticker 와 Start Date, End Date 를 입력해주세요 (소문자로 입력해야 합니다.)
나머지는 자동입니다. Try.py 전체코드를 실행만 시키세요. (맨 위 코드는 다른 기업에 대한 데이터를 불러올때 빈 폴더에 담기도록 하기 위함입니다. 그냥 진행시켜 주세요.)

# 경로는 Load Data.py 에 327번 라인에서 한번에 설정할 수 있습니다. 


Load data -> Technical과 Fundamental 에 대한 LSTM 모델 -> 앙상블 모델 순으로 진행이 됩니다.

정상적으로 작동이 된다면, Load data에서는 총 4개의 csv 파일이 저장되며, 경제지표와 재무제표에서 보간법을 이용한 그래프를 확인할 수 있습니다. 
                     Tech / Funda 모델에서는 예측모델의 시각화와, 오버핏을 확인할 수 있는 학습곡선 그래프를 확인할 수 있습니다.
                     Ensemble 모델에서는 최종 예측모델에 대한 시각화와, 오버핏을 확인할 수 있는 두가지 비주얼 그래프를 확인 할 수 있습니다.
                       - 또한, 터미널에 3가지 모델에 대한 다음날 주가예측을 실수 형태로 확인할 수 있습니다.
'''


filenames = ['dacon/final/Load Data.py',
             'dacon/final/Model/Technical analysis.py', 
             'dacon/final/Model/Fundamental analysis.py',              
             'dacon/final/Model/Stacking Ensemble.py']

for filename in filenames:
    with open(filename, 'r') as file:
        exec(file.read())
    
