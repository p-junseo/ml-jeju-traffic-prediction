import pandas as pd
import os 
from dotenv import load_dotenv

# 데이터 로드 함수
def load(data_folder="raw"):
    
    # 환경변수 로딩
    load_dotenv()

    team_project_folder = os.getenv('TEAM_PROJECT')

    # train, test 데이터 로딩
    train = pd.read_csv(team_project_folder + "\\data\\" + data_folder + "\\train.csv")
    test = pd.read_csv(team_project_folder + "\\data\\" + data_folder + "\\test.csv")

    return train, test