import pandas as pd
import numpy as np
import holidays
from sklearn.neighbors import NearestNeighbors
import collections

import data.jeju_traffic_data as jt



def split_date(data):

    # 연월일 구분
    data['base_date'] = pd.to_datetime(data['base_date'].astype(str), format='%Y%m%d')

    # 연/월/일 컬럼 생성후 데이터에 새로운 feature로 저장
    data['year']  = data['base_date'].dt.year
    data['month'] = data['base_date'].dt.month
    data['day']   = data['base_date'].dt.day

    return data



def set_is_holiday(data):

    # 한국 공휴일 불러오기 (2021, 2022)
    kr_holidays = holidays.KR(years=[2021, 2022])

    # 공휴일 날짜 문자열 리스트 (YYMMDD 형식)
    holiday_str = [d.strftime('%y%m%d') for d in kr_holidays]

    # 플래그 초기화
    data['is_holiday'] = 0

    # base_date를 YYMMDD 문자열로 변환
    data['base_date_str'] = data['base_date'].dt.strftime('%y%m%d')

    # 공휴일 날짜 매칭
    data.loc[data['base_date_str'].isin(holiday_str), 'is_holiday'] = 1

    # 요일 정보가 '토' or '일'이면 휴일로 간주
    data.loc[data['day_of_week'].isin(['토', '일']), 'is_holiday'] = 1

    # 임시로 만든 컬럼 삭제
    data.drop('base_date_str', axis=1, inplace=True)

    return data



def set_season(data):

    # 계절(season) 문자열 컬럼 생성 → 원-핫 인코딩
    # month 기준으로 봄(3~5), 여름(6~9), 가을(10~11), 겨울(12~2) 분류
    # 실제 계절하고 조금 다른긴한데 우리가 8월 예측해야하므로 앞뒤인 7,9월만을 여름으로 가정
    def get_season(m):
        if 3 <= m <= 5:
            return 'spring'
        elif 6 <= m <= 9:
            return 'summer'
        elif 10 <= m <= 11:
            return 'autumn'
        else:
            return 'winter'

    # 문자열 계절 컬럼
    data['season'] = data['month'].apply(get_season)

    # 원-핫 인코딩
    season_ohe = pd.get_dummies(data['season'], prefix='season')

    data = pd.concat([data, season_ohe], axis=1)

    # 원-핫 처리 후 문자열 컬럼 삭제
    data.drop('season', axis=1, inplace=True)

    return data



def set_time(data):

    # 시간 cos/sin 변환 추가 - 시간의 주기성
    data['cos_time'] = np.cos(2*np.pi*(data['base_hour']/24))
    data['sin_time'] = np.sin(2*np.pi*(data['base_hour']/24))

    return data



def merge(data):

    # base_hour별 target 평균을 train에서 계산해 → test에는 merge만 함
    df_hour = data.groupby('base_hour')['target'].mean().reset_index().rename(columns={'target':'hour_mean_target'})
    data = data.merge(df_hour, on='base_hour', how='left')

    return data



def merge_2(data):

    # [시간 + 제한속도] 조합 평균 통행시간
    df_whs = data.groupby(['base_hour','maximum_speed_limit'])['target'].mean().reset_index().rename(columns={'target':'whs_mean_target'})
    data = data.merge(df_whs, on=['base_hour','maximum_speed_limit'], how='left')

    return data



def incode_road_rating(data):

    # lane_count 자체는 그대로 써도 될듯
    # road_rating 원핫벡터화
    # road_rating 칼럼 일단 그대로 둠 - 마지막 단계에서 없애야할듯
    data['road_rating_103'] = (data['road_rating']==103).astype('int8')
    data['road_rating_106'] = (data['road_rating']==106).astype('int8')
    data['road_rating_107'] = (data['road_rating']==107).astype('int8')

    return data



def set_multi_speed_penalty(data):

    # multi_linked
    # target과 상관관게 없어 보여서 그냥 그대로 둠
    # 고속도로 기준 최대 상한 속도에서 얼마나 손해 보나?
    # 여러가지 칼럼 해봤는데 상관계수 0.48이라 쓸만한듯
    data['multi_speed_penalty'] = data['multi_linked'] * (120 - data['maximum_speed_limit'])

    return data



def set_speed_weight(data):

    # 제한속도가 높고, 중량제한이 클수록 도로 위 제약이 적어질수 있음
    # 중량 제한이 커지면 -> 트럭이 많아지니까 - 속도 느려질수도 있다
    data['speed_weight'] = data['maximum_speed_limit'] * data['weight_restricted']

    return data



def set_weight(data):

    # 중량제한 정보 유무 구분
    data['has_weight_info'] = (data['weight_restricted'] > 0).astype('int8')  # 0 → 정보 없음, 1 → 정보 있음

    # 중량제한 등급화 (정보 있는 경우만 구간화)

    data['weight_class'] = pd.cut(
        data['weight_restricted'],
        bins=[0, 32400, 43200, 60000],  # 0은 제외 (결측 취급)
        labels=[1, 2, 3]
    )

    # 정보 없는 구간(0.0)은 별도 코드 0으로 채우기
    data['weight_class'] = data['weight_class'].cat.add_categories([0])  # 0 추가
    data['weight_class'] = data['weight_class'].fillna(0).astype('int8')

    return data



def replace_by_road_weight(data):

    data.loc[(data['road_rating'] == 107) & (data['weight_restricted'] == 32400.0) & (data['road_name'] == "-"), 'road_name'] = "산서로"
    data.loc[(data['road_rating'] == 107) & (data['weight_restricted'] == 43200.0) & (data['road_name'] == "-"), 'road_name'] = "중문로"

    return data



def replace_by_latitude(data):

    latitude_start = [33.409416, 33.402546, 33.471164, 33.411255, 33.405319,
                    33.322018, 33.325096, 33.408431, 33.284189, 33.47339]

    road_name = ['산서로', '지방도1119호선', '일반국도12호선', '산서로', '산서로',
                '중문로', '중문로', '산서로', '중문로', '일반국도12호선']

    for i in range(len(latitude_start)):
        data.loc[(data['start_latitude'] == latitude_start[i]) & (data['road_name'] == '-'), 'road_name'] = road_name[i]

    return data



def replace_by_longitude(data):

    longitude_end = [126.261797, 126.259693]

    road_name = ['산서로', '산서로']

    for i in range(len(longitude_end)):
        data.loc[(data['end_longitude'] == longitude_end[i]) & (data['road_name'] == '-'), 'road_name'] = road_name[i]

    return data
    # 위도/경도별 KNN을 이용하여 대체



def replace_by_knn(train):

    # 결측치와 비결측치를 분리
    # 결측치 데이터
    missing_mask = train['road_name'] == '-'
    # 비결측치 데이터
    not_missing_mask = ~missing_mask

    # KNN 모델 학습: 비결측치 데이터가지고 유클리드 거리로 학습
    feature_cols = ['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']
    knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
    knn.fit(train.loc[not_missing_mask, feature_cols])

    # 결측치 행에 대해 KNN으로 가장 가까운 이웃 찾기
    # indices는 결측치가 있는 행의 가장 가까운 이웃들의 인덱스번호 리스트
    distances, indices = knn.kneighbors(train.loc[missing_mask, feature_cols])

    # 각 결측치 행에 대해 이웃들의 road_name 최빈값으로 대체
    road_names_array = train.loc[not_missing_mask, 'road_name'].values # 결측치가 아닌 road_name값들만 모은 벡터

    fill_values = []

    for idx_list in indices:
        neighbor_names = road_names_array[idx_list]  # neighbor_names에 idx_list에 해당하는 값만 불러오기
        mode_name = collections.Counter(neighbor_names).most_common(1)[0][0] # k를 3개했으므로 3개의 값중 최빈값을 추출
        fill_values.append(mode_name) # fill_values에다가 추가

    # 5. 결측치 채우기
    train.loc[missing_mask, 'road_name'] = fill_values

    return train



def remove_outlier(train):
    train = train[train['target'] < 100]
    return train


def preprocess(data):

    print("split_date start")
    data = split_date(data)
    print("split_date end")
    print("set_is_holiday start")
    data = set_is_holiday(data)
    print("set_is_holiday end")
    print("set_season start")
    data = set_season(data)
    print("set_season end")
    print("set_time start")
    data = set_time(data)
    print("set_time end")
    print("merge start")
    data = merge(data)
    print("merge end")
    print("merge_2 start")
    data = merge_2(data)
    print("merge_2 end")
    print("set_multi_speed_penalty start")
    data = set_multi_speed_penalty(data)
    print("set_multi_speed_penalty end")
    print("set_speed_weight start")
    data = set_speed_weight(data)
    print("set_speed_weight end")
    print("set_weight start")
    data = set_weight(data)
    print("set_weight end")
    print("replace_by_road_weight start")
    data = replace_by_road_weight(data)
    print("replace_by_road_weight end")
    print("replace_by_latitude start")
    data = replace_by_latitude(data)
    print("replace_by_latitude end")
    print("replace_by_longitude start")
    data = replace_by_longitude(data)
    print("replace_by_longitude end")

    print("data split start")
    date_condition = data['base_date'] < pd.to_datetime('20220701', format='%Y%m%d')
    train = data[date_condition].copy()
    test = data[~date_condition].copy()
    print("data split end")
    
    print("replace_missing_data start")
    train = replace_by_knn(train)
    print("replace_missing_data end")
    print("remove_outlier start")
    train = remove_outlier(train)
    print("remove_outlier end")
    return train, test



print("data load start")
whole_data, _ = jt.load(data_folder="external")
print("data load end")

whole_data.drop('id', axis=1)

print("preprocessing start")
train, test = preprocess(whole_data)
print("preprocessing end")

print("data save start")
train.to_csv('train_preprocessed.csv', index=False)
test.to_csv('test_preprocessed.csv', index=False)
print("data save end")