import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import jeju_traffic_data as jt

import plotly.graph_objs as go
import plotly.express as px

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)



# data infomation
def print_data_information(train, test):

    # data size
    print("Train Data Size :", train.shape)
    print("Test Data Size :", test.shape)
    print()

    # missing data
    print("Missing Train Data")
    print(train.isnull().sum())
    print("Total :", sum(train.isnull().sum()))
    print()
    print("Missing Test Data")
    print(test.isnull().sum())
    print("Total :", sum(test.isnull().sum()))
    print()

    # data central tendency
    print(train.describe())
    print()



# data distribution
def plot_data_distribution(train):

    # base_date
    plt.boxplot(train[["base_date"]])
    plt.title("base_date")
    plt.show()

    # day_of_week
    category_orders_dict = dict(day_of_week=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
    fig = px.histogram(train, x="day_of_week", category_orders=category_orders_dict)
    fig.show()

    # base_hour
    fig = px.histogram(train, x="base_hour")
    fig.show()

    # lane_count
    fig = px.histogram(train, x="lane_count")
    fig.show()

    # road_rating
    fig = px.histogram(train, x="road_rating")
    fig.show()

    # road_name
    fig = px.histogram(train, x="road_name")
    fig.show()

    # multi_linked
    fig = px.histogram(train, x="multi_linked")
    fig.show()

    # connect_code
    fig = px.histogram(train, x="connect_code")
    fig.show()

    # maximum_speed_limit
    fig = px.histogram(train, x="maximum_speed_limit")
    fig.show()

    # weight_restricted
    fig = px.histogram(train, x="weight_restricted")
    fig.show()

    # road_type
    fig = px.histogram(train, x="road_type")
    fig.show()

    # start_node_name
    fig = px.histogram(train, x="start_node_name")
    fig.show()

    # start_latitude
    plt.boxplot(train[["start_latitude"]])
    plt.title("start_latitude")
    plt.show()

    # start_longitude
    plt.boxplot(train[["start_longitude"]])
    plt.title("start_longitude")
    plt.show()

    # end_node_name
    fig = px.histogram(train, x="end_node_name")
    fig.show()

    # end_latitude
    plt.boxplot(train[["end_latitude"]])
    plt.title("end_latitude")
    plt.show()

    # end_longitude
    plt.boxplot(train[["end_longitude"]])
    plt.title("end_longitude")
    plt.show()

    # target
    plt.boxplot(train[["target"]])
    plt.title("target")
    plt.show()



# data correlation
def plot_data_correlation(train):

    # 요일별 평균 속도
    plt.figure(figsize=(8, 5))
    sns.barplot(data=train, x="day_of_week", y="target", ci=None, order=["월", "화", "수", "목", "금", "토", "일"])
    plt.title("요일별 평균 속도")
    plt.ylabel("평균 속도 (km/h)")
    plt.xlabel("요일")
    plt.tight_layout()
    plt.show()

    # 시간대별 평균 속도
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=train, x="base_hour", y="target", marker="o")
    plt.title("시간대별 평균 속도")
    plt.ylabel("평균 속도 (km/h)")
    plt.xlabel("시간대")
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 도로 등급별 평균 속도
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=train, x="road_rating", y="target")
    plt.title("도로 등급별 평균 속도 분포")
    plt.ylabel("평균 속도")
    plt.xlabel("도로 등급")
    plt.tight_layout()
    plt.show()

    # 차로 수와 평균 속도
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=train, x="lane_count", y="target", alpha=0.6)
    plt.title("차로 수와 평균 속도 관계")
    plt.xlabel("차로 수")
    plt.ylabel("평균 속도")
    plt.tight_layout()
    plt.show()

    # 제한 속도와 평균 속도
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=train, x="maximum_speed_limit", y="target", alpha=0.6)
    plt.title("제한 속도와 평균 속도 관계")
    plt.xlabel("제한 속도 (km/h)")
    plt.ylabel("평균 속도 (km/h)")
    plt.tight_layout()
    plt.show()



def main():

    train, test = jt.load()

    print_data_information(train, test)
    plot_data_distribution(train)
    plot_data_correlation(train)



main()