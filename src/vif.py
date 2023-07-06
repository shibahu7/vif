import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from statsmodels.stats.outliers_influence import variance_inflation_factor


def main():

    # サンプルデータを用意
    dataset = fetch_california_housing()

    # 標本データを取得
    data_x = pd.DataFrame(dataset.data, columns=dataset.feature_names)

    # 正解データを取得
    data_y = pd.DataFrame(dataset.target, columns=['target'])

    # vifを計算する
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(
        data_x.values, i) for i in range(data_x.shape[1])]
    vif["features"] = data_x.columns

    # vifを計算結果を出力する
    print(vif)

    # vifをグラフ化する
    plt.plot(vif["VIF Factor"])
    plt.show()


if __name__ == "__main__":
    main()
