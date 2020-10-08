# python 3.8.3
import os
import numpy as np
import pandas as pd
import requests as req
from bs4 import BeautifulSoup
import json
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, SGDRegressor
import webbrowser

oil_data = {'latest.csv': 'https://vipmember.tmtd.cpc.com.tw/mbwebs/ShowHistoryPrice_oil.aspx',
            '2019.csv': 'https://vipmember.tmtd.cpc.com.tw/mbwebs/ShowHistoryPrice_oil2019.aspx'}


def guard_dir(dir_name):
    # 檢查資料夾是否存在，若無則建立
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)


def _plot(data, title, ylabel, labels):
    # 將資料以不同顏色與label畫在同一張圖中並存檔
    color = ['m', 'c', 'r', 'b']
    plt.figure(figsize=(12.8, 7.2))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.axis('auto')
    for i in range(len(data)):
        plt.plot(data[i], color=color[i], label=labels[i])
    plt.legend(loc='upper left')
    plt.margins(x=0)
    plt.savefig(f'./img/{title}.png')


def split_data(data, n_step):
    # 將資料以n_step的大小切割成data與label，再轉乘numpy陣列回傳
    cnt = len(data) - n_step
    x, y = [], []
    for i in range(cnt):
        x.append(data[i:i+n_step])
        y.append(data[i+n_step])

    return np.array(x), np.array(y)


def get_data(url, file_name):
    # 取得網頁連線，若不成功則回傳狀態並終止
    r = req.get(url)
    if r.status_code != 200:
        print(r.status_code)
        exit()
    # 抓取網頁原碼並分析
    content = r.text
    soup = BeautifulSoup(content, 'lxml')
    # 找出所有column名稱
    cols = [c.text for c in soup.find_all(attrs={'scope': 'col'})]
    # 找出table並計算有多少資料需要爬
    table = soup.find('table', attrs={'id': 'MyGridView'})
    n_tr = len(table.find_all('tr'))
    # 抓取前四列的所有資料
    df = pd.DataFrame(columns=cols[:4])
    for i in range(4):
        data = []
        for j in range(2, n_tr, 1):
            # 92油價id的例外處理
            target_id = f'MyGridView_ctl{j:02d}_Label_{cols[i]}' if i != 1 else f'MyGridView_ctl{j:02d}_{cols[i]}'
            cont = soup.find(attrs={'id': target_id}).text
            # 資料為空則轉成nan，以便等等一次清除
            data.append(cont if cont != '' else np.nan)
        df[cols[i]] = data
    # 清除空資料並存檔，然後回傳資料
    df = df.dropna()
    df.to_csv(f'./data/{file_name}', index=False)
    return df


def load_dataset(data_name):
    # 每次都更新最新資料，而舊資料則只在遺失時才重抓
    for f in list(oil_data.keys()):
        if f == 'latest.csv':
            get_data(oil_data[f], f)
        else:
            if not os.path.exists(f'./data/{f}'):
                get_data(oil_data[f], f)
    # 將所有資料合併後存檔，並且回傳資料
    dfs = []
    for f in list(oil_data.keys()):
        dfs.append(pd.read_csv(f'./data/{f}'))
    for i in range(1, len(dfs)):
        dfs[0] = pd.concat([dfs[0], dfs[i]], axis=0, ignore_index=True)
    dfs[0].columns = ['Date', 'Unleaded gasoline 92',
                      'Unleaded gasoline 95', 'Unleaded gasoline 98']
    dfs[0].to_csv('./data/data.csv', index=False)
    return dfs[0]


def main():
    # 讀取資料
    df_data = load_dataset('./data/data.csv')
    # 反轉資料，因為我需要將資料由舊到新排序
    df_data = df_data.iloc[::-1]
    # 畫出三種油的歷史油價曲線
    _plot([df_data.iloc[:, i].values for i in range(1, 4)], 'oil price',
          'price', list(df_data.columns[1:]))
    # 設定訓練資料與測試資料的比例
    # 0.1指的是10%的測試資料
    vaild_rate = 0.1
    vaild_split = round(len(df_data) * vaild_rate * -1)
    # 依序處理不同的油種
    types, prices, trends = [], [], []
    for i in range(1, 4):
        data = df_data.iloc[:, i].to_numpy()
        # 利用迴圈找出最好的模型
        best_score, best_model = 0, None
        for step in range(1, len(df_data)+vaild_split):
            # 將資料切成data, label
            x, y = split_data(data, step)
            # 開始分train data與test data
            x_train, y_train, x_test, y_test = x[:vaild_split,
                                                 :], y[:vaild_split], x[vaild_split:, :], y[vaild_split:]
            # 建模型並訓練
            lr = LinearRegression().fit(x_train, y_train)
            # 將測試資料丟進模型算分
            lr_test_score = lr.score(x_test, y_test)
            # 將最好的模型與相關資料留下
            if best_score < lr_test_score:
                best_score = lr_test_score
                best_step = step
                best_model = lr
                x_pred, y_actual = x_test, y_test
        # 輸出目前油種、最佳模型的step與測試分數，並把其測試資料的預測結果、實際結果和其差異存檔，然後印出
        print('\n----------------------------------------------------')
        print(f'"{df_data.columns[i]}"\t best step: {best_step}')
        print(f'test_score: {best_score}')
        y_preds = np.round(best_model.predict(x_pred), 1)
        df_res = pd.DataFrame(columns=['Prediction', 'Actual', 'diff'])
        df_res['Prediction'] = y_preds
        df_res['Actual'] = y_actual
        df_res['diff'] = np.round(y_preds - y_actual, 1)
        df_res.to_csv(
            f'./res/{df_data.columns[i]}_step {best_step}_score {best_score}.csv', index=False)
        print(f'{df_res}\n----------------------------------------------------')
        # 將預測結果與訓練資料合併，並與原始資料一同做圖比較
        predict_data = np.concatenate((data[:vaild_split], y_preds))
        _plot([predict_data, data],
              f'Compare {df_data.columns[i]}', 'price', ['predict', 'actual'])
        # 將原始資料以最佳模型的step取出最後一筆資料，再與測試資料合併成模型接受的shape
        pred = np.concatenate(
            [x_pred[1:, :], data[np.newaxis, best_step*-1:]], 0)
        # 預測下一次的油價，並紀錄
        pred_res = np.round(best_model.predict(pred), 1)
        types.append(df_data.columns[i])
        prices.append(pred_res[-1])
        trends.append(pred_res[-1]-y_actual[-1])

    # 將各種油價與漲幅存成csv，並印出
    df_pred = pd.DataFrame(columns=['type', 'price', 'trend'])
    df_pred['type'] = types
    df_pred['price'] = prices
    df_pred['trend'] = trends
    df_pred.to_csv('./res/predict.csv', index=False)
    print(df_pred)


def html(template, website):
    # 讀取事先寫好的html模板，並用utf8來解碼，因為我的網頁是寫中文的
    with open(template, 'r', encoding='utf-8') as f:
        content = f.read()
    # 讀取我的歷史資料，並且將其轉成分別以[日期, 價錢]的格式轉成json檔，放入新的html
    df = pd.read_csv('./data/data.csv')
    for i in range(3):
        df_tmp = df[['Date', f'Unleaded gasoline 9{2+i*3}']]
        df_tmp.columns = ['date', 'price']
        df_tmp.to_json(
            f'./data/Unleaded gasoline 9{2+i*3}.json', orient='records')
        with open(f'./data/Unleaded gasoline 9{2+i*3}.json', 'r') as jfile:
            tmp_js = json.loads(jfile.read())
            content = content.replace(
                '{\n'+f'              /* data_9{2+i*3} */\n            ' + '}', str(tmp_js))
    # 讀取我的預測資料，並放入新的html
    df = pd.read_csv('./res/predict.csv')
    df.to_json('./res/predict.json', orient='records')
    with open('./res/predict.json', 'r') as jfile:
        pred_js = json.loads(jfile.read())
        content = content.replace(
            '{\n              /* predictResult */\n            }', str(pred_js))
    # 將新的html存起來
    with open(website, 'w') as f:
        f.write(content)
    webbrowser.open(website, new=0, autoraise=True)


if __name__ == '__main__':
    # 檢查資料夾是否存在
    guard_dir('data')
    guard_dir('img')
    guard_dir('res')
    # 執行主程式
    main()
    # 輸出網頁
    html('template.html', 'index.html')
