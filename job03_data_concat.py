import pandas as pd
import glob
import datetime

# last_data = []
# for i in range(6):
#     data_path = glob.glob('./crawling_data/data_{}*'.format(i))[-1]
#     last_data.append(data_path)
# print(last_data)

data_path = glob.glob("./crawling_data/*")
print(data_path)

df = pd.DataFrame()
for path in last_data:
    df_temp = pd.read_csv(path)
    df_temp.dropna(inplace = True)
    df = pd.concat([df, df_temp], axis = "rows", ignore_index = True)

print(df.head())
print(df["category"].value_counts())
print(df.info())
df.to_csv("./naver_news_titles_{}.csv".format(datetime.datetime.now().strftime("%Y%m%d")), index = False)