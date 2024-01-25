import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

df = pd.read_csv('./naver_news_titles_20240125.csv')
print(df.head())
df.info()

X = df['titles']
Y = df['category']

label_encoder = LabelEncoder()
labeled_y = label_encoder.fit_transform(Y)
print(labeled_y[:3])
label = label_encoder.classes_ # label을 무엇을 줬는지 확인 할 수 있다
print(label)
with open('./models/label_encoder.pickle', 'wb') as f: # pickle은 바이너리 상태로 그대로 저장 / 예를 들어 문자열을 저장하면 문자열로 숫자형이면 숫자형으로 가져온다
    pickle.dump(label_encoder, f) # f를 열어서 label_encoder를 dump(저장)한다
onehot_y = to_categorical(labeled_y)
print(onehot_y[:3])
print(X[1:5])
# 자연어 처리 시작
okt = Okt() # 형태소 분리를 해준다 / 형태소: 단어 하나하나 짜르는것을 의미?
temp = []
# for i in range(len(X)):
#     X[i] = okt.morphs(X[i])
# 분리하면 예를들어 ['문재인', '생일', '날', '엔', '산행'] 앞에와 같이 분리해준다 / stem=False: ['말씀드렸다']
# print(X[:5])

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True) # stem=True를 주면 다음과 같이 원형으로 바꿔준다: [말씀드리다]
    if i % 10000:
        print(i)
print(temp)
# print(X[0])
# 한글자짜리는 학습이 안된다 => [등장, 한] 이와같이 분리를 해주는데 '한'은 등장한으로 원래는 등장하다로 분류가 되어야하는데 분리가 완벽하게 되지 않기 때문이다
# 불용어: 예를들어 그녀는 엄마인지 누나인지 친구인지 누구인지 알 수가 없다 /이와같은 것을 불용어라고한다 => 감탄사, 접미사, 대명사 등이 있다
stopwords = pd.read_csv('./stopwords.csv', index_col=0)
for j in range(len(X)):
    words = []
    for i in range(len(X[j])):
        if len(X[j][i]) > 1: # 한글자짜리는 다 제외 하겠다
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])
    X[j] = ' '.join(words)
# print(X[:5])

token = Tokenizer()
token.fit_on_texts(X)
tokened_x = token.texts_to_sequences(X)
wordsize = len(token.word_index) + 1
# print(tokened_x)
print(wordsize)

with open('./models/news_token.pickle', 'wb') as f:
    pickle.dump(token, f)

max = 0
for i in range(len(tokened_x)):
    if max < len(tokened_x[i]):
        max = len(tokened_x[i])
print(max)

x_pad = pad_sequences(tokened_x, max)
print(x_pad)

X_train, X_test, Y_train, Y_test = train_test_split(x_pad, onehot_y, test_size=0.2)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test # xy는 형태가 튜플이다
xy = np.array(xy, dtype=object)
np.save('./news_data_max_{}_wordsize_{}'.format(max, wordsize), xy)
