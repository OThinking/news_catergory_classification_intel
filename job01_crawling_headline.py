import pandas as pd
from bs4 import BeautifulSoup # pip install bs4
import requests # pip install requests
import re # 파이썬 기본 패키지
import datetime

# category = ['Politics', 'Economic', 'social', 'Curture', 'World', 'IT']
# url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100'
# headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
# # 네이버가 트레픽관리를 한다고 크롤링을 막았을때 위와 같이 해더를 주면 해결 할 수 있다 / 헤더에다가 유저에이전트 정보를 담아서 줘야한다
# resp = requests.get(url, headers=headers) # requests: 클라이언트로 봐도 무방하다 / 서버에 요청
#
# print(resp)
# print(type(resp))
# # print(list(resp))
#
# soup = BeautifulSoup(resp.text, 'html.parser') # html형식으로 변경해줌
# # print(soup)
# title_tags = soup.select('.sh_text_headline')
# print(title_tags)
# print(len(title_tags))
# print(type(title_tags[0]))
# titles = []
# for title_tag in title_tags:
#     titles.append(re.compile('[^가-힣|a-z|A-Z]').sub(' ', title_tag.text))
#     # ^: 한글 영어 대소문자 말고 나머지 / 를 title_tag.text에서 빼고 띄어쓰기로 체우라는 의미
# print(titles)

category = ['Politics', 'Economic', 'social', 'Curture', 'World', 'IT']

df_titles = pd.DataFrame()
re_title = re.compile('[^가-힣|a-z|A-Z]')
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

for i in range(6):
    url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}'.format(i)
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    title_tags = soup.select('.sh_text_headline')
    titles = []
    for title_tag in title_tags:
        titles.append(re_title.sub(' ', title_tag.text))
    df_section_titles = pd.DataFrame(titles, columns=['titles'])
    df_section_titles['category'] = category[i]
    df_titles = pd.concat([df_titles, df_section_titles], axis='rows', ignore_index=True)
print(df_titles.head())
df_titles.info()
print(df_titles['category'].value_counts())
df_titles.to_csv('./crawling_data/naver_headline_news_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d')), index=False)
# datetime.now(): 현재시간이 나노세크로 되어있어 strftime을 사용하여 바꿔준다 / strftime: 문자열로 바꾸고 특정 포멧을 사용하여 년 월 일로 변경