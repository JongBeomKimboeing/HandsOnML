# 데이터 가져오기
'''
데이터를 다운로드하는 함수를 준비하면 특히 데이터가 정기적으로 바뀌는 경우에 유용하다.
데이터를 내려받는 일을 자동화하면 여러 기기에 데이터셋을 설치해야 할 때도 편리하다.

'''

import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/rickiepark/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing") # 경로를 병합하여 새 경로 생성 -> datasets/housing
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path): #os.path.isdir: 디렉토리 경로가 존재하는지 체크하기
        os.makedirs(housing_path) # 디렉토리 생성
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path) # 압축 풀기
    housing_tgz.close()

fetch_housing_data()
# 현재 작업 공간에 datasets/housing 디렉터리를 만들고
# housing.tgz 파일을 내려받고 같은 디렉터리에 압축을 풀어 housing.csv파일을 만든다.







# 1. 다운받은 데이터 확인하기

import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

print(housing.head()) # head(self,n=5) -> DataFrame 내의 처음 n줄의 데이터를 출력 (n=5가 기본)

print(housing.info()) # DataFrame 의 정보 출력
# 전체 행수 (20640), 각 특성의 데이터 타입과 널이 아닌 값의 개수를 확인하는 데 유용하다.

'''
[5 rows x 10 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20640 entries, 0 to 20639
Data columns (total 10 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   longitude           20640 non-null  float64
 1   latitude            20640 non-null  float64
 2   housing_median_age  20640 non-null  float64
 3   total_rooms         20640 non-null  float64
 4   total_bedrooms      20433 non-null  float64
 5   population          20640 non-null  float64
 6   households          20640 non-null  float64
 7   median_income       20640 non-null  float64
 8   median_house_value  20640 non-null  float64
 9   ocean_proximity     20640 non-null  object 
dtypes: float64(9), object(1)
memory usage: 1.6+ MB
None

'''

print('\n')
print(housing["ocean_proximity"].value_counts())
# value_counts() -> 유일한 값의 개수를 count 해준다. (범주형 데이터 출력)
'''
<1H OCEAN     9136
INLAND        6551
NEAR OCEAN    2658
NEAR BAY      2290
ISLAND           5
Name: ocean_proximity, dtype: int64
'''
print('\n')
print(housing.describe()) # dataframe의 통계 정보

# 백분위수는 전체 관측값에서 주어진 백분율이 속하는 하위 부분의 값을 나타낸다.
'''
ex)
25% 구역은 housing_median_age 가 18보다 작고, 50%는 29보다 작고, 75%는 37보다 작다.
이를 3분뒤수 라고 한다.
'''

'''
          longitude      latitude  ...  median_income  median_house_value
count  20640.000000  20640.000000  ...   20640.000000        20640.000000
mean    -119.569704     35.631861  ...       3.870671       206855.816909
std        2.003532      2.135952  ...       1.899822       115395.615874
min     -124.350000     32.540000  ...       0.499900        14999.000000
25%     -121.800000     33.930000  ...       2.563400       119600.000000
50%     -118.490000     34.260000  ...       3.534800       179700.000000
75%     -118.010000     37.710000  ...       4.743250       264725.000000
max     -114.310000     41.950000  ...      15.000100       500001.000000
'''


import matplotlib.pyplot as plt
#housing.hist(bins=50, figsize=(20,15)) # bins: 막대기의 개수를 지정해준다.  figsize: 그림 크기
#plt.savefig("attribute_histogram_plots")
#plt.show()

'''
attribute_histogram_plots를 보면

1. median_income이 US 달러로 표현돼 있지 않고 최댓값과 최솟값으로 스케일을 조정하여 만들었다.
-> 머신러닝에서 전처리된 데이터를 다루는 경우가 흔하고 이것이 문제가 되지는 않지만 
데이터가 어떻게 계산된 것인지 반드시 이해하고 있어야 한다.

2. housing_median_age 와 median_house_value 역시 최댓값과 최솟값으로 한정했다.
median_house_value는 레이블로 사용되기 때문에 심각한 문제가 될 수 있다.
즉, 가격이 한곗값을 넘어가지 않도록 머신러닝 알고리즘이 학습될수도 있다.
정확한 예측값이 필요한 경우 두가지 방법으로 해결할 수 있다.
a) 한곗값 밖의 구역에 대한 정확한 레이블을 구한다.
b) 훈련세트와 테스트셋에서 최댓값을 넘는 값들을 제거한다.

3. 특성들의 스케일이 서로 많이 다르다.

4. 많은 히스토그램의 꼬리가 두껍다. 즉, 가운데에서 왼쪽보다 오른쪽으로 더 멀리 뻗어있다.
이런 형태는 일부 머신러닝 알고리즘에서 패턴을 찾기 어렵게 만든다.
나중에 이런 특성들을 좀 더 종 모양의 분포가 되도록 변형 시키겠다.
'''






# 2. 테스트 세트 만들기

'''
만약, 테스트 세트를 미리 볼 경우, 겉으로 드러난 어떤 패턴에 속아 특정 머신러닝 모델을 선택할 수도 있다.
이 테스트 세트로 일반화 오차를 추정하면 매우 낙관적인 추정이 되며 시스템을 론칭했을 떄 기대한 성능이 
나오지 않을 것이다. 이를 데이터 스누핑(data snooping) 편향이라고 한다.

'''


import numpy as np

np.random.seed(42)

def split_train_test(data, test_ratio):
    shuffled_idices = np.random.permutation(len(data))
    # shuffle과 permutation의 차이는 shuffle은 데이터를 섞어서 직접 변경시키지만,
    # permutation 은 변경시키지 않는다.
    test_set_size = int(len(data)*test_ratio)
    test_indicies = shuffled_idices[:test_set_size] # 섞은
    train_indicies = shuffled_idices[test_set_size:]
    return data.iloc[train_indicies], data.iloc[test_indicies]
    # train_indicies, test_indicies 리스트에 해당하는 값을 정수 인덱싱한다.


train_set, test_set = split_train_test(housing, 0.2)

print(len(train_set))
print(len(test_set))

'''
문제점

1. 프로그램을 다시 실행하면 다른 테스트 세트가 생성된다. 여러 번 계속하면 우리는 전체 데이터셋을 보는 셈이므로 이렇게하면 안된다.

해결책
1) 처음 실행에서 테스트 세트를 저장하고 다음번 실행에서 이를 불러들인다.
2) 항상 같은 난수 인덱스가 생성되도록 np.random.permutation() 을 호출하기 전에 난수 발생기의 초깃값을 지정한다.
   (np.random.seed(42))
   
   

2. 업데이트된 데이터셋을 이용할 경우 테스트세트를 추가해야한다.

해결책
1) 데이터셋을 업데이트한 후에도 안정적인 훈련/테스트 분할을 위한 일반적인 해결책은
샘플의 식별자를 사용하여 테스트 세트로 보낼지 말지 정하는 것이다.

ex) 각 샘플마다 식별자의 해시값을 계산하여 해시 최댓값의 20% 보다 작거나 같은 샘플만 테스트 세트로 보낸다.

이렇게 하면 여러 번 반복 실행되면서 데이터셋이 갱신되더라도 테스트 세트가 동일하게 유지된다.
새로운 테스트 세트는 샘플의 20%를 갖게 되지만 이전에 훈련 세트에 있던 샘플은 포함시키지 않을 것이다.
'''


# 주택 데이터셋에는 식별자 칼럼이 없다. 대신 행의 인덱스를 ID로 사용하면 간단히 해결된다.

import hashlib

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# index를 기준으로 train_set, test_set 나누기
housing_with_id = housing.reset_index() # 'index' 열이 추가된 데이터 프레임이 반환된다.
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
print(train_set.head())
'''
   index  longitude  ...  median_house_value  ocean_proximity
0      0    -122.23  ...            452600.0         NEAR BAY
1      1    -122.22  ...            358500.0         NEAR BAY
2      2    -122.24  ...            352100.0         NEAR BAY
3      3    -122.25  ...            341300.0         NEAR BAY
6      6    -122.25  ...            299200.0         NEAR BAY

'''
# 행의 인덱스를 고유 식별자로 사용할 때 새 데이터는 데이터셋의 끝에 추가되어야 하고 어떤 행도 삭제되지 않아야한다.
# 이것이 불가능할 땐 고유 식별자를 만드는 데 안전한 특성을 사용해야 한다.
'''
ex)
구역의 위도와 경도는 몇백년 후 까지 안정적이라고 보장할 수 있으므로 두 값을 연결하여 다음과 같이 ID를 만들 수 있다.
'''

# id를 기준으로 train_set, test_set 나누기
housing_with_id["id"] = housing["longitude"] * 1000 + housing['latitude']
train_sets, test_sets = split_train_test_by_id(housing_with_id, 0.2, "id")

print(train_sets.head())


# 사이킷런은 데이터셋을 여러 서브셋으로 나누는 다양한 방법을 제공한다.
# 1) train_test_split
# 2) split_train_test
'''
train_test_split 은 split_train_test 과 비슷하지만, 두 가지 특성이 더 있다.

첫째, 앞서 설명한 난수 초깃값을 지정할 수 있는 random_state 매개변수가 있다.
둘째, 행의 개수가 같은 여러 개의 데이터셋을 넘겨서 같은 인덱스를 기반으로 나눌 수 있다.
(이는 예를 들어 데이테프레임이 레이블에 따라 여러 개로 나뉘어 있을 때 매우 유용하다.)
'''

from sklearn.model_selection import train_test_split

strain_set, stest_set = train_test_split(housing, test_size=0.2, random_state=42)
print(stest_set.head())
'''
[5 rows x 12 columns]
       longitude  latitude  ...  median_house_value  ocean_proximity
20046    -119.01     36.06  ...             47700.0           INLAND
3024     -119.46     35.14  ...             45800.0           INLAND
15663    -122.44     37.80  ...            500001.0         NEAR BAY
20484    -118.72     34.28  ...            218600.0        <1H OCEAN
9814     -121.93     36.62  ...            278000.0       NEAR OCEAN
'''

'''
지금까지는 무작위 샘플링 방식으로 데이터를 샘플링했다.
데이터셋이 충분히 크다면 일반적으로 괜찮지만 데이터셋이 적다면 샘플링 편향이 생길 가능성이 크다.

계층적 샘플링을 통해 샘플링 편향을 줄일 수 있다.
전체 모수는 계층이라는 동질의 그룹으로 나뉘고,
테스트 세트가 전체 모수를 대표하도록 각 계층에서 올바른 수의 샘플을 추출한다.
(계층 샘플링을 할 시에 계층별로 데이터셋에 충분한 샘플 수가 있어야 한다. 그렇지 않으면 계층의 중요도를 추정하는데 편향이 발생한다.)
(즉, 너무 많은 계층으로 나누면 안되고, 각 계층이 충분히 커야한다.)

결과적으로, 계층 샘플링을 사용해 만든 테스트 세트가 전체 데이터셋에 있는 소득 카테고리의 비율과 거의 같다.
반면, 무작위 샘플링으로 만든 테스트 세트는 비율이 많이 다르다.
'''

housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
# 카테고리 1은 0 ~ 1.5, 카테고리 2는 1.5 ~ 3 까지의 범위가 되는 식이다.

print(housing)
#housing["income_cat"].hist()
#plt.show()
# 결과를 보면, housing["median_income"] 과 비슷한 분포를 가지고 있다.

# housing["income_cat"] 를 기반으로 계층 샘플링을 해보자.
# 사이킷런의 StratifiedShuffleSplit 을 이용하여 계층 샘플링을 해보자.

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(housing, housing["income_cat"]): # .split(X,Y) -> 입력과 레이블이 매개변수이다.
    strat_train_set = housing.loc[train_idx]
    strat_test_set = housing.loc[test_idx]

# 아래 그래프를 보면 housing["income_cat"]과 동일한 분포를 가진 그래프를 볼 수 있다.
#strat_train_set["income_cat"].hist()
#plt.show()
#strat_test_set["income_cat"].hist()
#plt.show()


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "전체": income_cat_proportions(housing),
    "계층 샘플링": income_cat_proportions(strat_test_set),
    "무작위 샘플링": income_cat_proportions(test_set),
}).sort_index()
compare_props["무작위 샘플링 오류율"] = 100 * compare_props["무작위 샘플링"] / compare_props["전체"] - 100
compare_props["계층 샘플링 오류율"] = 100 * compare_props["계층 샘플링"] / compare_props["전체"] - 100

print(compare_props)
'''
결과: 계층 샘플링을 사용해 만든 테스트 세트가 전체 데이터셋에 있는 소득 카테고리의 비율과 거의 같다.
반면 일반 무작위 샘플링으로 만든 테스트 세트는 비율이 많이 달라졌다.



         전체    계층 샘플링   무작위 샘플링  무작위 샘플링 오류율  계층 샘플링 오류율
1  0.039826  0.039729  0.040213     0.973236   -0.243309
2  0.318847  0.318798  0.324370     1.732260   -0.015195
3  0.350581  0.350533  0.358527     2.266446   -0.013820
4  0.176308  0.176357  0.167393    -5.056334    0.027480
5  0.114438  0.114583  0.109496    -4.318374    0.127011
'''

# income_cat 특성을 삭제해서 데이터를 원래 상태로 돌린다.
# drop() -> axis=0이면 행을, axis=1이면 열을 삭제한다. inplace=True면 데이터 프레임 자체를 수정하고 아무런 값도 반환하지 않는다.

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat" , axis=1, inplace=True)





# 3. 데이터 이해를 위한 탐색과 시각화

# 훈련 세트에 대해 탐색해보자.
# 훈련 세트가 매우 크면 조작을 간단하고 빠르게 하기 위해 탐색을 위한 세트를 별도로 샘플링할 수 있다.
# (여기서는 훈련 세트가 작으므로 훈련세트 전체를 사용한다.)

# 훈련 세트를 손상시키지 않기 위해 복사본을 만들어 사용한다.
housing = strat_train_set.copy()


# 지리적 데이터 시각화
# 지리 정보(위도와 경도)가 있으니 모든 구역을 산점도로 만들어 데이터를 시각화하는 것은 좋은 생각이다.
# 아래 그래프는 캘리포니아 지역을 잘 나타내지만 어떤 특별한 패턴을 찾기는 힘들다.
'''
housing.plot(kind="scatter",x="longitude", y="latitude")
plt.savefig("bad_visualization_plot")
plt.show()
'''


# alpha=0.1 로 설정하면 데이터 포인트가 밀집된 영역을 잘 보여준다.
'''
housing.plot(kind="scatter",x="longitude", y="latitude", alpha=0.1)
plt.savefig("better_visualization_plot")
plt.show()

# s= 원의 반지름으로 구역의 인구를 나타낸다. c=색상은 가격, jet를 이용하여 colormap을 정의한다.
housing.plot(kind="scatter",x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100,
             label="population", figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()
plt.savefig("housing_prices_scatterplot")
plt.show()
'''


# 아래 코드는 캘리포니아 지도를 다운 받아 그래프를 그린 것이다
# 그냥 참고만...

'''
PROJECT_ROOT_DIR = os.path.join("datasets", "housing")
# Download the California image
images_path = os.path.join(PROJECT_ROOT_DIR, "images", "end_to_end_project")
os.makedirs(images_path, exist_ok=True)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
filename = "california.png"
print("Downloading", filename)
url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
urllib.request.urlretrieve(url, os.path.join(images_path, filename))

import matplotlib.image as mpimg
california_img=mpimg.imread(os.path.join(images_path, filename))
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="Population",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar(ticks=tick_values/prices.max())
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
plt.savefig("california_housing_prices_plot")
plt.show()
'''





# 상관관계 조사


#데이터셋이 너무 크지 않으므로 모든 특성 간의 표쥰 상관계수를 corr() 메서드를 이용해 쉽게 계산할 수 있다.

corr_matrix = housing.corr()
print('\n')
print(corr_matrix["median_house_value"].sort_values(ascending=False))
'''
median_house_value    1.000000
median_income         0.687160
total_rooms           0.135097
housing_median_age    0.114110
households            0.064506
total_bedrooms        0.047689
population           -0.026920
longitude            -0.047432
latitude             -0.142724
Name: median_house_value, dtype: float64
'''



# 특성 사이의 상관관계를 확인하는 다른 방법으로, 숫자형 특성 사이에 산점도를 그려주는 판다스의 scatter_matrix 함수를 사용한다.
# 여기서는 특성이 11 개여서 121개의 그래프가 되어 모두 보여줄 수 없어 상관관계가 높아보이는 특성 몇개만 알아본다.

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12,8))
#plt.savefig("scatter_matrix_plot")
#plt.show()
# 결과로 각 수치형 특성의 산점도와 각 수치형 특성의 히스토그램을 출력한다.


# 중간 주택 가격을 예측하는 데 가장 유용할 것 같은 특성은 중간 소득이므로 상관관계 산점도를 확대해보겠다

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
#plt.savefig("income_vs_house_value_scatterplot")
#plt.show()
'''
위 그래프에서 볼 수 있는 사실

1. 상관관계가 매우 강하다.

2. 앞서 본 가격 제한 값(500000 달러)에서 수평선으로 잘 보인다. 
이러한 직선에 가까운 형태는 45000달러 근처와 350000, 280000달러 에도 있고 그 아래에도 조금 더 있다.

알고리즘이 데이터에서 이런 이상한 형태를 학습하지 않도록 해당 구역을 제거하는 것이 좋다.
'''





# 4. 특성 조합으로 실험

'''
머신러닝 알고리즘용 데이터를 준비하기 전에 마지막으로 해볼 수 있는 것은 여러 특성의 조합을 시도해보는 것이다.

ex)
특정 구역의 방 개수는 얼마나 많은 가구 수가 있는지 모른다면 그다지 유용하지 않다.
진짜 필요한 것은 가구당 방 개수이다.
비슷하게 전체 침대 개수도 그 자체로는 유용하지 않다.
즉, 방 개수와 비교하는 게 낫다.
가구당 인원도 흥미로운 특성 조합일 것이다.
'''

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
'''
median_house_value          1.000000
median_income               0.687160
rooms_per_household         0.146285
total_rooms                 0.135097
housing_median_age          0.114110
households                  0.064506
total_bedrooms              0.047689
population_per_household   -0.021985
population                 -0.026920
longitude                  -0.047432
latitude                   -0.142724
bedrooms_per_room          -0.259984
Name: median_house_value, dtype: float64

Process finished with exit code 0

'''

'''
새로운 bedrooms_per_room 특성은 전체 방 개수나 침대 개수보다 중간 주책 가격과의 상관계가 훨씬 높다.
확실히 침대/방 개수도 구역 내 전체 방 개수보다 더 유용하다. 당연히 더 큰 집이 더 비싸다.
'''























