from MLFirstToAll_1 import preprocessData
import pandas as pd
import numpy as np

strat_train_set = preprocessData.strat_train_set
strat_test_set = preprocessData.strat_test_set

'''
2.5 머신러닝 알고리즘을 위한 데이터를 준비

머신러닝 알고리즘을 위해 데이터를 준비할 차례이다.
디 작업을 수동으로 하는 대신 함수를 만들어 자동화해야 하는 이유가 있다

- 어떤 데이터셋에 대해서도 데이터 변환을 손쉽게 반복할 수 있다 (예를 들어 다음번에 새로운 데이터셋을 사용할 때)
- 향후 프로젝트에 사용할 수 있는 변환 라이브러리를 점진적으로 구축하게 된다.
- 실제 시스템에서 알고리즘에 새 데이터를 주입하기 전에 변환시키는 데 이 함수를 사용할 수 있다.
- 여러 가지 데이터 변환을 쉽게 시도해볼 수 있고 어떤 조합이 가장 좋은지 확인하는 데 편리하다.
'''

'''
먼저 원래 훈련 세트로 복원하고, 예측 변수와 타깃값에 같은 변형을 적용하지 않기 위해
예측 변수와 레이블을 분리한다. (drop()은 데이터 복사본을 만들며 strat_train_set에는 영향을 주지 않는다.)
'''

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

'''
2.5.1 데이터 정제

대부분의 머신러닝 알고리즘은 누락된 특성을 다루지 못하므로 이를 처리할 수 있는 함수를 몇개 만들겠다.
앞서 total_bedrooms 특성에 값이 없는 경우를 보았는데 이를 고쳐보겠다.

방법은 세 가지 입니다.

1. 해당 구역을 제거합니다.
2. 전체 특성을 삭제합니다.
3. 어떤 값으로 채웁니다.(0, 평균,중간값 등)

데이터프레임의 dropna(), drop(), fillna() 메서도를 이용해 이런 작업을 간단하게 처리할 수 있다.
'''

# 1. 특성 값이 없는 부분 확인하기

sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
print(sample_incomplete_rows)
'''
	longitude	latitude	housing_median_age	total_rooms	total_bedrooms	population	households	median_income	ocean_proximity
4629	-118.30	34.07	18.0	3759.0	NaN	3296.0	1462.0	2.2708	<1H OCEAN
6068	-117.86	34.01	16.0	4632.0	NaN	3038.0	727.0	5.1762	<1H OCEAN
17923	-121.97	37.35	30.0	1955.0	NaN	999.0	386.0	4.6328	<1H OCEAN
13656	-117.30	34.05	6.0	    2155.0	NaN	1039.0	391.0	1.6675	INLAND
19252	-122.79	38.48	7.0	    6837.0	NaN	3468.0	1405.0	3.1662	<1H OCEAN
'''

# 2. 특성 값이 없는 부분 수정하기

# 옵션1 해당 구역을 제거
#sample_incomplete_rows.dropna(subset=["total_bedrooms"])
#print(sample_incomplete_rows)

# 옵션2 전체 특성을 삭제
#sample_incomplete_rows.drop("total_bedrooms", axis=1)
#print(sample_incomplete_rows)

# 옵션3
# -> 훈련 세트에서 중간값을 계산하고 누락된 값을 이 값으로 채워 넣는다.
# 반드시 계산한 중간값을 저장해야한다.
# 나중에 시스템을 평가할 때 테스트 세트에 있는 누락된 값과 시스템이 실제 운영될 때 새로운 데이터에 있는 누락된 값을 채워넣는데
# 필요하기 때문이다.
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True)
print(sample_incomplete_rows["total_bedrooms"])
'''
	longitude	latitude	housing_median_age	total_rooms	total_bedrooms	population	households	median_income	ocean_proximity
4629	-118.30	34.07	18.0	3759.0	433.0	3296.0	1462.0	2.2708	<1H OCEAN
6068	-117.86	34.01	16.0	4632.0	433.0	3038.0	727.0	5.1762	<1H OCEAN
17923	-121.97	37.35	30.0	1955.0	433.0	999.0	386.0	4.6328	<1H OCEAN
13656	-117.30	34.05	6.0	  2155.0	433.0	1039.0	391.0	1.6675	INLAND
19252	-122.79	38.48	7.0	  6837.0	433.0	3468.0	1405.0	3.1662	<1H OCEAN
'''


# 사이킷런의 SimpleImputer는 누락된 값을 손쉽게 다루도록 해준다.

# 먼저 누락된 값을 특성의 중간값으로 대체한다고 지정하여 SimpleImputer의 객체를 생성한다.
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

# 중간값이 수치형 특성에서만 계산될 수 있기 때문에 텍스트 특성인 ocean_proximity 를 제외한 데이터 복사본을 생성한다.
housing_num = housing.drop("ocean_proximity", axis=1)
# imputer 객체의 fit() 메서드를 사용해 훈련 데이터에 적용할 수 있다.
imputer.fit(housing_num)

'''
imputer는 각 특성의 중간값을 계산해서 그 결과를 객체의 statistics_ 속성에 저장한다.

"total_bedrooms" 특성에만 누락된 값이 있지만
나중에 시스템이 서비스될 때 새로운 데이터에서 어떤 값이 누락될 지 확신할 수 없으므로
모든 수치형 측성에 imputer를 적용하는 것이 바람직하다.
'''

print(imputer.statistics_)
'''
Name: total_bedrooms, dtype: float64
[-118.51     34.26     29.     2119.5     433.     1164.      408.
    3.5409]
'''

print(housing_num.median().values)
'''
[-118.51     34.26     29.     2119.5     433.     1164.      408.
    3.5409]
'''

# 이제 학습된 imputer 객체를 사용해 훈련 세트에서 누락된 값을 학습한 중간값으로 바꿀 수 있다.
X = imputer.transform(housing_num)
print(X)
'''
[[-121.89     37.29     38.     ...  710.      339.        2.7042]
 [-121.93     37.05     14.     ...  306.      113.        6.4214]
 [-117.2      32.77     31.     ...  936.      462.        2.8621]
 ...
 [-116.4      34.09      9.     ... 2098.      765.        3.2723]
 [-118.01     33.82     31.     ... 1356.      356.        4.0625]
 [-122.45     37.77     52.     ... 1269.      639.        3.575 ]]

'''
# 위 결과는 평범한 넘파이 배열이다. 이를 다시 판다스 데이터프레임으로 간단히 되돌릴 수 있다.

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
print(housing_tr)



'''
2.5.2 텍스트와 범주형 특성 다루기
지금까지 수치형 특성만 다루었다. 이제 텍스트 특성을 살펴보자.
'''

# 1. 텍스트 범주형 데이터 확인
# 아래 각 값은 카테고리를 나타낸다. 따라서 이 특성은 범주형 특성이다.

housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(10))
'''
[16512 rows x 8 columns]
      ocean_proximity
17606       <1H OCEAN
18632       <1H OCEAN
14650      NEAR OCEAN
3230           INLAND
3555        <1H OCEAN
19480          INLAND
8879        <1H OCEAN
13685          INLAND
4937        <1H OCEAN
4861        <1H OCEAN
'''

# 2. 카테고리를 텍스트에서 숫자로 변환
# 대부분의 머신러닝 알고리즘은 숫자를 다루므로 이 카테고리를 텍스트에서 숫자로 변환한다.
# 이를 위해 사이킷런의 OrdinalEncoder 클래스를 사용한다.

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])
'''
[[0.]
 [0.]
 [4.]
 [1.]
 [0.]
 [1.]
 [0.]
 [1.]
 [0.]
 [0.]]
'''

# categories_ 인스턴스 변수를 사용해 카테고리 목록을 얻을 수 있다.
print(ordinal_encoder.categories_)
'''
[array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
      dtype=object)]
'''

# 위 표현 방식의 문제는 머신러닝 알고리즘이 가까디 있는 두 값이 떨어져 있는 두 값보다 더 비슷하다고 생각한다는 점입니다.
'''
일부 경우에는 괜찮습니다. (예를들어 'bad', 'average', 'good', 'excellent')

하지만 이는 ocean_proximity 열에 해당되지 않습니다.

이 문제는 일반적으로 카테고리별 이진 특성을 만들어 해결한다.
한 특성만 1이고 (핫) 나머지는 0이므로 이를 원-핫 인코딩 이라고 부른다. 이따금 새로운 특성을 더미 특성이라고도 부른다.

사이킷런은 범주의 값을 원-핫 벡터로 바꾸기 위한 OneHotEncoder 클래스를 제공한다.
'''

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# 출력을 보면 넘파이 배열이 아니고 사이파이 희소행렬(sparse matrix) 입니다.
# 원 핫 인코딩에서 0을 모두 메모리에 저장하는 것은 낭비이므로 희소 행렬은 0이 아닌 원소의 위치만 저장한다.
print(housing_cat_1hot)
'''
  (0, 0)	1.0
  (1, 0)	1.0
  (2, 4)	1.0
  (3, 1)	1.0
  (4, 0)	1.0
  (5, 1)	1.0
  (6, 0)	1.0
  (7, 1)	1.0
  (8, 0)	1.0
  (9, 0)	1.0
  (10, 1)	1.0
  (11, 1)	1.0
  (12, 0)	1.0
  .
  .
  .
'''
# 행렬을 넘파이 배열로 바꾸려면 toarray() 메서드를 호출하면 된다.
print(housing_cat_1hot.toarray())
'''
[[1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1.]
 ...
 [0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0.]]

'''

# 아래 코드를 이용해 인코더의 카테고리 리스트를 얻을 수 있다.
print(cat_encoder.categories_)
'''
[array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
      dtype=object)]
'''

# OneHotEncoder를 만들 때 sparse=False로 지정할 수 있습니다:
cat_encoder = OneHotEncoder(sparse= False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)
'''
[[1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1.]
 ...
 [0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0.]]
'''


'''
2.5.3 나만의 변환기

사이킷런이 유용한 변환기를 많이 제공하지만
특별한 정제 작업이나 어떤 특성들을 조합하는 등의 작업을 위해 자신만의 변환기를 만들어야 할 때가 있다.
 
내가 만든 변환기를 사이킷런의 기능과 매끄럽게 연동하고 싶을 것이다.
사이킷런은 (상속이 아닌) 덕 타이핑을 지원하므로
 fit(), transfrom(), fir_transform() 메서드를 구현한 파이썬 클래스를 만들면 된다. 
'''

# 다음은 앞서 이야기한 조합 특성을 추가하는 간단한 변환기이다.

'''
아래의 경우 변환기가 add_bedrooms_per_room 하이퍼파라미터 하나를 가지며 기본값을 True로 지정한다.
이 특성을 추가하는 것이 머신러닝 알고리즘에 도움이 될지 안 될지 이 하이퍼파라미터로 쉽게 확인해볼 수 있다.

일반적으로 100% 확신이 없는 모든 데이터 준비 단계에 대해 하이퍼 파라미터를 추가할 수 있다.
이런 데이터 준비 단계를 자동화할수록 더 많은 조합을 자동으로 시도해볼 수 있고 최상의 조합을 찾을 가능성을 매우 높여준다.
(그리고 시간도 많이 절약된다.)
'''

from sklearn.base import BaseEstimator, TransformerMixin

# 열 인덱스
rooms_ix,  bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator , TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    # 사이킷런의 기능과 매끄럽게 연동하기 위해 fit 과 transform을 사용한다.
    def fit(self, X, y=None):
        return self  # 아무것도 하지 않습니다


    def transform(self,X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        # np.c_: 두 개의 1차원 배열을 칼럼으로 세로로 붙여서 2차원 배열 만들기 (열을 추가하는 방식으로 붙이기)
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.to_numpy())


# 데이터 프래임에 추가

housing_extra_attribs = pd.DataFrame(housing_extra_attribs,
                                     columns=list(housing.columns) + ["rooms_per_household", "population_per_household"],
                                     index= housing.index)
print(housing_extra_attribs.head())
'''
      longitude latitude  ... rooms_per_household population_per_household
17606   -121.89    37.29  ...             4.62537                   2.0944
18632   -121.93    37.05  ...             6.00885                  2.70796
14650    -117.2    32.77  ...             4.22511                  2.02597
3230    -119.61    36.31  ...             5.23229                  4.13598
3555    -118.59    34.23  ...             4.50581                  3.04785
'''



'''
2.5.4 특성 스케일링

데이터에 적용할 가장 중요한 변환 중 하나가 특성 스케일링(feature scaling)이다.
트리 기반 알고리즘을 제외한 머신러닝 알고리즘은 입력 숫자 특성의 스케일이 많이 다르면 잘 동작하지 않는다.
(단, 타깃값에 대한 스케일링은 일반적으로 불필요하다.)

모든 특성의 범위를 같도록 만들어주는 방법으로 min-max 스케일링과 표준화가 널리 사용된다.

1) min-max 스케일링 (정규화)
0 ~ 1 범위에 들도록 값을 이동하고 스케일을 조정한다.

-> 데이터에서 최솟값을 뺀 후 최댓값과 최솟값의 차이로 나누면 이렇게 할 수 있다.
사이킷런에는 이에 해당하는 MinMaxScaler 변환기를 제공한다.
0 ~ 1 사이를 원하지 않는다면 feature_range 매개변수로 범위를 변경할 수 있다.


2) 표준화

-> 먼저 평균을 뺀 후(표준화를 하면 항상 평균이 0이 된다.) 표준편차로 나누어 결과 분포의 분산이 1이 되도록 한다.

min-max 스케일링과는 달리 표준화는 범위의 상한과 하한이 없어 어떤 알고리즘에서는 문제가 될 수 있다.
(예를 들어 신경망은 종종 입력값의 범위로 0에서 1사이를 기대한다.)

그러나, 표준화는 이상치에 영향을 덜 받는다.
예를 들어 중간소득을 (잘목해서) 100이라 입력한 구역을 가정해보자.
min-max 스케일링은 0 ~ 15 사이의 모든 다른 값을 0 ~ 0.15로 만들어버리겠지만,
표준화는 크게 영향을 받지 않는다.

사이킷런에는 표준화를 위한 StandardScaler 변환기가 있다.



주의)
모든 변환기에서 스케일링은 (테스트 세트가 포함된) 전체 데이터가 아니고 훈련 데이터에 대해서만 fit() 메서드를 적용해야한다.
그런 다음 훈련 세트와 테스트 세트(그리고 새로운 데이터)에 대해 transform() 메서드를 사용한다.
'''

'''
2.5.5 변환 파이프라인

앞서 보았듯이 변환 단계가 많으며 정확한 순서대로 실행되어야한다.
다행히 사이킷런에는 연속된 변환을 순서대로 처리할 수 있도록 도와주는 Pipeline 클래스가 있다.
'''

# 다음은 숫자 특성을 처리하는 간단한 파이프라인이다.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")), # NAN값 처리
    ('attribs_adder', CombinedAttributesAdder()), # 특성 조합
    ('std_scaler', StandardScaler()) # 표준화
])

housing_num_tr = num_pipeline.fit_transform(housing_num)
print(housing_tr)

'''
       longitude  latitude  ...  households  median_income
17606    -121.89     37.29  ...       339.0         2.7042
18632    -121.93     37.05  ...       113.0         6.4214
14650    -117.20     32.77  ...       462.0         2.8621
3230     -119.61     36.31  ...       353.0         1.8839
3555     -118.59     34.23  ...      1463.0         3.0347
...          ...       ...  ...         ...            ...
6563     -118.13     34.20  ...       210.0         4.9312
12053    -117.56     33.88  ...       258.0         2.0682
13908    -116.40     34.09  ...       765.0         3.2723
11159    -118.01     33.82  ...       356.0         4.0625
15775    -122.45     37.77  ...       639.0         3.5750
'''

'''
Pipeline은 연속된 단계를 나타내는 이름/추정기 쌍의 목록을 입력으로 받는다.

마지막 단계에는 변환기와 추정기를 모두 사용할 수 있고 그 외에는 모두 변환기여야한다.
(즉, fit_transform() 메서드를 가지고 있어야 한다.)

파이프라인의 fit() 메서드를 호출하면 모든 변환기의 fit_transform() 메서드를 순서대로 호출하면서
한 단계의 출력을 다음 단계의 입력으로 전달한다.
마지박 단계에서는 fit() 메서드만 호출한다.
'''


'''
지금까지 범주형 열과 수치형 열을 각각 다루었다.
하나의 변환기로 각 열마다 적절한 변환을 적용하여 모든 열을 처리할 수 있다면 더 편리할 것이다.
-> 사이킷런 0.20버전에서 이런 기능을 위해 ColumnTransformer가 추가되었다.
'''

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num) # 수치형 열 이름의 리스트
cat_attribs = ["ocean_proximity"] # 범주형 열 이름의 리스트를 만든다.

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs), #(이름, 변환기, 변환기가 적용될 열 이름(또는 인덱스))
    # num_pipeline 을 이용해 변환
    ("cat", OneHotEncoder(), cat_attribs)
    # OneHotEncoder 을 이용해 변환
])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)

'''
[[-1.15604281  0.77194962  0.74333089 ...  0.          0.
   0.        ]
 [-1.17602483  0.6596948  -1.1653172  ...  0.          0.
   0.        ]
 [ 1.18684903 -1.34218285  0.18664186 ...  0.          0.
   1.        ]
 ...
 [ 1.58648943 -0.72478134 -1.56295222 ...  0.          0.
   0.        ]
 [ 0.78221312 -0.85106801  0.18664186 ...  0.          0.
   0.        ]
 [-1.43579109  0.99645926  1.85670895 ...  0.          1.
   0.        ]]
'''



'''
2.6 모델 선택과 훈련

2.6.1 훈련 세트에서 훈련하고 평가하기
이전 단계의 작업 덕분에 생각보다 훨씬 간단해졌다.
선형회귀 모델을 사용해서 모델을 훈련시켜보겠다.
'''

from sklearn.linear_model import LinearRegression
print('\n')
print("housing_prepared")
print(housing_prepared)
'''
[[-1.15604281  0.77194962  0.74333089 ...  0.          0.
   0.        ]
 [-1.17602483  0.6596948  -1.1653172  ...  0.          0.
   0.        ]
 [ 1.18684903 -1.34218285  0.18664186 ...  0.          0.
   1.        ]
 ...
 [ 1.58648943 -0.72478134 -1.56295222 ...  0.          0.
   0.        ]
 [ 0.78221312 -0.85106801  0.18664186 ...  0.          0.
   0.        ]
 [-1.43579109  0.99645926  1.85670895 ...  0.          1.
   0.        ]]
'''
print('\n')
print("housing_labels")
print(housing_labels)
'''
17606    286600.0
18632    340600.0
14650    196900.0
3230      46300.0
3555     254500.0
           ...   
6563     240200.0
12053    113000.0
13908     97800.0
11159    225900.0
15775    500001.0
Name: median_house_value, Length: 16512, dtype: float64

Process finished with exit code 0

'''

# LinearRegression 모델 훈련
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# 훈련 세트에 있는 몇 개 샘플에 대해 적용
some_data = housing.iloc[:5]
some_labels = housing_labels[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("예측:", lin_reg.predict(some_data_prepared))
print("레이블:", list(some_labels))

# 아주 정확한 예측은 아니지만(예를들어 첫 번째 예측은 40% 가까이 벗어났다.) 작동은 한다.

# 사이킷런의 mean_square_error 함수를 사용해 전체 훈련 세트에 대한 이 회귀 모델의 RMSE를 측정해본다.

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("RMSE:",lin_rmse) #RMSE: 68628.19819848923

'''
위 결과는 만족스럽지 못 하다.
위는 모델이 훈련 데이터에 과소적합된 사례이다.

해결 방법
1) 더 강력한 모델을 선택
2) 훈련 알고리즘에 더 좋은 특성을 주입하거나 모델의 규제를 감소
'''

# 더 강력한 모델 선택하기
# -> DecisionTreeREgressor를 훈련시켜본다.

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

print("예측:", tree_reg.predict(some_data_prepared))
print("레이블:", list(some_labels))
'''
예측: [286600. 340600. 196900.  46300. 254500.]
레이블: [286600.0, 340600.0, 196900.0, 46300.0, 254500.0]
'''
housing_predictions = tree_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("RMSE:",lin_rmse) # RMSE: 0.0

'''
위 모델은 데이터에 너무 심하게 과대적합된 것으로 보인다.
우리가 확신이 드는 모델이 론칭할 준비가 되기 전까지 테스트 세트는 사용하지 않으려 하므로
훈련 세트의 일부분으로 훈련을 하고 다른 일부분은 모델 검증에 사용해야한다.
'''

'''
2.6.2 교차 검증을 사용한 평가

1) 훈련 세트를 더 작은 훈련 세트와 검증 세트로 나누고,
더 작은 훈련 세트에서 모델을 훈련 시키고 검증 세트로 모델을 평가하는 방법

2) 사이킷런의 k-결 교차 검증 기능을 사용하는 방법
다음 코드는 훈련 세트를 폴드(fold)라 불리는 10개의 서브셋으로 무작위로 분할합니다.
그런 다음 결정 트리 모델을 10번 훈련하고 평가하는데, 매번 다른 폴드를 선택해 평가에 사용하고
나머지 9개 폴드는 훈련에 사용합니다. 10개의 평가 점수가 담긴 배열이 결과가 됩니다.

'''

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error",
                         cv=10)

tree_rmse_scores = np.sqrt(-scores) # 교차 검증 기능은 효용함수를 쓰기 때문에 음숫 값을 계산한다.
# 우리는 비용함수를 계산해야하므로, -를 붙여준다.

def display_scores(scores):
    print("점수:", scores)
    print("평균:", scores.mean())
    print("표준 편차:", scores.std())

display_scores(tree_rmse_scores)
'''
점수: [69327.01708558 65486.39211857 71358.25563341 69091.37509104
 70570.20267046 75529.94622521 69895.20650652 70660.14247357
 75843.74719231 68905.17669382]
평균: 70666.74616904806
표준 편차: 2928.322738055112
'''

lin_score = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error",
                         cv=10)

lin_rmse_scores = np.sqrt(-lin_score)
display_scores(lin_rmse_scores)
'''
점수: [66782.73843989 66960.118071   70347.95244419 74739.57052552
 68031.13388938 71193.84183426 64969.63056405 68281.61137997
 71552.91566558 67665.10082067]
평균: 69052.46136345083
표준 편차: 2731.674001798349
'''

'''
위 두 결과를 비교해보면, 선형 회귀 모델의 성능이 더 잘 나온다는 것을 볼 수 있다.
(점수가 오차를 의미하므로, 작을 수록 좋다.)

확실히 결정 트리 모델이 과대적합되어 선형 회귀 모델보다 성능이 나쁘다.

마지막으로 RandomForestRegressor 모델을 하나 더 시도해보자.

RandomForestRegressor 모델은 특성을 무작위로 선택해서 많은 결정 트리를 만들고 그 예측을 평균 내는 방식이다.
여러 다른 모델을 모아서 하나의 모델을 만드는 것을 앙상블 학습이라고 하며 머신러닝 알고리즘의 성능을 극대화하는 방법 중 하나이다.
'''
'''
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared,housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print("RMSE:",forest_rmse)
# RMSE: 18603.515021376355


forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
'''
'''
점수: [49519.80364233 47461.9115823  50029.02762854 52325.28068953
 49308.39426421 53446.37892622 48634.8036574  47585.73832311
 53490.10699751 50021.5852922 ]
평균: 50182.303100336096
표준 편차: 2097.0810550985693
'''

'''
랜덤 포레스트는 더 좋은 결과를 냈다.
하지만, 훈련 세트에 대한 점수가 검증 세트에 대한 점수보다 훨씬 낮으므로 이 모델도 여전히 훈련 세트에 과대적합되어 있다.

과대적합 해결 방법
1) 모델을 간단히 한다.
2) 규제를 이용해 제한 한다.
3) 더 많은 훈련 데이터를 모은다.


cf) 
실험한 모델을 모두 저장해두면 필요할 때 쉽게 모델을 복원할 수 있따.
교차 검증 점수와 실제 예측값은 물론 하이퍼파라미터와 훈련된 모델 파라미터 모두 저장해야한다.
이렇게 하면 여러 모델의 점수와 모델이 만든 오차를 쉽게 비교할 수 있다.
팡썬의 pickle 패키지나 큰 넘파이 배열을 저장하는 데 아주 효율적인 joblib를 사용하여 사이킷런 모델을 간단하게 저장할 수 있다.


import joblib

joblib.dump(my_model, "my_model.pkl")
# 그리고 나중에..

my_model_loaded = joblib.load("my_model.pkl")
'''

#cf)
'''
from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels,housing_predictions)
svm_rmse = np.sqrt(svm_mse)
print(svm_rmse) # 111094.6308539982
'''


'''
2.7 모델 세부 튜닝

가능성 있는 모델들을 추렸다고 가정하자.
이제 이 모델들을 세부 튜닝 해야한다.

2.7.1 그리드 탐색
가장 단순한 방법은 만족할 만한 하이퍼파라이터 조합을 찾을 때까지 수동으로 하이퍼파라미터를 조정하는 것이다.
이는 매우 지루한 작업이고 많은 경우의 수를 탐색하기에는 시간이 부족할 수도 있다.

대신 사이킷런의 GridSearchCV를 사용하는 것이 좋다.
탐색하고자 하는 하이퍼파라미터와 시도해볼 값을 지정하기만 하면 되기 때문이다.
그러면 가능한 모든 하이퍼파라미터 조합에 대해 교차 검증을 사용해 평가하게 된다.

'''
# 예를 들어 다음 코드는 RandomForestRegressor에 대한 최적의 하이퍼파라미터 조합을 탐색한다.

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = [
    # 12(=3×4)개의 하이퍼파라미터 조합을 시도합니다.
    {'n_estimators':[3, 10, 30], 'max_features':[2, 4, 6, 8]},
    # bootstrap은 False로 하고 6(=2×3)개의 조합을 시도합니다.
    {'bootstrap':[False], 'n_estimators': [3, 10], 'max_features':[2, 3, 4]},
]
forest_reg = RandomForestRegressor(random_state=42)
# 다섯 개의 폴드로 훈련하면 총 (12+6)*5=90번의 훈련이 일어납니다.
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

# 이는 시간이 꽤 오래 걸리지만 다음과 같이 최적의 조합을 얻을 수 있습니다.
print(grid_search.best_params_)
# {'max_features': 8, 'n_estimators': 30}
# cf) 8과 30은 탐색 범위의 최댓값이기 때문에 계속 접수가 향상될 가능성이 있으므로 더 큰 값으로 다시 검색해야 한다.


# 최적의 추정기에 직접 접근할 수도 있다.
print(grid_search.best_estimator_)
'''
RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=None, max_features=8, max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=30, n_jobs=None, oob_score=False,
                      random_state=42, verbose=0, warm_start=False)

'''

# 평가 점수도 확인할 수 있다.
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
'''
63669.11631261028 {'max_features': 2, 'n_estimators': 3}
55627.099719926795 {'max_features': 2, 'n_estimators': 10}
53384.57275149205 {'max_features': 2, 'n_estimators': 30}
60965.950449450494 {'max_features': 4, 'n_estimators': 3}
52741.04704299915 {'max_features': 4, 'n_estimators': 10}
50377.40461678399 {'max_features': 4, 'n_estimators': 30}
58663.93866579625 {'max_features': 6, 'n_estimators': 3}
52006.19873526564 {'max_features': 6, 'n_estimators': 10}
50146.51167415009 {'max_features': 6, 'n_estimators': 30}
57869.25276169646 {'max_features': 8, 'n_estimators': 3}
51711.127883959234 {'max_features': 8, 'n_estimators': 10}
49682.273345071546 {'max_features': 8, 'n_estimators': 30}
62895.06951262424 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
54658.176157539405 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
59470.40652318466 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
52724.9822587892 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
57490.5691951261 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
51009.495668875716 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}


위를 보면 'max_features'가 8, 'n_estimators'가 30일 때 최적의 솔루션임을 알 수 있다.
'''

print(pd.DataFrame(grid_search.cv_results_))
'''
   mean_fit_time  std_fit_time  ...  mean_train_score  std_train_score
0        0.063638      0.006973  ...     -1.105559e+09     2.220402e+07
1        0.181326      0.020414  ...     -5.818785e+08     7.345821e+06
2        0.535647      0.017979  ...     -4.394734e+08     2.966320e+06
3        0.083975      0.004107  ...     -9.848396e+08     4.084607e+07
4        0.258714      0.008163  ...     -5.163863e+08     1.542862e+07
5        0.775326      0.026159  ...     -3.879289e+08     8.571233e+06
6        0.099540      0.002545  ...     -9.023976e+08     2.591445e+07
7        0.328148      0.004699  ...     -5.013349e+08     3.100456e+06
8        0.997151      0.007763  ...     -3.841296e+08     3.617057e+06
9        0.143626      0.003824  ...     -8.883545e+08     2.750227e+07
10       0.521601      0.038017  ...     -4.923911e+08     1.459294e+07
11       1.456504      0.148108  ...     -3.810330e+08     4.871017e+06
12       0.100734      0.034631  ...      0.000000e+00     0.000000e+00
13       0.309040      0.031874  ...     -6.056027e-01     1.181156e+00
14       0.121657      0.036980  ...     -1.214568e+01     2.429136e+01
15       0.380541      0.049401  ...     -5.272080e+00     8.093117e+00
16       0.142728      0.026744  ...      0.000000e+00     0.000000e+00
17       0.471774      0.029633  ...     -3.028238e-03     6.056477e-03

[18 rows x 23 columns]
'''


'''
2.7.2 랜덤 탐색

그리드 탐색 방법은 이전 예제와 같이 비교적 적은 수의 조합을 탐구할 때 괜찮다.
하지만 하이퍼파라미터 탐색 공간이 커지면 RandomizedSearchCV를 사용하는 편이 더 좋다.

RandomizedSearchCV 는 GridSearchCV 와 거의 같은 방식으로 사용하지만 가능한 모든 조합을 시도하는 대신
각 반복마다 하이퍼파라미터에 임의의 수를 대입하여 지정한 횟수만큼 평가한다.
이 방식의 주요 장점은 두 가지이다.

1) 랜덤 탐색을 1000회 반복하도록 실행하면 하이퍼파라미터마다 각기 다른 1000개의 값을 탐색한다.
(그리드 탐색에서는 하이퍼파라미터마다 몇 개의 값만 탐색한다.)

2) 단순히 반복 횟수를 조절하는 것만으로 하이퍼파라미터 탐색에 투입할 컴퓨팅 자원을 제어할 수 있다.



2.7.3 앙상블 방법
여러 모델의 결과 값을 모아 prediction 하는 방식으로
최상의 단일 모델보다 더 나은 성능을 발휘할 때가 많다.
특히, 개개의 모델이 각기 다른 형태의 오차를 만들 때 그렇다.
'''


'''
2.7.4 최상의 모델과 오차분석

최상의 모델을 분석하면 문제에 대한 좋은 통찰을 얻는 경우가 많다.
예를 들어 RandomForestRegressor가 정확한 예측을 만들기 위한 각 특성의 상대적인 중요도를 알려준다.
'''
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
'''
[7.33442355e-02 6.29090705e-02 4.11437985e-02 1.46726854e-02
 1.41064835e-02 1.48742809e-02 1.42575993e-02 3.66158981e-01
 5.64191792e-02 1.08792957e-01 5.33510773e-02 1.03114883e-02
 1.64780994e-01 6.02803867e-05 1.96041560e-03 2.85647464e-03]
'''

# 중요도 다음에 그에 대응하는 특성 이름을 표시해본다.

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs # 모든 항목들을 연결
print(sorted(zip(feature_importances, attributes), reverse=True))

'''
[(0.36615898061813423, 'median_income'),
(0.16478099356159054, 'INLAND'),
(0.10879295677551575, 'pop_per_hhold'),
(0.07334423551601243, 'longitude'),
(0.06290907048262032, 'latitude'),
(0.056419179181954014, 'rooms_per_hhold'),
(0.053351077347675815, 'bedrooms_per_room'),
(0.04114379847872964, 'housing_median_age'),
(0.014874280890402769, 'population'),
(0.014672685420543239, 'total_rooms'), 
(0.014257599323407808, 'households'), 
(0.014106483453584104, 'total_bedrooms'), 
(0.010311488326303788, '<1H OCEAN'), 
(0.0028564746373201584, 'NEAR OCEAN'), 
(0.0019604155994780706, 'NEAR BAY'), 
(6.0280386727366e-05, 'ISLAND')]

위 정보를 바탕으로 덜 중요한 특성들을 제외할 수 있다.

시스템이 특정한 오차를 만들었다면 왜 그런 문제가 생겼는지 이해하고 문제를 해결하는 방법이 무엇인지 찾아야한다.
(추가 특성을 포함시키거나, 불필요한 특성을 제거하거나, 이상치를 제외하는 등)
'''


'''
2.7.5 테스트 세트로 시스템 평가하기

테스트 세트에서 예측 변수와 레이블을 얻은 후 full_pipeline을 사용해 데이터를 변환하고
(테스트 세트에서 훈련하면 안 되므로 fit_transform()이 아니라 transform()을 호출해야한다.)
테스트 세트에서 최종 모델을 평가한다.
'''

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1) # 레이블 제거
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print(final_rmse) # 47730.22690385927


'''
위 추정값ㅇ이 얼마나 정확한지 알고 싶을 것이다.
이를 위해 scipy.stats.t.interval()를 사용해 일반화 오차의 95% 신뢰 구간을 계산할 수 있다.
'''

from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
confidence_interval = np.sqrt(stats.t.interval(confidence, len(squared_errors)-1, loc=squared_errors.mean(),
                                  scale=stats.sem(squared_errors)))
print(confidence_interval) # 신뢰 구간 계산
# 결과: [45685.10470776 49691.25001878]

'''
하이퍼파리미터 튜닝을 많이 했다면 교차 검증을 사용해 측정한 것보다 조금 성능이 낮은 것이 보통이다.
(우리 시스템이 검증 데이터에서 좋은 성능을 내도록 세밀하게 튜닝되었기 때문에 새로운 데이터셋에는 잘 작동하지 않을 가능성이 크다.)
이 예제에서는 성능이 낮아지진 않았니만, 이런 경우가 생기더라도 테스트 세트에서 성능 수치를 좋게 하려고
하이퍼파라미터를 튜닝하려 시도해서는 안된다. 그렇게 향상된 성능은 새로운 데이터에 일반화되기 어렵다.
'''






















