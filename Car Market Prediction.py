import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dosyadan veri alınır
data = pd.read_csv('Arabalar.csv')

# 1- gereksiz sütunların silinmesi
data.drop(['date', 'engine_volume'], axis=1, inplace=True)

# 2- NaN veri silinmesi
data.dropna(inplace=True)

# 3- veri düzeltme
data['km'] = data['km'].astype(str)
data['year'] = data['year'].astype(str)
for index, row in data.iterrows():
    li = row['engine_power'].split('-')
    if (len(li) == 2):
        li[0] = li[0].strip()
        li[1] = li[1].strip()
        li[0] = str(int((int(li[0]) + int(li[1])) / 2))
        del li[1]
    data.at[index, 'engine_power'] = li[0]
    data.at[index, 'km'] = row['km'].replace('.', '')
    data.at[index, 'price'] = row['price'].replace('.', '')

# 4- veri tiplerini düzenleme
data['year'] = data['year'].astype(int)
data['km'] = data['km'].astype(float)
data['engine_power'] = data['engine_power'].astype(int)
data['price'] = data['price'].astype(int)

# 5- grafikler
def draw_graphic(attribute, rotat, xl, yl):
    data[attribute].value_counts().plot(kind='bar', color = list('rgbkymc'))
    plt.xticks(rotation=rotat)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.show()

### markaların dağılımı
##draw_graphic('brand', 90, 'Marka', 'Sayılar')
##
### renk dağılımı
##draw_graphic('color', 90, 'Renk', 'Sayılar')
##
### vites dağılımı
##draw_graphic('gear', 0, 'Vites Türü', 'Sayılar')
##
### yaş dağılımı
##draw_graphic('year', 90, 'Yaş', 'Sayılar')
##
### yakıt dağılımı
##draw_graphic('fuel', 0, 'Yakıt Türü', 'Sayılar')
##
### motor gücü dağılımı
##draw_graphic('engine_power', 90, 'Motor Gücü', 'Sayılar')

# 6- verilerin ölçeklenmesi
from sklearn.preprocessing import MinMaxScaler
def scale(column):
    sc = MinMaxScaler()
    data[[column]] = sc.fit_transform(data[[column]])
    return sc
data_scales = {'year':[], 'km':[], 'engine_power':[], 'price':[]}
for col in data_scales.keys():
    data_scales[col] = scale(col)

def scale_value(value, column):
    min_val = data_scales[column].data_min_[0]
    max_val = data_scales[column].data_max_[0]
    scale = 1 / (max_val - min_val)
    m = 0 - min_val * scale
    return float('{:.6f}'.format(value * scale + m))

# 7- kategorik verilerin nümerikleştirilmesi
from sklearn.preprocessing import LabelEncoder
def encode(column):
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    return le
data_classes = {'brand':[], 'model':[], 'color':[], 'fuel':[], 'gear':[]}
for col in data_classes.keys():
    data_classes[col] = encode(col)

def encode_value(value, column):
    return list(data_classes[column].classes_).index(value)

# 8- verilerin ayrılması
from sklearn.model_selection import train_test_split
X = data.iloc[:,0:8]
y = data.iloc[:,8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=X['brand'])

from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
def get_metrics(y_test, y_pred):
    print('R^2 Korelasyon: ' + str(r2_score(y_test, y_pred)))
    print('Maksimum Hata: ' + str(max_error(y_test, y_pred)))
    print('Mutlak Hata: ' + str(mean_absolute_error(y_test, y_pred)))
    print('Hata Karesi: ' + str(mean_squared_error(y_test, y_pred)))

##print('DOĞRUSAL REGRESYON' + 10*'-') 
### 9- doğrusal regresyon ile eğitim
##from sklearn.linear_model import LinearRegression
##lr = LinearRegression()
##lr.fit(X_train, y_train)
##y_pred = lr.predict(X_test)
##get_metrics(y_test, y_pred)

print('KARAR AĞACI' + 10*'-') 
# 10- karar ağacı ile eğitim
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(criterion='friedman_mse', random_state=0)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
get_metrics(y_test, y_pred)

##print('DESTEK VEKTÖR MAKİNESİ' + 10*'-') 
### 11- destek vektör makineleri ile eğitim
##from sklearn.svm import SVR
##svm = SVR()
##svm.fit(X_train, y_train)
##y_pred = svm.predict(X_test)
##get_metrics(y_test, y_pred)

# 12- tahmin aşaması
def predict_value(brand, model, year, km, color, fuel, gear, engine_power, trainer):
    brand = encode_value(brand, 'brand')
    model = encode_value(model, 'model')
    year = scale_value(year, 'year')
    km = scale_value(km, 'km')
    color = encode_value(color, 'color')
    fuel = encode_value(fuel, 'fuel')
    gear = encode_value(gear, 'gear')
    engine_power = scale_value(engine_power, 'engine_power')
    d = {'brand':brand, 'model':model, 'year':year, 'km':km, 'color':color, 'fuel':fuel, 'gear':gear, 'engine_power':engine_power}
    dataframe = pd.DataFrame(data=d, index=[0])
    raw_price = trainer.predict(dataframe)
    min_val = data_scales['price'].data_min_[0]
    max_val = data_scales['price'].data_max_[0]
    scale = 1 / (max_val - min_val)
    m = 0 - min_val * scale
    return (raw_price - m) / scale

print(predict_value('Renault',
                    'Renault 1.5 dCi Joy',
                    2017,
                    54000,
                    'Beyaz',
                    'Dizel',
                    'Düz',
                    90,
                    dt)[0])
    
    
