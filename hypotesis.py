#%%
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
# %%
data = pd.read_csv('car_price_prediction.csv')
print(df)
# %%
# Данные
x = data['Prod. year']
y = data['Price']

# Рассчитываем коэффициент корреляции и p-value
correlation, p_value = stats.pearsonr(x, y)
print(f'Коэффициент корреляции: {correlation}')
print(f'p-value: {p_value}')

# Строим линейную регрессию
model = LinearRegression()
X = x.values.reshape(-1, 1)
model.fit(X, y)

# Выводим уравнение регрессии
slope = model.coef_[0]
intercept = model.intercept_
print(f'Уравнение регрессии: y = {slope:.2f} * x + {intercept:.2f}')

# Строим график рассеяния и линию регрессии
plt.figure(figsize=(10, 6))
plt.scatter(x, y, marker='o')
plt.plot(x, model.predict(X), color='red', linewidth=2)
plt.title('Линейная зависимость между Год производства и Ценой')
plt.xlabel('Год производства')
plt.ylabel('Цена')
plt.show()

# Проверяем статистическую значимость
alpha = 0.05
if p_value < alpha:
    print("Отвергаем нулевую гипотезу. Существует статистически значимая зависимость.")
else:
    print("Не отвергаем нулевую гипотезу. Нет статистически значимой зависимости.")
# %%
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns 


# %%
data = pd.read_csv('car_price_prediction.csv')
# %%
# Замена символа '-' на NaN в столбце 'Levy'
data['Levy'] = pd.to_numeric(data['Levy'], errors='coerce')

# Выбор всех доступных признаков, за исключением 'Price' (целевая переменная)
features = data.columns.tolist()
features.remove('Price')

# Определение числовых и категориальных признаков
numeric_features = data.select_dtypes(include=['float64']).columns
categorical_features = data.select_dtypes(include=['object']).columns

# Создание предобработчика числовых признаков
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Создание предобработчика категориальных признаков
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Создание трансформатора колонок
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Создание итогового пайплайна
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(data[features], data['Price'], test_size=0.2, random_state=42)

# Обучение модели и предсказание
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Оценка точности модели
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')




# %%
# Замена символа '-' на NaN в столбце 'Levy'
data['Levy'] = pd.to_numeric(data['Levy'], errors='coerce')

# Преобразование 'Mileage' в числовой формат
data['Mileage'] = data['Mileage'].replace({',': '', ' km': ''}, regex=True)
data['Mileage'] = pd.to_numeric(data['Mileage'], errors='coerce')

# Визуализация связи между ценой и пробегом
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Mileage', y='Price', data=data)
plt.title('Scatter Plot of Mileage vs Price')
plt.xlabel('Mileage (in km)')
plt.ylabel('Price')
plt.show()

# Вычисление коэффициента корреляции
correlation = data['Mileage'].corr(data['Price'])
print(f'Коэффициент корреляции между Mileage и Price: {correlation}')
print(f'Коэффициент корреляции между Mileage и Price: {correlation}')
# %%
import scipy.stats as stats
# %%
# Рассчитаем коэффициент корреляции и p-value
correlation, p_value = stats.pearsonr(data['Mileage'].dropna(), data['Price'].dropna())

# Выведем результаты
print(f'Коэффициент корреляции между Mileage и Price: {correlation}')
print(f'p-value: {p_value}')

# Проведем статистический тест
alpha = 0.05
if p_value < alpha:
    print("Отвергаем нулевую гипотезу. Существует статистически значимая корреляция.")
else:
    print("Не отвергаем нулевую гипотезу. Нет статистически значимой корреляции.")
# %%
