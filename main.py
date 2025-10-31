import pandas as pd
import numpy as np

print("=" * 50)
print("ШАГ 1: Загрузка и первичный осмотр данных")
print("=" * 50)

try:
    df = pd.read_csv('ncr_ride_bookings.csv')
    print("Файл 'ncr_ride_bookings.csv' загружен успешно!")
except FileNotFoundError:
    try:
        df = pd.read_csv('датаcer_uber_pides_bookings.csv')
        print("Файл 'датаcer_uber_pides_bookings.csv' загружен успешно!")
    except FileNotFoundError:
        print("ОШИБКА: Файл не найден!")
        exit()

print("\n1. Первые 5 строк данных:")
print(df.head())
print("\n" + "-" * 30)

print("2. Общая информация о данных:")
df.info()
print("\n" + "-" * 30)

print("3. Статистическое описание числовых столбцов:")
print(df.describe())
print("\n" + "-" * 30)

print("4. Количество строк и столбцов:")
print(f"Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")
print("\n" + "-" * 30)

print("ВСЕ СТОЛБЦЫ В ДАННЫХ:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\n" + "-" * 30)

print("=" * 50)
print("ШАГ 2: СТАТИСТИЧЕСКИЙ ОБЗОР ДАННЫХ")
print("=" * 50)

print("1. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ ПО СТОЛБЦАМ:")
print("-" * 40)
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_info = pd.DataFrame({
    'Пропущенных значений': missing_values,
    'Процент пропусков': missing_percentage.round(2)
})
print(missing_info)
print("\n" + "-" * 30)

print("2. ТИПЫ ДАННЫХ ПО СТОЛБЦАМ:")
print("-" * 40)
dtypes_info = pd.DataFrame({
    'Тип данных': df.dtypes,
    'Уникальных значений': df.nunique()
})
print(dtypes_info)
print("\n" + "-" * 30)

print("3. УНИКАЛЬНЫЕ ЗНАЧЕНИЯ В КАТЕГОРИАЛЬНЫХ СТОЛБЦАХ:")
print("-" * 40)


categorical_cols = df.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    print(f"\nСтолбец: {col}")
    print(f"Количество уникальных значений: {df[col].nunique()}")
    print("Уникальные значения:")
    unique_vals = df[col].unique()

    if len(unique_vals) > 10:
        print(f"Первые 10: {unique_vals[:10]}")
        print(f"... и ещё {len(unique_vals) - 10} значений")
    else:
        for val in unique_vals:
            print(f"  - {val}")
    print("-" * 20)

print("\n" + "-" * 30)

print("4. ДЕТАЛЬНЫЙ АНАЛИЗ КЛЮЧЕВЫХ КАТЕГОРИАЛЬНЫХ СТОЛБЦОВ:")
print("-" * 40)

status_cols = [col for col in df.columns if 'status' in col.lower()]
vehicle_cols = [col for col in df.columns if any(x in col.lower() for x in ['vehicle', 'auto', 'car', 'type'])]

print("Возможные столбцы статуса:", status_cols)
print("Возможные столбцы типа транспорта:", vehicle_cols)

if status_cols:
    status_col = status_cols[0]
    print(f"\nАНАЛИЗ СТОЛБЦА '{status_col}':")
    print(f"Количество уникальных статусов: {df[status_col].nunique()}")
    print("Распределение по статусам:")
    status_counts = df[status_col].value_counts()
    print(status_counts)

    print("\nПроцентное распределение:")
    status_percentage = (df[status_col].value_counts(normalize=True) * 100).round(2)
    print(status_percentage)
else:
    print("\nСтолбец статуса бронирования не найден")

if vehicle_cols:
    vehicle_col = vehicle_cols[0]
    print(f"\nАНАЛИЗ СТОЛБЦА '{vehicle_col}':")
    print(f"Количество уникальных типов транспорта: {df[vehicle_col].nunique()}")
    print("Распределение по типам транспорта:")
    vehicle_counts = df[vehicle_col].value_counts()
    print(vehicle_counts)

    print("\nПроцентное распределение:")
    vehicle_percentage = (df[vehicle_col].value_counts(normalize=True) * 100).round(2)
    print(vehicle_percentage)
else:
    print("\nСтолбец типа транспорта не найден")

print("\n" + "-" * 30)

print("5. ОБЩАЯ СТАТИСТИКА ЧИСЛОВЫХ СТОЛБЦОВ:")
print("-" * 40)
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    numeric_stats = df[numeric_cols].describe().round(2)
    print(numeric_stats)
else:
    print("Числовые столбцы не найдены")

print("\n" + "-" * 30)


print("=" * 50)
print("ШАГ 3: Выборка и фильтрация данных")
print("=" * 50)

print("Для выполнения заданий нам нужно определить правильные названия столбцов.")
print("Посмотрите на список выше и введите нужные названия:")

all_columns = df.columns.tolist()

booking_id_cols = [col for col in all_columns if 'booking' in col.lower() and 'id' in col.lower()]
datetime_cols = [col for col in all_columns if any(x in col.lower() for x in ['date', 'time', 'datetime'])]
status_cols = [col for col in all_columns if 'status' in col.lower()]
vehicle_cols = [col for col in all_columns if 'vehicle' in col.lower() or 'auto' in col.lower()]
payment_cols = [col for col in all_columns if 'payment' in col.lower()]
value_cols = [col for col in all_columns if 'value' in col.lower() or 'price' in col.lower() or 'cost' in col.lower()]

print(f"Возможные столбцы для Booking ID: {booking_id_cols}")
print(f"Возможные столбцы для даты/времени: {datetime_cols}")
print(f"Возможные столбцы для статуса: {status_cols}")
print(f"Возможные столбцы для типа транспорта: {vehicle_cols}")
print(f"Возможные столбцы для оплаты: {payment_cols}")
print(f"Возможные столбцы для стоимости: {value_cols}")

booking_id_col = booking_id_cols[0] if booking_id_cols else 'Booking ID'
datetime_col = datetime_cols[0] if datetime_cols else 'booking_datetime'
status_col = status_cols[0] if status_cols else 'Booking Status'
vehicle_col = vehicle_cols[0] if vehicle_cols else 'Vehicle Type'
payment_col = payment_cols[0] if payment_cols else 'Payment Method'
value_col = value_cols[0] if value_cols else 'Booking Value'

print(f"\nИспользуем столбцы:")
print(f"Booking ID: {booking_id_col}")
print(f"Дата/время: {datetime_col}")
print(f"Статус: {status_col}")
print(f"Тип транспорта: {vehicle_col}")
print(f"Оплата: {payment_col}")
print(f"Стоимость: {value_col}")

print("\n" + "-" * 30)

print("1. Выбранные столбцы (первые 5 строк):")
try:
    selected_columns = df[[booking_id_col, datetime_col, status_col, vehicle_col, payment_col]]
    print(selected_columns.head())
except KeyError as e:
    print(f"Ошибка: столбец {e} не найден. Проверьте названия столбцов.")

print("\n" + "-" * 30)

print("2. Бронирования со статусом 'Cancelled by Driver':")
try:
    cancelled_by_driver = df[df[status_col] == 'Cancelled by Driver']
    print(cancelled_by_driver)
    print(f"Найдено: {len(cancelled_by_driver)} записей")
except KeyError:
    print("Столбец статуса не найден")

print("\n" + "-" * 30)

print("3. Бронирования Auto с Booking Value >500:")
try:
    auto_high_value = df[(df[vehicle_col] == 'Auto') & (df[value_col] > 500)]
    print(auto_high_value)
    print(f"Найдено: {len(auto_high_value)} записей")
except KeyError:
    print("Не удалось выполнить фильтрацию. Проверьте названия столбцов.")

print("\n" + "-" * 30)

print("4. Бронирования за март 2024 года:")
try:
    # Сначала найдем правильное название столбца с датой
    datetime_columns = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'datetime'])]

    if datetime_columns:
        datetime_col = datetime_columns[0]
        print(f"Используем столбец даты: {datetime_col}")

        df[datetime_col] = pd.to_datetime(df[datetime_col])
        march_2024 = df[(df[datetime_col] >= '2024-03-01') & (df[datetime_col] <= '2024-03-31')]

        if len(march_2024) > 0:
            print(march_2024)
            print(f"Найдено: {len(march_2024)} записей")
        else:
            print("Записей за март 2024 года не найдено")
    else:
        print("Столбцы с датой не найдены в данных")

except Exception as e:
    print(f"Ошибка при работе с датами: {e}")

print("=" * 50)
print("АНАЛИЗ ЗАВЕРШЕН!")
print("=" * 50)

print("=" * 60)
print("ШАГ 4: ГЕНЕРАЦИЯ И ПОДБОР Х-ФАКТОРОВ ДЛЯ МОДЕЛИ СТОИМОСТИ")
print("=" * 60)

# Создаем копию DataFrame для работы
df_enhanced = df.copy()

print("Создаем новые features для модели...")
print("-" * 50)

# Находим столбец с датой/временем автоматически
datetime_columns = [col for col in df_enhanced.columns if any(x in col.lower() for x in ['date', 'time', 'datetime'])]
if datetime_columns:
    datetime_column = datetime_columns[0]
    print(f"Используем столбец даты: {datetime_column}")
    try:
        df_enhanced[datetime_column] = pd.to_datetime(df_enhanced[datetime_column])
    except Exception as e:
        print(f"Ошибка преобразования даты: {e}")
        datetime_column = None
else:
    datetime_column = None
    print("ВНИМАНИЕ: Столбец с датой/временем не найден!")

# Находим другие важные столбцы
value_columns = [col for col in df_enhanced.columns if any(x in col.lower() for x in ['value', 'price', 'cost', 'amount'])]
value_col = value_columns[0] if value_columns else None

vehicle_columns = [col for col in df_enhanced.columns if any(x in col.lower() for x in ['vehicle', 'auto', 'car', 'type'])]
vehicle_col = vehicle_columns[0] if vehicle_columns else None

status_columns = [col for col in df_enhanced.columns if 'status' in col.lower()]
status_col = status_columns[0] if status_columns else None

print(f"Столбец стоимости: {value_col}")
print(f"Столбец транспорта: {vehicle_col}")
print(f"Столбец статуса: {status_col}")

print("\nПроверка типов данных перед созданием фич...")
print(df_enhanced.dtypes)

# Предположим, что у нас есть следующие базовые столбцы (замените на актуальные):
# df['booking_datetime'], df['pickup_lat'], df['pickup_lng'], df['dropoff_lat'], df['dropoff_lng'],
# df['vehicle_type'], df['driver_id'], df['booking_value']

# Для демонстрации создадим копию DataFrame
df_enhanced = df.copy()

print("Создаем новые features для модели...")
print("-" * 50)
# Находим столбец с датой/временем автоматически
datetime_columns = [col for col in df_enhanced.columns if any(x in col.lower() for x in ['date', 'time', 'datetime'])]
if datetime_columns:
    datetime_column = datetime_columns[0]
    print(f"Используем столбец даты: {datetime_column}")
    df_enhanced[datetime_column] = pd.to_datetime(df_enhanced[datetime_column])
else:
    datetime_column = None
    print("ВНИМАНИЕ: Столбец с датой/временем не найден!")
    print("ВНИМАНИЕ: Столбец с датой/временем не найден!")

print("1. ВРЕМЕННЫЕ ФАКТОРЫ:")
# Извлекаем компоненты даты и времени
df_enhanced['hour_of_day'] = df_enhanced[datetime_column].dt.hour
df_enhanced['day_of_week'] = df_enhanced[datetime_column].dt.dayofweek # 0=Пн, 6=Вс
df_enhanced['month'] = df_enhanced[datetime_column].dt.month
df_enhanced['is_weekend'] = (df_enhanced['day_of_week'] >= 5).astype(int)

# Временные периоды дня
def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 22:
        return 'Evening'
    else:
        return 'Late_Night'

df_enhanced['time_of_day'] = df_enhanced['hour_of_day'].apply(get_time_of_day)

# Пиковые/непиковые часы (можно настроить под специфику города)
df_enhanced['is_peak_hour'] = ((df_enhanced['hour_of_day'] >= 7) & (df_enhanced['hour_of_day'] <= 10) |
                              (df_enhanced['hour_of_day'] >= 17) & (df_enhanced['hour_of_day'] <= 20)).astype(int)

print("✓ Часы, день недели, месяц, выходные/будни")
print("✓ Время суток (утро/день/вечер/ночь)")
print("✓ Пиковые часы")

print("\n2. ГЕОГРАФИЧЕСКИЕ ФАКТОРЫ:")

# Проверяем, есть ли у нас координаты в данных
coord_columns = ['pickup_lat', 'pickup_lng', 'dropoff_lat', 'dropoff_lng']
has_coordinates = all(col in df_enhanced.columns for col in coord_columns)

if has_coordinates:
    print("Найдены координаты - рассчитываем расстояние...")


    # Функция для расчета расстояния без внешних библиотек
    def calculate_distance(row):
        try:
            # Простая формула расчета расстояния по координатам (приблизительная)
            lat1, lon1 = row['pickup_lat'], row['pickup_lng']
            lat2, lon2 = row['dropoff_lat'], row['dropoff_lng']

            # Формула гаверсинусов для расчета расстояния между двумя точками на сфере
            from math import radians, sin, cos, sqrt, atan2

            # Преобразование градусов в радианы
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

            # Разница координат
            dlat = lat2 - lat1
            dlon = lon2 - lon1

            # Формула гаверсинусов
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            radius_earth = 6371  # Радиус Земли в км
            distance = radius_earth * c

            return distance
        except:
            return None


    df_enhanced['distance_km'] = df_enhanced.apply(calculate_distance, axis=1)
    print(f"✓ Расстояние рассчитано для {df_enhanced['distance_km'].notna().sum()} поездок")

    # Базовые географические фичи на основе координат
    df_enhanced['is_short_trip'] = (df_enhanced['distance_km'] < 3).astype(int)
    df_enhanced['is_long_trip'] = (df_enhanced['distance_km'] > 10).astype(int)
    print("✓ Флаги коротких/длинных поездок")

else:
    print("Координаты не найдены - пропускаем расчет расстояния")
    # Создаем заглушки чтобы код не ломался
    df_enhanced['distance_km'] = None
    df_enhanced['is_short_trip'] = 0
    df_enhanced['is_long_trip'] = 0

print("✓ Базовые географические фичи созданы")

print("\n3. ФАКТОРЫ СПРОСА И ПРЕДЛОЖЕНИЯ:")

# Агрегация по временным промежуткам для создания features спроса
hourly_demand = df_enhanced.groupby(['hour_of_day', 'day_of_week']).size().reset_index(name='historical_hourly_demand')
df_enhanced = df_enhanced.merge(hourly_demand, on=['hour_of_day', 'day_of_week'], how='left')

# "Коэффициент загруженности" - можно рассчитать на основе:
# - количества поездок в этот час
# - количества доступных водителей
# - погодных условий

# Простой пример: нормализованный спрос по часам
df_enhanced['demand_factor'] = df_enhanced['historical_hourly_demand'] / df_enhanced['historical_hourly_demand'].max()

print("✓ Исторический спрос по часам и дням недели")
print("✓ Коэффициент загруженности/спроса")

print("\n4. ФАКТОРЫ ТРАНСПОРТА И ВОДИТЕЛЯ:")

# Тип транспортного средства (уже есть, но нужно закодировать)
vehicle_type_mapping = {'Auto': 1, 'Mini': 1, 'Sedan': 2, 'SUV': 3, 'Luxury': 4}
df_enhanced['vehicle_type_encoded'] = df_enhanced[vehicle_col].map(vehicle_type_mapping).fillna(0).astype(int)

# Если есть данные о водителях:
# df_enhanced['driver_rating'] = ... # рейтинг водителя
# df_enhanced['driver_experience_months'] = ... # опыт водителя

# Стоимость обслуживания типа транспорта (можно задать вручную)
vehicle_cost_map = {'Auto': 1.0, 'Mini': 1.2, 'Sedan': 1.5, 'SUV': 2.0, 'Luxury': 3.0}
df_enhanced['vehicle_cost_multiplier'] = df_enhanced[vehicle_col].map(vehicle_cost_map).fillna(0).astype(int)

print("✓ Закодированный тип транспорта")
print("✓ Множитель стоимости для типа транспорта")
print("✓ Рейтинг и опыт водителя (если данные доступны)")

print("\n5. ВНЕШНИЕ ФАКТОРЫ:")

# Создаем базовые внешние факторы на основе имеющихся данных
print("Создаем базовые внешние факторы...")

# 1. Сезонность на основе месяца
df_enhanced['season'] = df_enhanced['month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
})


# 2. Простые праздничные дни (можно расширить под вашу страну)
def is_holiday(month, day_of_week):
    # Простой пример: выходные + некоторые праздники
    if day_of_week >= 5:  # Суббота и воскресенье
        return 1
    # Можно добавить конкретные даты праздников
    return 0


df_enhanced['is_holiday'] = df_enhanced.apply(
    lambda row: is_holiday(row['month'], row['day_of_week']), axis=1
)


# 3. Условные погодные условия на основе сезона и времени суток
def estimate_weather_condition(season, hour, month):
    if season == 'Winter':
        return 'Cold'
    elif season == 'Summer':
        if 11 <= hour <= 16:  # Дневные часы летом
            return 'Hot'
        else:
            return 'Warm'
    elif season == 'Spring' or season == 'Autumn':
        return 'Mild'
    else:
        return 'Normal'


df_enhanced['estimated_weather'] = df_enhanced.apply(
    lambda row: estimate_weather_condition(row['season'], row['hour_of_day'], row['month']), axis=1
)


# 4. Уровень пробок на основе времени суток и дня недели
def estimate_traffic(hour, day_of_week):
    # Пиковые часы пробок
    morning_peak = (7 <= hour <= 10)
    evening_peak = (17 <= hour <= 20)
    weekend = (day_of_week >= 5)

    if weekend:
        return 'Low'
    elif morning_peak or evening_peak:
        return 'High'
    else:
        return 'Medium'


df_enhanced['traffic_level'] = df_enhanced.apply(
    lambda row: estimate_traffic(row['hour_of_day'], row['day_of_week']), axis=1
)

# 5. Кодируем категориальные фичи
weather_mapping = {'Cold': 0, 'Mild': 1, 'Warm': 2, 'Hot': 3, 'Normal': 1}
traffic_mapping = {'Low': 0, 'Medium': 1, 'High': 2}

df_enhanced['weather_encoded'] = df_enhanced['estimated_weather'].map(weather_mapping)
df_enhanced['traffic_encoded'] = df_enhanced['traffic_level'].map(traffic_mapping)

print("✓ Сезонность (зима/весна/лето/осень)")
print("✓ Выходные и праздничные дни")
print("✓ Оценка погодных условий")
print("✓ Уровень пробок (низкий/средний/высокий)")
print("✓ Закодированные версии для ML")

print("\n6. ИСТОРИЧЕСКИЕ ФАКТОРЫ:")

print("Создаем исторические и сессионные фичи...")

# Проверяем наличие идентификаторов
id_columns = [col for col in df_enhanced.columns if
              any(x in col.lower() for x in ['user', 'driver', 'customer', 'rider'])]

if id_columns:
    print(f"Найдены идентификаторы: {id_columns}")

    # Берем первый попавшийся ID столбец для демонстрации
    id_column = id_columns[0]

    # 1. Количество бронирований на пользователя/водителя
    booking_counts = df_enhanced.groupby(id_column).size().reset_index(name=f'{id_column}_booking_count')
    df_enhanced = df_enhanced.merge(booking_counts, on=id_column, how='left')

    # 2. Средняя стоимость бронирований по пользователю/водителю
    avg_value_by_id = df_enhanced.groupby(id_column)[value_col].mean().reset_index(
        name=f'{id_column}_avg_booking_value')
    df_enhanced = df_enhanced.merge(avg_value_by_id, on=id_column, how='left')

    # 3. Время с последнего бронирования (если есть временная метка)
    if datetime_column in df_enhanced.columns:
        df_enhanced = df_enhanced.sort_values(by=[id_column, datetime_column])
        df_enhanced[f'{id_column}_time_since_last_booking'] = df_enhanced.groupby(id_column)[
                                                                  datetime_column].diff().dt.total_seconds() / 3600  # в часах

    print(f"✓ Количество бронирований по {id_column}")
    print(f"✓ Средняя стоимость по {id_column}")
    print(f"✓ Время с последнего бронирования по {id_column}")

# 4. Анализ отмен (если есть столбец статуса)
if status_col in df_enhanced.columns:
    print("Анализируем статистику отмен...")

    # Общий rate отмен в данных
    total_cancellations = (df_enhanced[status_col].str.contains('cancel', case=False, na=False)).sum()
    df_enhanced['global_cancellation_rate'] = total_cancellations / len(df_enhanced)

    # Временные паттерны отмен
    cancellation_by_hour = df_enhanced[df_enhanced[status_col].str.contains('cancel', case=False, na=False)].groupby(
        'hour_of_day').size()
    total_by_hour = df_enhanced.groupby('hour_of_day').size()
    hour_cancellation_rates = (cancellation_by_hour / total_by_hour).fillna(0).astype(int)

    # Добавляем rate отмен по часам
    df_enhanced['hourly_cancellation_rate'] = df_enhanced['hour_of_day'].map(hour_cancellation_rates)

    print("✓ Глобальный rate отмен")
    print("✓ Rate отмен по часам суток")

# 5. Сессионные фичи - накопленная статистика по временным отрезкам
if datetime_column in df_enhanced.columns:
    print("Создаем сессионные фичи...")

    try:
        # Упрощенный подход - группировка по дням
        df_enhanced['booking_date'] = df_enhanced[datetime_column].dt.date

        # Количество бронирований в тот же день
        daily_counts = df_enhanced.groupby('booking_date').size().reset_index(name='daily_bookings')
        df_enhanced = df_enhanced.merge(daily_counts, on='booking_date', how='left')

        # Средняя стоимость в тот же день
        if value_col in df_enhanced.columns:
            daily_avg_value = df_enhanced.groupby('booking_date')[value_col].mean().reset_index(name='daily_avg_value')
            df_enhanced = df_enhanced.merge(daily_avg_value, on='booking_date', how='left')

        # Относительный порядок бронирования в течение дня
        df_enhanced = df_enhanced.sort_values(by=[datetime_column])
        df_enhanced['booking_order_in_day'] = df_enhanced.groupby('booking_date').cumcount() + 1

        # Удаляем временный столбец
        df_enhanced = df_enhanced.drop('booking_date', axis=1)

        print("✓ Бронирования за день")
        print("✓ Средняя стоимость за день")
        print("✓ Порядковый номер брони в течение дня")

    except Exception as e:
        print(f"Ошибка при создании сессионных фич: {e}")
        # Создаем заглушки без использования fillna для object типов
        df_enhanced['daily_bookings'] = 1
        if value_col in df_enhanced.columns:
            avg_val = float(df_enhanced[value_col].mean())
            df_enhanced['daily_avg_value'] = avg_val
        else:
            df_enhanced['daily_avg_value'] = 0.0
        df_enhanced['booking_order_in_day'] = 1

# 6. Дополнительные агрегации по типам транспорта (упрощенная версия)
if vehicle_col in df_enhanced.columns:
    print("Создаем упрощенные фичи транспорта...")

    try:
        # Простая популярность - просто количество использования типа транспорта
        vehicle_popularity = df_enhanced[vehicle_col].value_counts(normalize=True)
        df_enhanced['vehicle_overall_popularity'] = df_enhanced[vehicle_col].map(vehicle_popularity)

        # Простая фича - является ли тип транспорта популярным (>10%)
        df_enhanced['is_popular_vehicle'] = (df_enhanced['vehicle_overall_popularity'] > 0.1).astype(int)

        print("✓ Общая популярность типов транспорта")
        print("✓ Флаг популярного транспорта")

    except Exception as e:
        print(f"Ошибка при создании упрощенных фич транспорта: {e}")
        df_enhanced['vehicle_overall_popularity'] = 0.0
        df_enhanced['is_popular_vehicle'] = 0

print("\n7. ПОДГОТОВКА ДАННЫХ ДЛЯ ML:")

# Выбираем только числовые и категориальные колонки для модели
feature_columns = [
    'hour_of_day', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
    'vehicle_type_encoded', 'vehicle_cost_multiplier', 'demand_factor'
    # 'distance_km', 'driver_rating', 'temperature', 'is_rainy', 'is_holiday', 'traffic_index'
]

# One-Hot Encoding для категориальных признаков
categorical_columns = ['time_of_day']  # добавьте другие категориальные колонки
if categorical_columns:
    df_encoded = pd.get_dummies(df_enhanced, columns=categorical_columns, prefix=categorical_columns)
    # Добавляем новые колонки в feature_columns
    new_categorical_features = [col for col in df_encoded.columns if any(col.startswith(prefix) for prefix in categorical_columns)]
    feature_columns.extend(new_categorical_features)
else:
    df_encoded = df_enhanced.copy()

# Целевая переменная
target_column = value_col  # 'booking_value'

# Создаем финальный dataset для ML
ml_dataset = df_encoded[feature_columns + [target_column]].copy()

# Удаляем строки с пропущенными значениями
ml_dataset = ml_dataset.dropna()

print(f"ФИНАЛЬНЫЙ НАБОР ДАННЫХ ДЛЯ ML:")
print(f"- Количество features: {len(feature_columns)}")
print(f"- Количество samples: {len(ml_dataset)}")
print(f"- Features: {feature_columns}")
print(f"- Target: {target_column}")

print("\nПервые 5 строк подготовленных данных:")
print(ml_dataset.head())

print("\n" + "=" * 60)
print("РЕКОМЕНДАЦИИ ПО ВЫБОРУ МОДЕЛИ И ДАЛЬНЕЙШИМ ШАГАМ")
print("=" * 60)

print("""
ДЛЯ РЕГРЕССИИ СТОИМОСТИ ПОЕЗДКИ РЕКОМЕНДУЮСЯ:

1. **Gradient Boosting модели** (наиболее эффективны):
   - XGBoost (XGBRegressor)
   - LightGBM (LGBMRegressor) 
   - CatBoost (CatBoostRegressor)

2. **Классические ML модели** (для базового сравнения):
   - Random Forest Regressor
   - Linear Regression (с полиномиальными features)
   - Support Vector Regression (SVR)

3. **Нейронные сети** (для сложных нелинейных зависимостей):
   - MLP Regressor (многослойный перцептрон)
   - TabNet (специализирован для табличных данных)

СЛЕДУЮЩИЕ ШАГИ:
""")

# Практический пример подготовки данных для ML
print("ПРАКТИЧЕСКИЙ ПРИМЕР ДЛЯ НАЧАЛА РАБОТЫ:")
print("-" * 40)

# Подготовка финального датасета
print("1. Подготовка данных для ML:")

# Выбираем только числовые колонки и исключаем целевую переменную
numeric_features = ml_dataset.select_dtypes(include=[np.number]).columns.tolist()
if target_column in numeric_features:
    numeric_features.remove(target_column)

print(f"   - Числовые features: {len(numeric_features)}")
print(f"   - Размер датасета: {ml_dataset.shape}")

if len(numeric_features) > 0:
    print("""
2. Базовый пример кода для начала:

   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestRegressor
   from sklearn.metrics import mean_absolute_error, r2_score

   # Разделение данных
   X = ml_dataset[numeric_features]
   y = ml_dataset['{}']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Базовая модель
   model = RandomForestRegressor(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)

   # Предсказания и оценка
   y_pred = model.predict(X_test)
   print(f"MAE: {{mean_absolute_error(y_test, y_pred):.2f}}")
   print(f"R²: {{r2_score(y_test, y_pred):.3f}}")

   # Важность фич
   feature_importance = pd.DataFrame({{
       'feature': numeric_features,
       'importance': model.feature_importances_
   }}).sort_values('importance', ascending=False)
   print(feature_importance.head(10))
""".format(target_column))

print("""
МЕТРИКИ ДЛЯ ОЦЕНКИ:
- MAE (Mean Absolute Error) - простая интерпретация
- RMSE (Root Mean Square Error) - штрафует за большие ошибки
- R² (R-squared) - доля объясненной дисперсии  
- MAPE (Mean Absolute Percentage Error) - процентная ошибка

РЕКОМЕНДАЦИИ:
1. Начните с RandomForest для быстрого базового уровня
2. Затем пробуйте Gradient Boosting (XGBoost/LightGBM)
3. Используйте кросс-валидацию для надежной оценки
4. Анализируйте важность фич для feature engineering
""")

print("=" * 60)
print("АНАЛИЗ И ГЕНЕРАЦИЯ ФИЧЕЙ ЗАВЕРШЕНЫ!")
print("=" * 60)
print("Готовый датасет для ML сохранен в переменной: ml_dataset")
print(f"Размер: {ml_dataset.shape}, Features: {len(numeric_features)}, Target: {target_column}")
print("=" * 60)
