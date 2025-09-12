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
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    march_2024 = df[(df[datetime_col] >= '2024-03-01') & (df[datetime_col] <= '2024-03-31')]
    print(march_2024)
    print(f"Найдено: {len(march_2024)} записей")
except KeyError:
    print("Столбец с датой не найден")
except Exception as e:
    print(f"Ошибка при работе с датами: {e}")

print("=" * 50)
print("АНАЛИЗ ЗАВЕРШЕН!")
print("=" * 50)