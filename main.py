import pandas as pd
import numpy as np
from ab_testing import ABTestingSplitSystem
import os
from datetime import datetime, timedelta

def create_test_data(n_users: int = 1000, n_days: int = 100) -> pd.DataFrame:
    """
    Создает тестовые данные для A/B тестирования
    
    Args:
        n_users (int): Количество пользователей
        n_days (int): Количество дней
        
    Returns:
        pd.DataFrame: DataFrame с тестовыми данными
    """
    print(f"Создаю тестовые данные для {n_users} пользователей за {n_days} дней...")
    
    try:
        # Создаем даты
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        
        # Создаем данные
        data = []
        for user_id in range(1, n_users + 1):
            # Генерируем базовые характеристики пользователя
            user_mean = np.random.normal(100, 20)  # Среднее значение для пользователя
            user_std = np.random.uniform(5, 15)    # Стандартное отклонение для пользователя
            
            # Генерируем значения для каждого дня
            for date in dates:
                value = np.random.normal(user_mean, user_std)
                data.append({
                    'user_id': user_id,
                    'date': date.strftime('%Y-%m-%d'),  # Преобразуем дату в строку
                    'value': max(0, value)  # Убеждаемся, что значения неотрицательные
                })
            
            # Показываем прогресс каждые 100 пользователей
            if user_id % 100 == 0:
                print(f"Обработано {user_id} пользователей ({user_id/n_users*100:.1f}%)")
        
        # Создаем DataFrame
        df = pd.DataFrame(data)
        print("Данные успешно созданы!")
        
        # Проверяем корректность данных
        assert len(df) == n_users * n_days, "Неверное количество записей"
        assert df['user_id'].nunique() == n_users, "Неверное количество пользователей"
        assert df['date'].nunique() == n_days, "Неверное количество дней"
        assert not df['value'].isna().any(), "Обнаружены пропущенные значения"
        
        return df
    
    except Exception as e:
        print(f"Ошибка при создании тестовых данных: {str(e)}")
        raise

def save_data(df: pd.DataFrame, directory: str = 'data') -> None:
    """
    Сохраняет данные в CSV файл
    
    Args:
        df (pd.DataFrame): DataFrame для сохранения
        directory (str): Директория для сохранения
    """
    # Создаем директорию, если она не существует
    os.makedirs(directory, exist_ok=True)
    
    # Сохраняем данные
    filename = os.path.join(directory, 'train_data.csv')
    df.to_csv(filename, index=False)
    print(f"Данные сохранены в {filename}")

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует данные из широкого формата (даты в колонках) в длинный формат
    
    Args:
        df (pd.DataFrame): DataFrame в широком формате
        
    Returns:
        pd.DataFrame: DataFrame в длинном формате с колонками user_id, date, value
    """
    # Сбрасываем индекс и используем его как user_id
    df = df.reset_index().rename(columns={'index': 'user_id'})
    
    # Преобразуем в длинный формат
    df_long = df.melt(
        id_vars=['user_id'],
        var_name='date',
        value_name='value'
    )
    
    # Преобразуем даты в datetime
    df_long['date'] = pd.to_datetime(df_long['date'])
    
    # Сортируем по user_id и date
    df_long = df_long.sort_values(['user_id', 'date'])
    
    return df_long

def main():
    # Генерируем тестовые данные
    print("Генерация тестовых данных...")
    df = create_test_data()
    
    # Создаем директорию для данных, если она не существует
    os.makedirs('data', exist_ok=True)
    
    # Сохраняем данные
    df.to_csv('data/train_data.csv', index=False)
    print("Данные сохранены в data/train_data.csv")
    
    # Создаем экземпляр системы A/B тестирования
    system = ABTestingSplitSystem(df)
    
    # Находим группы
    print("\nПоиск статистически похожих групп...")
    groups = system.find_groups(10, max_iterations=1000)
    
    if groups is not None:
        print("Найдены статистически похожие группы!")
        
        # Визуализируем результаты
        print("\nСоздание визуализации...")
        system.visualize_groups(groups)
        print("Результаты сохранены в директории results/")
    else:
        print("Не удалось найти статистически похожие группы.")

if __name__ == "__main__":
    main() 