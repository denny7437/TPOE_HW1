import pandas as pd
from ab_testing import ABTestingSplitSystem

def main():
    # Загрузка данных
    try:
        df = pd.read_csv('train_data.csv')
        print("Данные успешно загружены")
        print(f"Размер данных: {df.shape}")
    except FileNotFoundError:
        print("Ошибка: Файл train_data.csv не найден")
        return
    
    # Создание экземпляра системы A/B тестирования
    ab_system = ABTestingSplitSystem(df)
    
    # Поиск похожих групп
    groups = ab_system.find_groups(group_num=10)  # Генерируем 10 групп, ищем 3 похожие
    
    if groups is None:
        print("Не удалось найти похожие группы")
        return
    
    print(f"Найдено {len(groups)} похожих групп")
    
    # Генерация случайных групп для сравнения
    random_groups = ab_system.generate_groups(3)
    
    # Визуализация результатов
    ab_system.visualize_groups(groups, random_groups)

if __name__ == "__main__":
    main() 