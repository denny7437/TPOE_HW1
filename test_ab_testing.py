import unittest
import pandas as pd
import numpy as np
from ab_testing import ABTestingSplitSystem
import random
import os

class TestABTestingSplitSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Создаем тестовые данные"""
        # Создаем синтетические данные для тестирования
        np.random.seed(42)  # Для воспроизводимости результатов
        
        # Создаем данные для 100 пользователей за 10 дней
        user_ids = list(range(1, 101))
        dates = pd.date_range(start='2024-01-01', periods=10)
        
        data = []
        for user_id in user_ids:
            for date in dates:
                value = np.random.normal(100, 15)  # Генерируем случайные значения
                data.append({
                    'user_id': user_id,
                    'date': date,
                    'value': value
                })
        
        cls.test_df = pd.DataFrame(data)
        cls.ab_system = ABTestingSplitSystem(cls.test_df)

    def test_group_sizes(self):
        """Проверяем, что группы примерно одинакового размера"""
        n_groups = 3
        groups = self.ab_system.generate_groups(n_groups)
        
        # Проверяем количество групп
        self.assertEqual(len(groups), n_groups)
        
        # Проверяем размеры групп
        unique_users_per_group = [len(group['user_id'].unique()) for group in groups]
        
        # Разница между максимальным и минимальным размером группы должна быть не больше 1
        self.assertLessEqual(max(unique_users_per_group) - min(unique_users_per_group), 1)
        
        # Проверяем, что сумма пользователей во всех группах равна общему количеству пользователей
        total_users = len(self.test_df['user_id'].unique())
        self.assertEqual(sum(unique_users_per_group), total_users)

    def test_no_user_overlap(self):
        """Проверяем, что пользователи не повторяются между группами"""
        groups = self.ab_system.generate_groups(3)
        
        # Собираем всех пользователей из всех групп
        all_users = set()
        for group in groups:
            group_users = set(group['user_id'].unique())
            
            # Проверяем пересечение с уже собранными пользователями
            self.assertEqual(len(group_users & all_users), 0)
            all_users.update(group_users)

    def test_data_integrity(self):
        """Проверяем, что данные пользователей сохраняются полностью"""
        groups = self.ab_system.generate_groups(3)
        
        # Проверяем, что для каждого пользователя сохранены все его записи
        for group in groups:
            for user_id in group['user_id'].unique():
                user_records = group[group['user_id'] == user_id]
                self.assertEqual(len(user_records), 10)  # У каждого пользователя должно быть 10 записей

    def test_random_distribution(self):
        """Проверяем, что распределение действительно случайное"""
        # Генерируем группы несколько раз и проверяем, что они различаются
        first_groups = self.ab_system.generate_groups(3)
        second_groups = self.ab_system.generate_groups(3)
        
        # Сравниваем пользователей в первых группах двух разных разбиений
        first_users = set(first_groups[0]['user_id'].unique())
        second_users = set(second_groups[0]['user_id'].unique())
        
        # Группы должны отличаться (вероятность полного совпадения крайне мала)
        self.assertNotEqual(first_users, second_users)

    def test_similar_groups(self):
        """Проверяем, что похожие группы определяются как похожие"""
        # Создаем две группы с похожими характеристиками
        np.random.seed(42)
        
        # Создаем данные для двух групп по 50 пользователей
        data1 = []
        data2 = []
        
        # Генерируем данные с одинаковыми параметрами распределения
        for user_id in range(1, 51):
            mean_value = np.random.normal(100, 5)  # Среднее значение для пользователя
            std_value = np.random.uniform(10, 20)  # Стандартное отклонение для пользователя
            
            for day in range(10):
                # Первая группа
                value1 = np.random.normal(mean_value, std_value)
                data1.append({
                    'user_id': user_id,
                    'date': f'2024-01-{day+1}',
                    'value': value1
                })
                
                # Вторая группа
                value2 = np.random.normal(mean_value, std_value)
                data2.append({
                    'user_id': user_id + 50,  # Разные ID для разных групп
                    'date': f'2024-01-{day+1}',
                    'value': value2
                })
        
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)
        
        # Проводим тесты
        p_value_means, p_value_var = self.ab_system.conduct_tests_on_pair(df1, df2)
        
        # Проверяем, что p-значения больше порогового уровня значимости (например, 0.05)
        # Это означает, что нулевая гипотеза о равенстве средних/дисперсий не отвергается
        self.assertGreater(p_value_means, 0.05)
        self.assertGreater(p_value_var, 0.05)

    def test_different_groups(self):
        """Проверяем, что различные группы определяются как различные"""
        np.random.seed(42)
        
        # Создаем данные для двух групп с разными характеристиками
        data1 = []
        data2 = []
        
        # Первая группа: среднее = 100, std = 15
        for user_id in range(1, 51):
            for day in range(10):
                value = np.random.normal(100, 15)
                data1.append({
                    'user_id': user_id,
                    'date': f'2024-01-{day+1}',
                    'value': value
                })
        
        # Вторая группа: среднее = 120, std = 25 (явно отличающиеся параметры)
        for user_id in range(51, 101):
            for day in range(10):
                value = np.random.normal(120, 25)
                data2.append({
                    'user_id': user_id,
                    'date': f'2024-01-{day+1}',
                    'value': value
                })
        
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)
        
        # Проводим тесты
        p_value_means, p_value_var = self.ab_system.conduct_tests_on_pair(df1, df2)
        
        # Проверяем, что p-значения меньше порогового уровня значимости
        # Это означает, что нулевая гипотеза о равенстве средних/дисперсий отвергается
        self.assertLess(p_value_means, 0.05)
        self.assertLess(p_value_var, 0.05)

    def test_conduct_tests_similar_groups(self):
        """Проверяем, что функция находит похожие группы"""
        np.random.seed(42)
        
        # Создаем 5 похожих групп
        groups = []
        for group_id in range(5):
            data = []
            for user_id in range(group_id * 20 + 1, (group_id + 1) * 20 + 1):
                mean_value = np.random.normal(100, 5)
                std_value = np.random.uniform(10, 20)
                
                for day in range(10):
                    value = np.random.normal(mean_value, std_value)
                    data.append({
                        'user_id': user_id,
                        'date': f'2024-01-{day+1}',
                        'value': value
                    })
            groups.append(pd.DataFrame(data))
        
        # Ищем 3 похожие группы
        result = self.ab_system.conduct_tests(groups, required_groups=3)
        
        # Проверяем, что нашли группы
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)

    def test_conduct_tests_different_groups(self):
        """Проверяем, что функция не находит похожие группы среди различных"""
        np.random.seed(42)
        
        # Создаем 5 различных групп
        groups = []
        means = [80, 90, 100, 110, 120]
        stds = [10, 15, 20, 25, 30]
        
        for group_id, (mean, std) in enumerate(zip(means, stds)):
            data = []
            for user_id in range(group_id * 20 + 1, (group_id + 1) * 20 + 1):
                for day in range(10):
                    value = np.random.normal(mean, std)
                    data.append({
                        'user_id': user_id,
                        'date': f'2024-01-{day+1}',
                        'value': value
                    })
            groups.append(pd.DataFrame(data))
        
        # Пытаемся найти 3 похожие группы
        result = self.ab_system.conduct_tests(groups, required_groups=3)
        
        # Проверяем, что не нашли группы
        self.assertIsNone(result)

    def test_conduct_tests_insufficient_groups(self):
        """Проверяем обработку случая с недостаточным количеством групп"""
        # Создаем только 2 группы
        groups = self.ab_system.generate_groups(2)
        
        # Пытаемся найти 3 похожие группы
        result = self.ab_system.conduct_tests(groups, required_groups=3)
        
        # Проверяем, что получили None
        self.assertIsNone(result)

    def test_find_groups_invalid_input(self):
        """Проверяем обработку некорректного входного параметра"""
        with self.assertRaises(ValueError):
            self.ab_system.find_groups(2)  # Меньше необходимого количества групп

    def test_find_groups_success(self):
        """Проверяем успешный поиск похожих групп"""
        np.random.seed(42)  # Для воспроизводимости
        
        # Создаем данные с похожими характеристиками
        data = []
        for user_id in range(1, 101):
            mean_value = np.random.normal(100, 5)
            std_value = np.random.uniform(10, 20)
            
            for day in range(10):
                value = np.random.normal(mean_value, std_value)
                data.append({
                    'user_id': user_id,
                    'date': f'2024-01-{day+1}',
                    'value': value
                })
        
        # Создаем новый экземпляр системы с подготовленными данными
        test_system = ABTestingSplitSystem(pd.DataFrame(data))
        
        # Пытаемся найти группы с небольшим количеством итераций
        result = test_system.find_groups(group_num=5, max_iterations=100)
        
        # Проверяем результат
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        
        # Проверяем, что найденные группы действительно похожи
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                p_means, p_var = test_system.conduct_tests_on_pair(result[i], result[j])
                self.assertGreater(p_means, 0.05)
                self.assertGreater(p_var, 0.05)

    def test_find_groups_no_similar(self):
        """Проверяем случай, когда похожие группы не могут быть найдены"""
        # Создаем данные с экспоненциально растущими характеристиками
        data = []
        for user_id in range(1, 101):
            # Экспоненциально растущие значения для каждого пользователя
            mean_value = np.exp(user_id / 10)  # Экспоненциальный рост средних значений
            std_value = np.exp(user_id / 20)   # Экспоненциальный рост стандартных отклонений
            
            for day in range(10):
                value = np.random.normal(mean_value, std_value)
                data.append({
                    'user_id': user_id,
                    'date': f'2024-01-{day+1}',
                    'value': value
                })
        
        # Создаем новый экземпляр системы с подготовленными данными
        test_system = ABTestingSplitSystem(pd.DataFrame(data))
        
        # Пытаемся найти группы с небольшим количеством итераций
        result = test_system.find_groups(group_num=5, max_iterations=10)
        
        # Проверяем, что группы не найдены
        self.assertIsNone(result)

    def test_calculate_user_metrics(self):
        """
        Тест функции _calculate_user_metrics
        Проверяет корректность вычисления метрик для групп пользователей
        """
        # Создаем тестовые данные с явно выраженными группами
        user_data = []
        # Группа 1: низкие значения
        for i in range(10):
            user_data.extend([{'user_id': f'low_{i}', 'value': random.uniform(1, 10)} for _ in range(5)])
        # Группа 2: средние значения
        for i in range(10):
            user_data.extend([{'user_id': f'med_{i}', 'value': random.uniform(40, 60)} for _ in range(5)])
        # Группа 3: высокие значения
        for i in range(10):
            user_data.extend([{'user_id': f'high_{i}', 'value': random.uniform(90, 100)} for _ in range(5)])
            
        test_df = pd.DataFrame(user_data)
        system = ABTestingSplitSystem(test_df)
        
        # Вычисляем метрики
        metrics = system._calculate_user_metrics()
        
        # Проверяем основные характеристики результата
        self.assertIsInstance(metrics, pd.DataFrame)
        self.assertGreaterEqual(len(metrics), 3)  # Минимум 3 группы
        
        # Проверяем наличие всех необходимых колонок
        required_columns = {'group_id', 'size', 'mean', 'std', 'cv', 'min', 'max', 'median', 'user_ids'}
        self.assertTrue(all(col in metrics.columns for col in required_columns))
        
        # Проверяем, что все пользователи распределены по группам
        total_users = sum(len(group) for group in metrics['user_ids'])
        self.assertEqual(total_users, len(test_df['user_id'].unique()))
        
        # Проверяем корректность вычисленных метрик
        self.assertTrue((metrics['std'] >= 0).all())  # Стандартное отклонение неотрицательно
        self.assertTrue((metrics['cv'] >= 0).all())   # Коэффициент вариации неотрицателен
        self.assertTrue((metrics['max'] >= metrics['min']).all())  # Максимум больше минимума
        
        # Проверяем, что группы действительно различаются
        means = metrics['mean'].values
        self.assertTrue(any(abs(means[i] - means[j]) > 10 for i in range(len(means)) for j in range(i+1, len(means))))

    def test_calculate_user_metrics_edge_cases(self):
        """
        Тест функции _calculate_user_metrics для граничных случаев
        """
        # Случай 1: Все значения одинаковые
        user_ids = []
        values = []
        for i in range(15):
            user_ids.extend([f'user_{i}'] * 3)
            values.extend([10.0] * 3)
        
        data1 = pd.DataFrame({
            'user_id': user_ids,
            'value': values
        })
        system1 = ABTestingSplitSystem(data1)
        metrics1 = system1._calculate_user_metrics()
        self.assertTrue((metrics1['std'] == 0).all())  # Стандартное отклонение должно быть 0
        self.assertTrue((metrics1['cv'] == 0).all())   # Коэффициент вариации должен быть 0
        
        # Случай 2: Очень маленькие значения
        user_ids = []
        values = []
        for i in range(15):
            user_ids.extend([f'user_{i}'] * 3)
            values.extend([1e-10] * 3)
        
        data2 = pd.DataFrame({
            'user_id': user_ids,
            'value': values
        })
        system2 = ABTestingSplitSystem(data2)
        metrics2 = system2._calculate_user_metrics()
        self.assertFalse(metrics2['cv'].isna().any())  # Не должно быть NaN в cv
        
        # Случай 3: Минимальное количество пользователей
        user_ids = []
        values = []
        for i in range(3):
            user_ids.extend([f'user_{i}'] * 3)
            values.extend(range(i*3, (i+1)*3))
        
        data3 = pd.DataFrame({
            'user_id': user_ids,
            'value': values
        })
        system3 = ABTestingSplitSystem(data3)
        metrics3 = system3._calculate_user_metrics()
        self.assertGreaterEqual(len(metrics3), 3)  # Должно быть минимум 3 группы

    def test_visualize_groups(self):
        """
        Тест функции visualize_groups
        Проверяет создание визуализации для групп
        """
        # Создаем тестовые группы с разными характеристиками
        np.random.seed(42)
        
        # Создаем даты для тестовых данных
        dates = pd.date_range(start='2024-01-01', periods=10)
        
        # Создаем три группы с разными характеристиками
        groups = []
        means = [100, 110, 120]
        stds = [10, 12, 15]
        
        for group_id, (mean, std) in enumerate(zip(means, stds)):
            data = []
            for user_id in range(group_id * 20 + 1, (group_id + 1) * 20 + 1):
                for date in dates:
                    value = np.random.normal(mean, std)
                    data.append({
                        'user_id': user_id,
                        'date': date,
                        'value': value
                    })
            groups.append(pd.DataFrame(data))
        
        # Создаем случайные группы
        random_groups = self.ab_system.generate_groups(3)
        
        try:
            # Проверяем визуализацию только с группами
            self.assertTrue(self.ab_system.visualize_groups(groups))
            self.assertTrue(os.path.exists('results/group_comparison.png'))
            self.assertTrue(os.path.exists('results/group_statistics.csv'))
            self.assertTrue(os.path.exists('results/metadata.csv'))
            for i in range(len(groups)):
                self.assertTrue(os.path.exists(f'results/group_{i+1}.csv'))
            
            # Удаляем файлы после проверки
            if os.path.exists('results'):
                import shutil
                shutil.rmtree('results')
            
            # Проверяем визуализацию с случайными группами
            self.assertTrue(self.ab_system.visualize_groups(groups, random_groups))
            self.assertTrue(os.path.exists('results/group_comparison.png'))
            self.assertTrue(os.path.exists('results/group_statistics.csv'))
            self.assertTrue(os.path.exists('results/metadata.csv'))
            for i in range(len(groups)):
                self.assertTrue(os.path.exists(f'results/group_{i+1}.csv'))
            for i in range(len(random_groups)):
                self.assertTrue(os.path.exists(f'results/random_group_{i+1}.csv'))
            
            # Проверяем содержимое метаданных
            metadata = pd.read_csv('results/metadata.csv')
            self.assertEqual(metadata['number_of_groups'].iloc[0], len(groups))
            self.assertTrue(metadata['has_random_groups'].iloc[0])
            
            # Удаляем файлы после проверки
            if os.path.exists('results'):
                import shutil
                shutil.rmtree('results')
            
        except Exception as e:
            # В случае ошибки удаляем созданные файлы
            if os.path.exists('results'):
                import shutil
                shutil.rmtree('results')
            self.fail(f"Визуализация вызвала исключение: {str(e)}")

    def test_visualize_groups_edge_cases(self):
        """
        Тест функции visualize_groups для граничных случаев
        """
        # Случай 1: Группы с одинаковыми значениями
        groups = []
        for i in range(3):
            data = []
            for user_id in range(i * 5 + 1, (i + 1) * 5 + 1):
                for _ in range(10):
                    data.append({
                        'user_id': user_id,
                        'date': pd.Timestamp('2024-01-01'),
                        'value': 100.0
                    })
            groups.append(pd.DataFrame(data))
        
        try:
            self.assertTrue(self.ab_system.visualize_groups(groups))
            self.assertTrue(os.path.exists('results/group_comparison.png'))
            self.assertTrue(os.path.exists('results/group_statistics.csv'))
            
            # Проверяем статистики для групп с одинаковыми значениями
            stats = pd.read_csv('results/group_statistics.csv')
            self.assertTrue((stats['Стд. откл.'] == 0).all())
            
            # Удаляем файлы после проверки
            if os.path.exists('results'):
                import shutil
                shutil.rmtree('results')
            
        except Exception as e:
            if os.path.exists('results'):
                import shutil
                shutil.rmtree('results')
            self.fail(f"Визуализация для одинаковых значений вызвала исключение: {str(e)}")
        
        # Случай 2: Группы с минимальным размером
        min_groups = []
        for i in range(3):
            data = []
            data.append({
                'user_id': i,
                'date': pd.Timestamp('2024-01-01'),
                'value': float(i)
            })
            min_groups.append(pd.DataFrame(data))
        
        try:
            self.assertTrue(self.ab_system.visualize_groups(min_groups))
            self.assertTrue(os.path.exists('results/group_comparison.png'))
            self.assertTrue(os.path.exists('results/group_statistics.csv'))
            
            # Проверяем статистики для минимальных групп
            stats = pd.read_csv('results/group_statistics.csv')
            self.assertTrue((stats['Размер'] == 1).all())
            
            # Удаляем файлы после проверки
            if os.path.exists('results'):
                import shutil
                shutil.rmtree('results')
            
        except Exception as e:
            if os.path.exists('results'):
                import shutil
                shutil.rmtree('results')
            self.fail(f"Визуализация для минимальных групп вызвала исключение: {str(e)}")

if __name__ == '__main__':
    unittest.main() 