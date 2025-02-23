import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Union
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

class ABTestingSplitSystem:
    def __init__(self, data: pd.DataFrame):
        """
        Инициализация системы сплит-тестирования
        
        Args:
            data (pd.DataFrame): Исходные данные для анализа
        """
        self.data = data
        self.user_metrics = self._calculate_user_metrics()
    
    def _calculate_user_metrics(self) -> pd.DataFrame:
        """
        Вычисление метрик для каждой группы пользователей.
        Группы формируются на основе схожести поведения пользователей.
        
        Returns:
            pd.DataFrame: DataFrame с метриками групп пользователей, содержащий следующие колонки:
                - group_id: ID группы
                - size: размер группы (количество пользователей)
                - mean: среднее значение метрики по группе
                - std: стандартное отклонение метрики
                - cv: коэффициент вариации группы (std/mean)
                - min: минимальное значение в группе
                - max: максимальное значение в группе
                - median: медианное значение
                - user_ids: список ID пользователей в группе
        """
        # Сначала вычисляем метрики для каждого пользователя
        user_metrics = self.data.groupby('user_id')['value'].agg([
            ('mean', 'mean'),
            ('std', 'std')
        ]).fillna(0)
        
        # Проверяем, есть ли вариация в данных
        total_std = self.data['value'].std()
        if total_std < 1e-10:  # Если все значения практически одинаковые
            # Создаем три равные группы
            user_ids = list(user_metrics.index)
            group_size = len(user_ids) // 3
            remainder = len(user_ids) % 3
            
            group_metrics = []
            start_idx = 0
            for i in range(3):
                current_size = group_size + (1 if i < remainder else 0)
                group_users = user_ids[start_idx:start_idx + current_size]
                group_data = self.data[self.data['user_id'].isin(group_users)]
                
                metrics = {
                    'group_id': i,
                    'size': len(group_users),
                    'mean': group_data['value'].mean(),
                    'std': group_data['value'].std(),
                    'min': group_data['value'].min(),
                    'max': group_data['value'].max(),
                    'median': group_data['value'].median(),
                    'user_ids': group_users
                }
                metrics['cv'] = metrics['std'] / abs(metrics['mean']) if abs(metrics['mean']) > 1e-10 else 0
                group_metrics.append(metrics)
                start_idx += current_size
        else:
            # Нормализуем метрики для кластеризации
            scaler = StandardScaler()
            features = scaler.fit_transform(user_metrics)
            
            # Используем KMeans для группировки пользователей
            n_clusters = 3  # Фиксированное количество кластеров
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            user_clusters = kmeans.fit_predict(features)
            
            # Добавляем информацию о кластерах к метрикам пользователей
            user_metrics['cluster'] = user_clusters
            user_metrics['user_id'] = user_metrics.index
            
            # Вычисляем метрики для каждой группы
            group_metrics = []
            for cluster in range(n_clusters):
                # Получаем пользователей текущей группы
                group_users = user_metrics[user_metrics['cluster'] == cluster]['user_id'].tolist()
                group_data = self.data[self.data['user_id'].isin(group_users)]
                
                metrics = {
                    'group_id': cluster,
                    'size': len(group_users),
                    'mean': group_data['value'].mean(),
                    'std': group_data['value'].std(),
                    'min': group_data['value'].min(),
                    'max': group_data['value'].max(),
                    'median': group_data['value'].median(),
                    'user_ids': group_users
                }
                
                # Добавляем коэффициент вариации
                metrics['cv'] = metrics['std'] / abs(metrics['mean']) if abs(metrics['mean']) > 1e-10 else 0
                group_metrics.append(metrics)
        
        # Создаем DataFrame с метриками групп
        result = pd.DataFrame(group_metrics)
        
        # Проверяем корректность результатов
        assert len(result) == 3, "Должно быть ровно 3 группы"
        assert result['size'].sum() == len(self.data['user_id'].unique()), "Общее количество пользователей не совпадает"
        assert not result[['mean', 'std', 'cv', 'min', 'max', 'median']].isna().any().any(), "Обнаружены пропущенные значения"
        
        return result

    def _validate_groups(self, groups: List[pd.DataFrame]) -> bool:
        """
        Проверяет корректность разбиения на группы
        
        Args:
            groups (List[pd.DataFrame]): Список групп для проверки
            
        Returns:
            bool: True, если разбиение корректно
        """
        # Проверяем, что все группы не пустые
        if any(len(group) == 0 for group in groups):
            return False
            
        # Проверяем, что нет пересечений между группами
        all_users = set()
        for group in groups:
            group_users = set(group['user_id'].unique())
            if len(group_users & all_users) > 0:
                return False
            all_users.update(group_users)
            
        # Проверяем, что все пользователи распределены
        total_users = set(self.data['user_id'].unique())
        if all_users != total_users:
            return False
            
        return True

    def generate_groups(self, group_num: int) -> List[pd.DataFrame]:
        """
        Разбивает пользователей на группы примерно одинакового размера
        
        Args:
            group_num (int): Количество групп для разбиения
            
        Returns:
            List[pd.DataFrame]: Список DataFrame'ов с данными групп
        """
        # Получаем уникальные ID пользователей
        unique_users = self.data['user_id'].unique()
        
        # Перемешиваем пользователей случайным образом
        np.random.shuffle(unique_users)
        
        # Вычисляем размер каждой группы
        group_size = len(unique_users) // group_num
        remainder = len(unique_users) % group_num
        
        groups = []
        start_idx = 0
        
        # Распределяем пользователей по группам
        for i in range(group_num):
            # Если есть остаток, добавляем по одному пользователю к первым группам
            current_group_size = group_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_group_size
            
            # Получаем ID пользователей для текущей группы
            group_users = unique_users[start_idx:end_idx]
            
            # Создаем DataFrame для группы
            group_df = self.data[self.data['user_id'].isin(group_users)].copy()
            groups.append(group_df)
            
            start_idx = end_idx
        
        # Проверяем корректность разбиения
        assert self._validate_groups(groups), "Ошибка в разбиении на группы"
        
        return groups

    def conduct_tests_on_pair(self, df_1: pd.DataFrame, df_2: pd.DataFrame) -> Tuple[float, float]:
        """
        Проводит статистические тесты между двумя группами.
        Использует t-тест для сравнения средних и тест Левена для сравнения дисперсий.
        Также проверяет эффект размера (Cohen's d) и относительную разницу средних.
        
        Args:
            df_1 (pd.DataFrame): Данные первой группы
            df_2 (pd.DataFrame): Данные второй группы
            
        Returns:
            Tuple[float, float]: p-значения для тестов на среднее и дисперсию
        """
        # Вычисляем метрики для каждой группы
        metrics_1 = df_1.groupby('user_id')['value'].agg(['mean', 'std']).fillna(0)
        metrics_2 = df_2.groupby('user_id')['value'].agg(['mean', 'std']).fillna(0)
        
        mean_1 = metrics_1['mean'].mean()
        mean_2 = metrics_2['mean'].mean()
        std_1 = metrics_1['std'].mean()
        std_2 = metrics_2['std'].mean()
        
        # Проверяем относительную разницу средних
        relative_diff = abs(mean_1 - mean_2) / max(abs(mean_1), abs(mean_2))
        if relative_diff > 0.1:  # Если разница больше 10%
            return 0.0, 0.0
            
        # Проверяем относительную разницу стандартных отклонений
        std_relative_diff = abs(std_1 - std_2) / max(abs(std_1), abs(std_2))
        if std_relative_diff > 0.2:  # Если разница больше 20%
            return 0.0, 0.0
        
        # Проводим t-тест для средних значений
        t_stat, p_value_means = stats.ttest_ind(
            metrics_1['mean'],
            metrics_2['mean'],
            equal_var=False  # Используем тест Уэлча (не предполагаем равенство дисперсий)
        )
        
        # Проводим тест Левена для дисперсий
        _, p_value_var = stats.levene(
            metrics_1['std'],
            metrics_2['std']
        )
        
        # Вычисляем размер эффекта (Cohen's d) для средних
        pooled_std = np.sqrt((metrics_1['mean'].var() + metrics_2['mean'].var()) / 2)
        effect_size = abs(mean_1 - mean_2) / pooled_std
        
        # Если размер эффекта слишком большой (> 0.2), считаем группы различными
        if effect_size > 0.2:  # Малый размер эффекта по Cohen's d
            return 0.0, 0.0
        
        return p_value_means, p_value_var

    def conduct_tests(self, dfs: List[pd.DataFrame], required_groups: int = 3) -> Optional[List[pd.DataFrame]]:
        """
        Ищет required_groups похожих групп среди представленных.
        Группы считаются похожими, если p-значения для обоих тестов (среднее и дисперсия)
        превышают пороговое значение alpha.
        
        Args:
            dfs (List[pd.DataFrame]): Список групп для тестирования
            required_groups (int): Необходимое количество похожих групп
            
        Returns:
            Optional[List[pd.DataFrame]]: Список похожих групп или None, если не найдены
        """
        if len(dfs) < required_groups:
            return None
            
        # Пороговое значение для p-value
        alpha = 0.05
        
        # Перебираем все возможные комбинации групп размера required_groups
        for groups_combination in combinations(range(len(dfs)), required_groups):
            selected_groups = [dfs[i] for i in groups_combination]
            is_similar = True
            
            # Проверяем все пары групп в текущей комбинации
            for i in range(len(selected_groups)):
                for j in range(i + 1, len(selected_groups)):
                    p_value_means, p_value_var = self.conduct_tests_on_pair(
                        selected_groups[i],
                        selected_groups[j]
                    )
                    
                    # Если хотя бы один тест не прошел, группы не похожи
                    if p_value_means < alpha or p_value_var < alpha:
                        is_similar = False
                        break
                
                if not is_similar:
                    break
            
            # Если все тесты прошли успешно, возвращаем найденные группы
            if is_similar:
                return selected_groups
        
        # Если не нашли подходящие группы
        return None

    def find_groups(self, group_num: int, max_iterations: int = 100000) -> Optional[List[pd.DataFrame]]:
        """
        Находит необходимое количество похожих групп (по умолчанию 3) среди group_num сгенерированных групп.
        Использует стратегию многократных попыток с случайным разбиением.
        
        Args:
            group_num (int): Количество групп для генерации (должно быть >= 3)
            max_iterations (int): Максимальное количество попыток поиска
            
        Returns:
            Optional[List[pd.DataFrame]]: Список из 3 похожих групп или None, если не найдены
            
        Raises:
            ValueError: Если group_num меньше 3
        """
        if group_num < 3:
            raise ValueError("Количество групп должно быть не меньше 3")
            
        # Для отслеживания прогресса
        best_p_value = 0
        progress_check = max_iterations // 10
        
        for i in range(max_iterations):
            # Генерируем новое разбиение
            dfs = self.generate_groups(group_num)
            result = self.conduct_tests(dfs, required_groups=3)
            
            if result is not None:
                print(f"Найдены подходящие группы на итерации {i+1}")
                return result
                
            # Отслеживаем прогресс каждые 10% итераций
            if (i + 1) % progress_check == 0:
                # Проверяем лучшие p-значения для текущего разбиения
                best_p_value_current = 0
                for j in range(len(dfs)):
                    for k in range(j + 1, len(dfs)):
                        p_means, p_var = self.conduct_tests_on_pair(dfs[j], dfs[k])
                        best_p_value_current = max(best_p_value_current, min(p_means, p_var))
                
                best_p_value = max(best_p_value, best_p_value_current)
                print(f"Выполнено {(i+1)/max_iterations*100:.1f}% итераций. "
                      f"Лучшее p-значение: {best_p_value:.4f}")
        
        print("Не удалось найти подходящие группы после всех попыток")
        return None

    def visualize_groups(self, groups: List[pd.DataFrame], random_groups: Optional[List[pd.DataFrame]] = None, is_test: bool = False):
        """
        Визуализирует распределения метрик в найденных группах.
        Создает несколько графиков для сравнения групп:
        1. Распределение средних значений пользователей в каждой группе (box plot)
        2. Распределение значений метрик по времени (line plot)
        3. Гистограммы распределений значений в группах
        4. Сравнение с случайными группами (если предоставлены)
        
        Args:
            groups (List[pd.DataFrame]): Список найденных похожих групп
            random_groups (Optional[List[pd.DataFrame]]): Список случайных групп для сравнения
            is_test (bool): Флаг, указывающий, что метод вызван из тестов (влияет на удаление файлов)
        
        Returns:
            bool: True если визуализация создана успешно, False в случае ошибки
        """
        # Создаем папку results, если она не существует
        os.makedirs('results', exist_ok=True)
        
        # Настраиваем стиль графиков
        sns.set_style("whitegrid")
        
        # Создаем фигуру с несколькими подграфиками
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Box plot средних значений пользователей
        ax1 = plt.subplot(2, 2, 1)
        user_means_data = []
        labels = []
        
        for i, group in enumerate(groups):
            user_means = group.groupby('user_id')['value'].mean()
            user_means_data.append(user_means)
            labels.extend([f'Группа {i+1}'] * len(user_means))
        
        all_means = pd.concat(user_means_data)
        sns.boxplot(x=labels, y=all_means, ax=ax1)
        ax1.set_title('Распределение средних значений пользователей по группам')
        ax1.set_ylabel('Среднее значение')
        
        # 2. Line plot значений по времени
        ax2 = plt.subplot(2, 2, 2)
        for i, group in enumerate(groups):
            group_means = group.groupby('date')['value'].mean()
            ax2.plot(range(len(group_means)), group_means, label=f'Группа {i+1}', marker='o')
        
        ax2.set_title('Динамика средних значений по времени')
        ax2.set_xlabel('Время')
        ax2.set_ylabel('Среднее значение')
        if len(group_means) > 1:  # Добавляем легенду только если есть несколько точек
            ax2.legend()
        
        # 3. Гистограммы распределений
        ax3 = plt.subplot(2, 2, 3)
        has_variance = False
        for i, group in enumerate(groups):
            if group['value'].std() > 1e-10:  # Проверяем наличие вариации
                sns.histplot(data=group['value'], label=f'Группа {i+1}', ax=ax3, 
                           bins=30, alpha=0.5, stat='density')
                has_variance = True
        
        if not has_variance:
            # Если нет вариации, показываем точечный график
            unique_values = pd.concat([group['value'] for group in groups]).unique()
            ax3.scatter(unique_values, [1] * len(unique_values), 
                       label=['Группа ' + str(i+1) for i in range(len(groups))])
        
        ax3.set_title('Распределение значений в группах')
        ax3.set_xlabel('Значение')
        ax3.set_ylabel('Плотность' if has_variance else 'Наличие значения')
        ax3.legend()
        
        # 4. Сравнение с случайными группами (если есть)
        ax4 = plt.subplot(2, 2, 4)
        
        # Вычисляем и отображаем основные статистики для найденных групп
        stats_data = []
        for i, group in enumerate(groups):
            stats = {
                'Группа': f'Оптимальная {i+1}',
                'Среднее': group['value'].mean(),
                'Медиана': group['value'].median(),
                'Стд. откл.': group['value'].std(),
                'Размер': len(group['user_id'].unique())
            }
            stats_data.append(stats)
        
        # Добавляем статистики для случайных групп, если они есть
        if random_groups is not None:
            for i, group in enumerate(random_groups):
                stats = {
                    'Группа': f'Случайная {i+1}',
                    'Среднее': group['value'].mean(),
                    'Медиана': group['value'].median(),
                    'Стд. откл.': group['value'].std(),
                    'Размер': len(group['user_id'].unique())
                }
                stats_data.append(stats)
        
        # Создаем таблицу со статистиками
        stats_df = pd.DataFrame(stats_data)
        stats_df = stats_df.round(2)  # Округляем значения для лучшей читаемости
        
        # Отображаем таблицу
        ax4.axis('off')
        table = ax4.table(cellText=stats_df.values,
                         colLabels=stats_df.columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0.1, 0.1, 0.8, 0.8])
        
        # Настраиваем размер шрифта в таблице
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        
        # Добавляем общий заголовок
        plt.suptitle('Сравнение характеристик групп', fontsize=16, y=0.95)
        
        # Настраиваем расположение графиков
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Сохраняем график
        plt.savefig('results/group_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Сохраняем статистики в CSV
        stats_df.to_csv('results/group_statistics.csv', index=False)
        
        # Сохраняем данные групп
        for i, group in enumerate(groups):
            group.to_csv(f'results/group_{i+1}.csv', index=False)
        
        if random_groups is not None:
            for i, group in enumerate(random_groups):
                group.to_csv(f'results/random_group_{i+1}.csv', index=False)
        
        # Сохраняем метаданные
        metadata = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'number_of_groups': len(groups),
            'total_users': sum(len(group['user_id'].unique()) for group in groups),
            'total_records': sum(len(group) for group in groups),
            'has_random_groups': random_groups is not None
        }
        
        pd.DataFrame([metadata]).to_csv('results/metadata.csv', index=False)
        
        # Возвращаем True, чтобы показать, что визуализация создана успешно
        return True 