import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from itertools import permutations
import os
import sys
from time import time
from functools import lru_cache
import warnings

warnings.filterwarnings("ignore")

matplotlib.use('Agg')


def generate_cities(n_cities=20, seed=42):
    np.random.seed(seed)
    return np.random.rand(n_cities, 2) * 100


def distance_matrix(points):
    n = len(points)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(points[i] - points[j])
    return dist


def calculate_distance(points, path):
    return sum(np.linalg.norm(points[path[i]] - points[path[i + 1]])
               for i in range(len(path) - 1))


def exact_tsp(points):
    n = len(points)
    if n <= 1:
        return [0], 0.0

    min_path = None
    min_dist = float('inf')

    for perm in permutations(range(1, n)):
        current_path = [0] + list(perm) + [0]
        current_dist = calculate_distance(points, current_path)

        if current_dist < min_dist:
            min_dist = current_dist
            min_path = current_path

    return min_path, min_dist


def nearest_neighbor_tsp(points):
    path = [0]
    unvisited = set(range(1, len(points)))

    while unvisited:
        last = path[-1]
        next_node = min(unvisited, key=lambda x: np.linalg.norm(points[last] - points[x]))
        path.append(next_node)
        unvisited.remove(next_node)

    path.append(0)
    return path, calculate_distance(points, path)


def held_karp_tsp(points):
    n = len(points)
    if n <= 1:
        return [0], 0.0

    dist = distance_matrix(points)

    # Кэш для подзадач!!!!
    @lru_cache(maxsize=None)
    def dp(mask, last):
        if mask == (1 << n) - 1:
            return dist[last][0], [0]

        min_dist = float('inf')
        best_path = []

        for city in range(n):
            if not (mask & (1 << city)):
                new_mask = mask | (1 << city)
                d, path = dp(new_mask, city)
                total = dist[last][city] + d

                if total < min_dist:
                    min_dist = total
                    best_path = [city] + path

        return min_dist, best_path

    # Рекурсивный вызов
    total_dist, path = dp(1, 0)
    full_path = [0] + path
    return full_path, total_dist


def solve_tsp(points, method='auto'):
    n = len(points)

    if method == 'exact':
        return exact_tsp(points)
    elif method == 'greedy':
        return nearest_neighbor_tsp(points)
    elif method == 'held-karp':
        return held_karp_tsp(points)
    else:
        if n <= 5:
            return exact_tsp(points)
        elif n <= 12:
            return held_karp_tsp(points)
        else:
            return nearest_neighbor_tsp(points)


def cluster_cities(cities, n_clusters):
    try:
        kmeans = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=10,
            random_state=42
        )
        labels = kmeans.fit_predict(cities[1:]) #исключение депо

        clusters = []
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0] + 1
            cluster = np.vstack([cities[0], cities[cluster_indices]])
            clusters.append(cluster)

        clusters.sort(key=lambda x: len(x))
        return clusters

    except Exception as e:
        print(f"Ошибка кластеризации: {str(e)}")
        return None


def plot_solution(cities, routes, title, filename):
    try:
        plt.figure(figsize=(14, 10))

        # Города
        plt.scatter(cities[1:, 0], cities[1:, 1],
                    c='black', s=150, label='Города', zorder=3)
        plt.scatter(cities[0, 0], cities[0, 1],
                    c='red', s=300, marker='s', label='Депо', zorder=4)

        # Маршруты
        colors = plt.cm.tab10.colors
        for i, route in enumerate(routes):
            plt.plot(route[:, 0], route[:, 1], 'o-',
                     color=colors[i % len(colors)],
                     linewidth=3,
                     markersize=10,
                     markeredgecolor='black',
                     markeredgewidth=1,
                     label=f'Коммивояжёр {i + 1}',
                     zorder=2)

        plt.title(title, fontsize=16, pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        os.makedirs('results', exist_ok=True)
        save_path = os.path.join('results', filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Визуализация сохранена в {save_path}")

    except Exception as e:
        print(f"Ошибка визуализации: {str(e)}")


def solve_mtsp(cities, n_salesmen, method='auto'):
    try:
        print(f"\nНачало решения для {len(cities)} городов и {n_salesmen} коммивояжёров")
        print(f"Используемый метод: {method}")
        start_time = time()

        # Кластеризация
        clusters = cluster_cities(cities, n_salesmen)
        if clusters is None:
            return False

        print("Распределение по кластерам:")
        for i, cluster in enumerate(clusters):
            print(f"  Кластер {i + 1}: {len(cluster)} городов")

        # Решение для каждого кластера
        routes = []
        total_distance = 0

        for i, cluster in enumerate(clusters):
            cluster_start = time()
            path, dist = solve_tsp(cluster, method)
            routes.append(cluster[path])
            total_distance += dist
            print(f"  Кластер {i + 1}: длина {dist:.2f}, время {time() - cluster_start:.2f}с")

        # Визуализация
        method_name = {
            'exact': 'Точный метод',
            'greedy': 'Жадный алгоритм',
            'held-karp': 'Алгоритм Хелда-Карпа',
            'auto': 'Комбинированный метод'
        }.get(method, method)

        title = (f"Решение mTSP ({method_name})\n"
                 f"Коммивояжёров: {n_salesmen} | Городов: {len(cities)}\n"
                 f"Общая дистанция: {total_distance:.2f}")

        filename = f'mtsp_solution_{method}.png'
        plot_solution(cities, routes, title, filename)

        print(f"\nОбщее время выполнения: {time() - start_time:.2f} секунд")
        return True

    except Exception as e:
        print(f"\nОшибка: {str(e)}")
        return False


def get_user_input():
    print("=== Решение задачи mTSP ===")

    try:
        n_cities = int(input("Введите количество городов (минимум 2): "))
        if n_cities < 2:
            raise ValueError("Количество городов должно быть не менее 2")

        n_salesmen = int(input("Введите количество коммивояжёров (минимум 1): "))
        if n_salesmen < 1:
            raise ValueError("Количество коммивояжёров должно быть не менее 1")

        print("\nДоступные методы решения:")
        print("1. Точный метод (полный перебор, до 10 городов)")
        print("2. Жадный алгоритм (быстрый, но не всегда оптимальный)")
        print("3. Алгоритм Хелда-Карпа (точный, до 15 городов)")
        print("4. Комбинированный (автоматический выбор лучшего метода)")

        method_choice = input("Выберите метод (1-4): ").strip()
        methods = {
            '1': 'exact',
            '2': 'greedy',
            '3': 'held-karp',
            '4': 'auto'
        }

        method = methods.get(method_choice, 'auto')
        return n_cities, n_salesmen, method

    except ValueError as e:
        print(f"Ошибка ввода: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        n_cities, n_salesmen, method = get_user_input()

        cities = generate_cities(n_cities)

        success = solve_mtsp(cities, n_salesmen, method)

        if not success:
            print("Не удалось найти решение")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nПрервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\nНепредвиденная ошибка: {str(e)}")
        sys.exit(1)