import numpy as np
import random
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def initialize_population(pop_size: int, n_models: int, init_weights=None):
    """
    Khởi tạo quần thể các vector trọng số (weights).
    Nếu init_weights được cung cấp, đưa nó vào quần thể và chuẩn hóa.
    """
    pop = []
    if init_weights is not None:
        init_weights = np.clip(init_weights, 1e-6, None)
        init_weights /= init_weights.sum()
        pop.append(init_weights)
        pop_size -= 1
    for _ in range(pop_size):
        w = np.random.rand(n_models)
        w /= w.sum()
        pop.append(w)
    return pop

def compute_distance_matrix(population):
    """
    Tính ma trận khoảng cách Euclidean giữa các cá thể trong quần thể.
    """
    size = len(population)
    dist_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            dist = np.linalg.norm(population[i] - population[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

def fitness_sharing(scores, dist_matrix, sigma_share=0.3, alpha=1):
    """
    Tính điểm fitness đã chia sẻ (fitness sharing).
    Penalize các cá thể gần nhau để giữ đa dạng.
    """
    size = len(scores)
    shared_scores = np.zeros(size)
    for i in range(size):
        sh_sum = 0
        for j in range(size):
            dist = dist_matrix[i, j]
            if dist < sigma_share:
                sh_sum += 1 - (dist / sigma_share) ** alpha
        shared_scores[i] = scores[i] / max(sh_sum, 1e-6)
    return shared_scores

def evaluate_with_niching(population, meta_X, y_true, metric='f1', sigma_share=0.3):
    """
    Đánh giá quần thể với niching (fitness sharing).
    Trả về danh sách điểm gốc (không chia sẻ) và quần thể đã được sắp xếp theo fitness chia sẻ.
    """
    metric_func = {
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred > 0.5),
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred > 0.5),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred > 0.5),
        'auc': lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
    }[metric]

    scores, normalized_pop = [], []
    for w in population:
        w = np.clip(w, 1e-6, None)
        w /= w.sum()
        normalized_pop.append(w)
        y_pred = np.dot(meta_X, w)
        scores.append(metric_func(y_true, y_pred))

    dist_matrix = compute_distance_matrix(normalized_pop)
    shared_scores = fitness_sharing(scores, dist_matrix, sigma_share=sigma_share)

    idx = np.argsort(shared_scores)[::-1]
    sorted_pop = [normalized_pop[i] for i in idx]
    sorted_scores = [scores[i] for i in idx]  # trả điểm gốc để giữ chọn cá thể tốt thật
    return sorted_scores, sorted_pop

def selection(population, scores, elite_k=3, tourn_size=3):
    next_pop = population[:elite_k]  # Giữ top-k
    while len(next_pop) < len(population):
        candidates = random.sample(list(zip(population, scores)), tourn_size)
        winner = max(candidates, key=lambda x: x[1])[0]
        next_pop.append(winner.copy())
    return next_pop


def crossover(population, prob):
    """
    Lai ghép (crossover) với xác suất prob.
    Blend crossover với alpha ngẫu nhiên [0.4, 0.6].
    """
    next_pop = [population[0]]  # Giữ cá thể tốt nhất
    i = 1
    while i < len(population):
        if random.random() < prob and i + 1 < len(population):
            p1, p2 = population[i], population[i + 1]
            alpha = random.uniform(0.4, 0.6)
            c1 = alpha * p1 + (1 - alpha) * p2
            c2 = alpha * p2 + (1 - alpha) * p1
            c1 /= c1.sum()
            c2 /= c2.sum()
            next_pop += [c1, c2]
            i += 2
        else:
            next_pop.append(population[i])
            i += 1
    return next_pop[:len(population)]

def adaptive_mutation(prob, gen, max_gen, min_prob=0.05):
    """
    Xác suất đột biến giảm tuyến tính theo số thế hệ.
    """
    return max(min_prob, prob * (1 - gen / max_gen))

def mutation(population, prob, scale=0.1):
    next_pop = [population[0]]  # Giữ cá thể tốt nhất
    for ind in population[1:]:
        mutant = ind.copy()
        for j in range(len(mutant)):
            if random.random() < prob:
                std = scale * mutant[j]  # Tỷ lệ theo giá trị
                mutant[j] += np.random.normal(0, std)
        mutant = np.clip(mutant, 1e-6, None)
        mutant /= mutant.sum()
        next_pop.append(mutant)
    return next_pop


def GA_weighted(
    meta_X_train, y_train,
    meta_X_val, y_val,
    pop_size=15, generations=10,
    crossover_prob=0.9, mutation_prob=0.1,
    mutation_scale=0.1,  # <-- thêm dòng này
    metric='precision', verbose=True,
    init_weights=None,
    sigma_share=0.2
):
    """
    Thuật toán GA tối ưu trọng số theo metric cho stacking.
    Áp dụng niching (fitness sharing) để giữ đa dạng.
    """
    n_models = meta_X_train.shape[1]
    if init_weights is not None:
        init_weights = np.clip(init_weights, 1e-6, None)
        init_weights /= init_weights.sum()

    population = initialize_population(pop_size, n_models, init_weights)
    best_scores = []
    best_vector = None
    best_score = -np.inf

    for gen in range(generations):
        scores, population = evaluate_with_niching(population, meta_X_val, y_val, metric, sigma_share=sigma_share)

        # Luôn giữ cá thể tốt nhất
        if scores[0] > best_score:
            best_score = scores[0]
            best_vector = population[0].copy()

        best_scores.append(best_score)
        if verbose:
            print(f'Gen {gen+1}/{generations} - best {metric}: {scores[0]:.4f}')

        population = selection(population, scores, elite_k=3)
        population = crossover(population, crossover_prob)
        mut_p = adaptive_mutation(mutation_prob, gen, generations)
        population = mutation(population, mut_p, mutation_scale)

        # Kiểm tra không để tệ đi
        new_scores, _ = evaluate_with_niching(population, meta_X_val, y_val, metric, sigma_share=sigma_share)
        if new_scores[0] < best_score:
            population[0] = best_vector.copy()

    return best_vector, best_scores