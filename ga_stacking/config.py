"""
Module: config
---------------
Hyperparameters for GA-stacking pipeline.
"""
POP_SIZE = 15                # Giữ nhỏ để tính toán nhẹ, nhưng đủ đa dạng
GENERATIONS = 10             # Cân bằng giữa chất lượng & thời gian tiến hóa
CROSSOVER_PROB = 0.9         # Crossover mạnh hơn mutation
MUTATION_PROB = 0.1          # Mutation cao hơn để tăng đa dạng
SIGMA_SHARE = 0.2            # Phạm vi chia sẻ fitness (nên giữ khoảng 0.2–0.4)
MUTATION_SCALE = 0.1       # Mức biến đổi gene (dùng trong mutation)
CV_FOLDS = 3
METRIC = 'precision'