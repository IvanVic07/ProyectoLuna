import json
import time

tic = time.time()

def generate_combinations():
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    combinations = []

    # Función para generar las combinaciones
    def create_combinations(prefix, k):
        if k == 0:
            combinations.append(prefix)
            return
        for char in chars:
            create_combinations(prefix + char, k - 1)

    create_combinations('', 4)

    return combinations

def save_combinations_to_json(combinations, filename):
    with open(filename, 'w') as f:
        json.dump(combinations, f)

# Generar combinaciones
combinations = generate_combinations()

# Guardar combinaciones en un archivo JSON
filename = 'combinations.json'
save_combinations_to_json(combinations, filename)

print(f"Combinaciones generadas y guardadas en {filename}")

toc = time.time()
print(toc - tic, 'segundos transcurridos')
