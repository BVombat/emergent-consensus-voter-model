# ============================================================
# КОДЫ ЭКСПЕРИМЕНТОВ, ПРИВЕДШИЕ К РЕЗУЛЬТАТАМ
# ============================================================
# Все коды предназначены для выполнения в Google Colab (Python 3.12)
# Необходимые библиотеки: networkx, numpy, pandas, tqdm
# ============================================================

# ------------------------------------------------------------
# 1. Опровержение правила большинства (версия 5.0)
#    Синхронное правило большинства на случайных регулярных графах
#    N=500, k=2..5, 30 графов, 100 начальных состояний, лимит 2000 шагов
# ------------------------------------------------------------
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

N = 500
ks = [2,3,4,5]
n_graphs = 30
n_initial = 100
max_steps = 2000

results_majority = []

for k in ks:
    for graph_idx in tqdm(range(n_graphs), desc=f"Majority k={k}"):
        G = nx.random_regular_graph(k, N, seed=graph_idx)
        adj_list = {node: list(G.neighbors(node)) for node in range(N)}
        for init_seed in range(n_initial):
            np.random.seed(init_seed + graph_idx*1000 + k*10000)
            state = np.random.choice([0,1], size=N)
            step = 0
            while step < max_steps:
                # Проверка консенсуса
                if np.all(state == 0) or np.all(state == 1):
                    break
                # Синхронное правило большинства
                new_state = np.zeros(N, dtype=int)
                for node in range(N):
                    neigh_states = [state[nei] for nei in adj_list[node]]
                    ones = sum(neigh_states)
                    zeros = len(neigh_states) - ones
                    if ones > zeros:
                        new_state[node] = 1
                    elif zeros > ones:
                        new_state[node] = 0
                    else:  # ничья - сохраняем текущее
                        new_state[node] = state[node]
                state = new_state
                step += 1
            consensus = np.all(state == 0) or np.all(state == 1)
            results_majority.append({'k':k, 'graph_id':graph_idx, 'init_seed':init_seed,
                                     'consensus':consensus, 'steps':step})
df_majority = pd.DataFrame(results_majority)
df_majority.to_csv('majority_regular_results.csv', index=False)

# ------------------------------------------------------------
# 2. Асинхронная модель голосования (voter model)
#    N=500, k=5, 30 графов, 100 начальных состояний, лимит 1 млн шагов
#    (потом дорасчёт неудач с 3 млн шагов)
# ------------------------------------------------------------
import random
N = 500
k = 5
n_graphs = 30
n_initial = 100
max_steps = 1_000_000

results_asynch = []

for graph_idx in tqdm(range(n_graphs), desc="Asynch voter"):
    G = nx.random_regular_graph(k, N, seed=graph_idx)
    adj_list = {node: list(G.neighbors(node)) for node in range(N)}
    for init_seed in range(n_initial):
        random.seed(init_seed + graph_idx*1000)
        state = [random.choice([0,1]) for _ in range(N)]
        n0 = state.count(0)
        n1 = N - n0
        step = 0
        while step < max_steps and n0 != 0 and n1 != 0:
            node = random.randrange(N)
            neighbor = random.choice(adj_list[node])
            new_val = state[neighbor]
            old_val = state[node]
            if old_val != new_val:
                state[node] = new_val
                if new_val == 0:
                    n0 += 1
                    n1 -= 1
                else:
                    n0 -= 1
                    n1 += 1
            step += 1
        consensus = (n0 == 0 or n1 == 0)
        results_asynch.append({'graph_id':graph_idx, 'init_seed':init_seed,
                               'consensus':consensus, 'steps':step if consensus else max_steps})
df_asynch = pd.DataFrame(results_asynch)
df_asynch.to_csv('voter_asynch_k5_N500.csv', index=False)

# ------------------------------------------------------------
# 2b. Дорасчёт неудачных запусков асинхронного voter (16 запусков)
#     с увеличенным лимитом 3 млн шагов
# ------------------------------------------------------------
failed = df_asynch[~df_asynch['consensus']].copy()
max_steps_resume = 3_000_000
results_resume = []
for idx, row in failed.iterrows():
    graph_id = int(row['graph_id'])
    init_seed = int(row['init_seed'])
    G = nx.random_regular_graph(5, 500, seed=random.Random(graph_id))
    adj_list = {node: list(G.neighbors(node)) for node in range(500)}
    random.seed(init_seed + graph_id*1000)
    state = [random.choice([0,1]) for _ in range(500)]
    n0 = state.count(0); n1 = 500 - n0
    step = 0
    while step < max_steps_resume and n0 != 0 and n1 != 0:
        node = random.randrange(500)
        neighbor = random.choice(adj_list[node])
        new_val = state[neighbor]
        old_val = state[node]
        if old_val != new_val:
            state[node] = new_val
            if new_val == 0: n0 += 1; n1 -= 1
            else: n0 -= 1; n1 += 1
        step += 1
    consensus = (n0 == 0 or n1 == 0)
    results_resume.append({'graph_id':graph_id, 'init_seed':init_seed,
                           'consensus':consensus, 'steps':step})
df_resume = pd.DataFrame(results_resume)
# Объединение с успешными
df_complete = pd.concat([df_asynch[df_asynch['consensus']], df_resume], ignore_index=True)
df_complete.to_csv('voter_asynch_k5_N500_complete.csv', index=False)

# ------------------------------------------------------------
# 3. Синхронная модель голосования (parallel voter)
#    N=500, k=5, 30 графов, 100 начальных состояний, лимит 1 млн шагов
# ------------------------------------------------------------
N = 500
k = 5
n_graphs = 30
n_initial = 100
max_steps = 1_000_000

results_sync = []

for graph_idx in tqdm(range(n_graphs), desc="Sync voter k=5"):
    G = nx.random_regular_graph(k, N, seed=graph_idx)
    adj_list = [list(G.neighbors(node)) for node in range(N)]
    for init_seed in range(n_initial):
        random.seed(init_seed + graph_idx*1000)
        state = [random.choice([0,1]) for _ in range(N)]
        step = 0
        while step < max_steps:
            n0 = state.count(0)
            n1 = N - n0
            if n0 == 0 or n1 == 0:
                break
            new_state = [0]*N
            for node in range(N):
                neighbor = random.choice(adj_list[node])
                new_state[node] = state[neighbor]
            state = new_state
            step += 1
        consensus = (state.count(0)==0 or state.count(0)==N)
        results_sync.append({'graph_id':graph_idx, 'init_seed':init_seed,
                             'consensus':consensus, 'steps':step})
df_sync = pd.DataFrame(results_sync)
df_sync.to_csv('voter_sync_k5_N500.csv', index=False)

# ------------------------------------------------------------
# 4. Синхронный voter на регулярных графах разных k (3,4,6)
#    N=200, 10 графов, 50 начальных состояний, лимит 200000
# ------------------------------------------------------------
N = 200
n_graphs = 10
n_initial = 50
max_steps = 200_000
results_k = []

for k in [3,4,6]:
    for graph_idx in tqdm(range(n_graphs), desc=f"Sync voter k={k}"):
        G = nx.random_regular_graph(k, N, seed=graph_idx)
        adj_list = [list(G.neighbors(node)) for node in range(N)]
        for init_seed in range(n_initial):
            random.seed(init_seed + graph_idx*1000 + k*10000)
            state = [random.choice([0,1]) for _ in range(N)]
            step = 0
            while step < max_steps:
                n0 = state.count(0); n1 = N - n0
                if n0 == 0 or n1 == 0: break
                new_state = [0]*N
                for node in range(N):
                    neighbor = random.choice(adj_list[node])
                    new_state[node] = state[neighbor]
                state = new_state
                step += 1
            consensus = (state.count(0)==0 or state.count(0)==N)
            results_k.append({'k':k, 'graph_id':graph_idx, 'init_seed':init_seed,
                              'consensus':consensus, 'steps':step})
df_k = pd.DataFrame(results_k)
df_k.to_csv('voter_sync_k3_4_6_N200.csv', index=False)

# ------------------------------------------------------------
# 5. Синхронный voter на неравномерном графе (степени 2 и 6)
#    N=200, 10 графов, 50 начальных состояний
# ------------------------------------------------------------
import random
import networkx as nx
N = 200
n_graphs = 10
n_initial = 50
max_steps = 200_000

results_irr = []
for graph_idx in tqdm(range(n_graphs), desc="Irregular graph"):
    degrees = [2]*(N//2) + [6]*(N//2)
    random.seed(graph_idx)
    random.shuffle(degrees)
    G = nx.random_degree_sequence_graph(degrees, seed=graph_idx, tries=50)
    while G is None or not nx.is_connected(G):
        G = nx.random_degree_sequence_graph(degrees, seed=graph_idx+1000, tries=50)
        if G is not None and nx.is_connected(G): break
        graph_idx += 1
    adj_list = [list(G.neighbors(node)) for node in range(N)]
    for init_seed in range(n_initial):
        random.seed(init_seed + graph_idx*1000)
        state = [random.choice([0,1]) for _ in range(N)]
        step = 0
        while step < max_steps:
            n0 = state.count(0); n1 = N - n0
            if n0 == 0 or n1 == 0: break
            new_state = [0]*N
            for node in range(N):
                neighbor = random.choice(adj_list[node])
                new_state[node] = state[neighbor]
            state = new_state
            step += 1
        consensus = (state.count(0)==0 or state.count(0)==N)
        results_irr.append({'graph_id':graph_idx, 'init_seed':init_seed,
                            'consensus':consensus, 'steps':step})
df_irr = pd.DataFrame(results_irr)
df_irr.to_csv('voter_sync_irregular_deg2_6_N200.csv', index=False)

# ------------------------------------------------------------
# 6. Синхронный voter на графе малого мира (Watts-Strogatz)
#    N=200, средняя степень=4, p=0.1, 10 графов, 50 начальных состояний
# ------------------------------------------------------------
N = 200
n_graphs = 10
n_initial = 50
max_steps = 200_000

results_ws = []
for graph_idx in tqdm(range(n_graphs), desc="Watts-Strogatz"):
    G = nx.watts_strogatz_graph(N, 4, 0.1, seed=graph_idx)
    # Убедимся, что граф связный
    if not nx.is_connected(G):
        G = nx.connected_component_subgraphs(G)[0]  # берём гигантскую компоненту
    adj_list = [list(G.neighbors(node)) for node in range(N)]
    for init_seed in range(n_initial):
        random.seed(init_seed + graph_idx*1000)
        state = [random.choice([0,1]) for _ in range(N)]
        step = 0
        while step < max_steps:
            n0 = state.count(0); n1 = N - n0
            if n0 == 0 or n1 == 0: break
            new_state = [0]*N
            for node in range(N):
                neighbor = random.choice(adj_list[node])
                new_state[node] = state[neighbor]
            state = new_state
            step += 1
        consensus = (state.count(0)==0 or state.count(0)==N)
        results_ws.append({'graph_id':graph_idx, 'init_seed':init_seed,
                           'consensus':consensus, 'steps':step})
df_ws = pd.DataFrame(results_ws)
df_ws.to_csv('voter_sync_ws_N200.csv', index=False)

# ------------------------------------------------------------
# 7. Синхронный voter на безмасштабном графе (Barabási-Albert)
#    N=200, m=2 (средняя степень 4), 10 графов, 50 начальных состояний
# ------------------------------------------------------------
N = 200
n_graphs = 10
n_initial = 50
max_steps = 200_000

results_ba = []
for graph_idx in tqdm(range(n_graphs), desc="Barabasi-Albert"):
    G = nx.barabasi_albert_graph(N, 2, seed=graph_idx)
    adj_list = [list(G.neighbors(node)) for node in range(N)]
    for init_seed in range(n_initial):
        random.seed(init_seed + graph_idx*1000)
        state = [random.choice([0,1]) for _ in range(N)]
        step = 0
        while step < max_steps:
            n0 = state.count(0); n1 = N - n0
            if n0 == 0 or n1 == 0: break
            new_state = [0]*N
            for node in range(N):
                neighbor = random.choice(adj_list[node])
                new_state[node] = state[neighbor]
            state = new_state
            step += 1
        consensus = (state.count(0)==0 or state.count(0)==N)
        results_ba.append({'graph_id':graph_idx, 'init_seed':init_seed,
                           'consensus':consensus, 'steps':step})
df_ba = pd.DataFrame(results_ba)
df_ba.to_csv('voter_sync_ba_N200.csv', index=False)

# ------------------------------------------------------------
# 8. Синхронный voter на кольце (цикл) – двудольный граф
#    N=200, 5 графов (разные seed, топология одна), 100 начальных состояний
# ------------------------------------------------------------
N = 200
n_graphs = 5
n_initial = 100
max_steps = 100_000

results_ring = []
for graph_idx in tqdm(range(n_graphs), desc="Ring graph"):
    G = nx.cycle_graph(N)
    adj_list = [list(G.neighbors(node)) for node in range(N)]
    for init_seed in range(n_initial):
        random.seed(init_seed + graph_idx*1000)
        state = [random.choice([0,1]) for _ in range(N)]
        step = 0
        while step < max_steps:
            n0 = state.count(0); n1 = N - n0
            if n0 == 0 or n1 == 0: break
            new_state = [0]*N
            for node in range(N):
                neighbor = random.choice(adj_list[node])
                new_state[node] = state[neighbor]
            state = new_state
            step += 1
        consensus = (state.count(0)==0 or state.count(0)==N)
        results_ring.append({'graph_id':graph_idx, 'init_seed':init_seed,
                             'consensus':consensus, 'steps':step})
df_ring = pd.DataFrame(results_ring)
df_ring.to_csv('voter_sync_ring_N200.csv', index=False)

# ------------------------------------------------------------
# 9. Проверка связности графов для неудачных запусков (пример)
# ------------------------------------------------------------
import networkx as nx
import pandas as pd
import random

df = pd.read_csv('voter_asynch_k5_N500.csv')
failed = df[~df['consensus']]
failed_graphs = failed['graph_id'].unique()
for gid in failed_graphs:
    G = nx.random_regular_graph(5, 500, seed=random.Random(gid))
    print(f"Graph {gid}: connected = {nx.is_connected(G)}")