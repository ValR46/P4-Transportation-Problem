import pandas as pd
import numpy as np
df = pd.read_excel("opti_value.xlsx", index_col=0)
# example:
# P4 Proj    D1   D2   D3   D4  Supply
# S1         12   13    4    6    500
# S2          6    4   10   11    700
# S3         10    9   12    4    800
# Demand    400   900  200  500
cost_matrix = df.iloc[:-1, :-1].values
supply = df['Supply'][:-1].values
demand = df.loc['Demand'].values[:-1]

num_supply = len(supply)
num_demand = len(demand)

total_supply = sum(supply)
total_demand = sum(demand)
if total_supply != total_demand:
    raise ValueError("Unbalanced Problem: Adjust demands or supplies")

def print_solution(allocation):
    df_alloc = pd.DataFrame(allocation, 
                            index=df.index[:-1], 
                            columns=df.columns[:-1])
    print(df_alloc)


############################################
# 1. Northwest Corner Rule
############################################
def northwest_corner_rule(supply, demand):
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    allocation = np.zeros((num_supply, num_demand), dtype=np.float64)

    i, j = 0, 0
    while i < num_supply and j < num_demand:
        alloc = min(supply_copy[i], demand_copy[j])
        allocation[i, j] = alloc
        supply_copy[i] -= alloc
        demand_copy[j] -= alloc

        if supply_copy[i] == 0:
            i += 1
        elif demand_copy[j] == 0:
            j += 1

    return allocation

nw_allocation = northwest_corner_rule(supply, demand)
print("Initial Feasible Solution using Northwest Corner Rule:")
print_solution(nw_allocation)


############################################
# 2. Minimum Cost Method
############################################
def minimum_cost_method(supply, demand, cost_matrix):
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    allocation = np.zeros((num_supply, num_demand), dtype=np.float64)

    cost_list = []
    for i in range(num_supply):
        for j in range(num_demand):
            cost_list.append((cost_matrix[i, j], i, j))
    cost_list.sort(key=lambda x: x[0])  # sort by cost

    for cost, i, j in cost_list:
        if supply_copy[i] > 0 and demand_copy[j] > 0:
            alloc = min(supply_copy[i], demand_copy[j])
            allocation[i, j] = alloc
            supply_copy[i] -= alloc
            demand_copy[j] -= alloc

    return allocation

min_cost_allocation = minimum_cost_method(supply, demand, cost_matrix)
print("\nInitial Feasible Solution using Minimum Cost Method:")
print_solution(min_cost_allocation)


############################################
# 3. Minimum Row Cost Method
############################################
def minimum_row_cost_method(supply, demand, cost_matrix):
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    allocation = np.zeros((num_supply, num_demand), dtype=np.float64)

    for i in range(num_supply):
        cols_sorted = np.argsort(cost_matrix[i, :])
        for c in cols_sorted:
            if supply_copy[i] == 0:
                break
            if demand_copy[c] > 0:
                alloc = min(supply_copy[i], demand_copy[c])
                allocation[i, c] = alloc
                supply_copy[i] -= alloc
                demand_copy[c] -= alloc

    return allocation

min_row_cost_allocation = minimum_row_cost_method(supply, demand, cost_matrix)
print("\nInitial Feasible Solution using Minimum Row Cost Method:")
print_solution(min_row_cost_allocation)


############################################
# 4. Vogel's Approximation Method (VAM)
############################################
def vogels_approximation_method(supply, demand, cost_matrix):
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    allocation = np.zeros((num_supply, num_demand), dtype=np.float64)

    supply_list = list(supply_copy)
    demand_list = list(demand_copy)

    active_rows = [True]*num_supply
    active_cols = [True]*num_demand

    def calc_penalties():
        row_penalties = []
        col_penalties = []

        for i in range(num_supply):
            if active_rows[i] and supply_list[i] > 0:
                row_costs = [cost_matrix[i, j] for j in range(num_demand) if active_cols[j] and demand_list[j] > 0]
                if len(row_costs) >= 2:
                    sorted_costs = sorted(row_costs)
                    penalty = sorted_costs[1] - sorted_costs[0]
                elif len(row_costs) == 1:
                    penalty = row_costs[0]
                else:
                    penalty = 0
                row_penalties.append((penalty, 'row', i))
        
        for j in range(num_demand):
            if active_cols[j] and demand_list[j] > 0:
                col_costs = [cost_matrix[i, j] for i in range(num_supply) if active_rows[i] and supply_list[i] > 0]
                if len(col_costs) >= 2:
                    sorted_costs = sorted(col_costs)
                    penalty = sorted_costs[1] - sorted_costs[0]
                elif len(col_costs) == 1:
                    penalty = col_costs[0]
                else:
                    penalty = 0
                col_penalties.append((penalty, 'col', j))

        return row_penalties, col_penalties

    while any(s > 0 for s in supply_list) and any(d > 0 for d in demand_list):
        row_penalties, col_penalties = calc_penalties()
        if not row_penalties and not col_penalties:
            break

        combined = row_penalties + col_penalties
        if not combined:
            break
        max_penalty = max(combined, key=lambda x: x[0])

        if max_penalty[1] == 'row':
            i = max_penalty[2]
            active_col_costs = [(cost_matrix[i, j], j) for j in range(num_demand) if active_cols[j] and demand_list[j]>0]
            if not active_col_costs:
                active_rows[i] = False
                continue
            chosen_col = min(active_col_costs, key=lambda x: x[0])[1]
            alloc = min(supply_list[i], demand_list[chosen_col])
            allocation[i, chosen_col] = alloc
            supply_list[i] -= alloc
            demand_list[chosen_col] -= alloc
            if supply_list[i] == 0:
                active_rows[i] = False
            if demand_list[chosen_col] == 0:
                active_cols[chosen_col] = False
        else:
            j = max_penalty[2]
            active_row_costs = [(cost_matrix[i, j], i) for i in range(num_supply) if active_rows[i] and supply_list[i]>0]
            if not active_row_costs:
                active_cols[j] = False
                continue
            chosen_row = min(active_row_costs, key=lambda x: x[0])[1]
            alloc = min(supply_list[chosen_row], demand_list[j])
            allocation[chosen_row, j] = alloc
            supply_list[chosen_row] -= alloc
            demand_list[j] -= alloc
            if supply_list[chosen_row] == 0:
                active_rows[chosen_row] = False
            if demand_list[j] == 0:
                active_cols[j] = False

    return allocation

vam_allocation = vogels_approximation_method(supply, demand, cost_matrix)
print("\nInitial Feasible Solution using Vogel's Approximation Method:")
print_solution(vam_allocation)
