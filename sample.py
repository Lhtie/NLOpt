import gurobipy as gp
import numpy as np
from scipy import stats
import json

with open('../dataset/nlp4lp_oct_23_2023/model_building_in_mathematical_programming/problem_7/data.json') as f:
    data = json.load(f)

inputone = np.array(data['inputone'])
manpowerone = np.array(data['manpowerone'])
inputtwo = np.array(data['inputtwo'])
manpowertwo = np.array(data['manpowertwo'])
stock = np.array(data['stock'])
capacity = np.array(data['capacity'])
manpower_limit = data['manpower_limit']
K = len(inputone)
T = 3

model = gp.Model()

produce = np.empty((K, T), dtype=object)
buildcapa = np.empty((K, T), dtype=object)
stockhold = np.empty((K, T), dtype=object)
for k in range(K):
    for t in range(T):
        # amount of units produced by industry k in year t
        produce[k, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name="produce_{}_{}".format(k, t))
        # amount of units used to build productive capacity for industry k in year t
        buildcapa[k, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name="buildcapa_{}_{}".format(k, t))
        # amount of stock of industry k held in year t
        stockhold[k, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name="stockhold_{}_{}".format(k, t))

# set objective
# maximize total production in the last two years
obj = gp.LinExpr()
for k in range(K):
    obj.addTerms(1, produce[k, 2])
    obj.addTerms(1, produce[k, 3])
model.setObjective(obj, gp.GRB.MAXIMIZE)

# set constraints
for k in range(K):
    # production in year t + 1 requires input in year t
    for t in range(1, T):
        lhs = gp.LinExpr()
        lhs.addTerms(1, stockhold[k, t - 1])
        for j in range(K):
            lhs.addTerms(inputone[k, j], produce[j, t - 1])
        lhs.addTerms(manpowerone[k], model.params.MinProdRelTol)
        model.addConstr(lhs >= produce[k, t])

    # building productive capacity for industry k in year t requires input in year t
    for t in range(1, T):
        lhs = gp.LinExpr()
        for j in range(K):
            lhs.addTerms(inputtwo[k, j], buildcapa[j, t - 1])
        lhs.addTerms(manpowertwo[k], model.params.MinProdRelTol)
        model.addConstr(lhs >= buildcapa[k, t])

    # input from an industry in year t results in a (permanent) increase in productive capacity in year t + 2
    for t in range(1, T - 1):
        model.addConstr(buildcapa[k, t] == capacity[k, t + 2] - capacity[k, t + 1])

    # stockholding
    for t in range(1, T):
        lhs = gp.LinExpr()
        lhs.addTerms(1, stockhold[k, t - 1])
        lhs.addTerms(-1, produce[k, t])
        lhs.addTerms(1, stock[k])
        model.addConstr(lhs <= stockhold[k, t])

# manpower limit
for t in range(1, T):
    manpower = gp.LinExpr()
    for k in range(K):
        manpower.addTerms(manpowerone[k], produce[k, t])
        manpower.addTerms(manpowertwo[k], buildcapa[k, t])
    model.addConstr(manpower <= manpower_limit)

# solve
model.optimize()

# get objective value
obj_val = model.objVal

# print results
for k in range(K):
    print("Industry {}".format(k))
    for t in range(T):
        print("Produce: {}".format(produce[k, t].X))
        print("Buildcapa: {}".format(buildcapa[k, t].X))
        print("Stockhold: {}".format(stockhold[k, t].X))