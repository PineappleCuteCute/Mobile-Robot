import os
import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

# scenarios = ['dense', 'maze', 'room', 'trap']
scenario = input("Scenario: ")
actions = ['ENV_DECOMPOSITION', 'GLOBAL_PLANNING', 'LOCAL_REPLAN', 'DECISION_MAKING']
algorithms = os.listdir(scenario)

result = []
for algorithm in algorithms:
    df = pd.read_csv(scenario + '/' + algorithm, sep=' ', names=['Map', 'Action', 'Time'])
    index = pd.MultiIndex.from_product([df['Map'].unique(), actions], names=['Map', 'Action'])
    df = df.groupby(['Map', 'Action']).aggregate(['sum', 'mean'])
    df = df.reindex(index, fill_value=0)
    df.to_excel(scenario + '.xlsx', sheet_name=algorithm)
    result.append(df)
    # res.to_excel(writer, sheet_name=algorithm)

with pd.ExcelWriter(scenario + ".xlsx") as writer:
    for i in range(4):
        result[i].to_excel(writer, sheet_name=algorithms[i])