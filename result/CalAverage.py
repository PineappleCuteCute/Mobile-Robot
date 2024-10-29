import pandas as pd
import os

#Change directory if needed
scenario = input("Enter scenario: ")
algorithms = os.listdir(scenario)

result = []
for algorithm in algorithms:
    df = pd.read_csv(scenario + '/' + algorithm, sep=' ', names=['Map', 'Length', 'Smoothness', 'Time', 'Err'])
    df = df.drop(['Time', 'Err'], axis=1)
    df['Map'] = df['Map'].transform(lambda x: x[:-1])
    df.to_excel(scenario + '.xlsx', sheet_name=algorithm)
    df.loc[len(df.index)] = ['Mean', df['Length'].mean(), df['Smoothness'].mean()]
    df.loc[len(df.index)] = ['Std', df['Length'].std(), df['Smoothness'].std()]
    result.append(df)
    # df.to_excel(scenario + '.xlsx', sheet_name=algorithm)

with pd.ExcelWriter(scenario + ".xlsx") as writer:
    for i in range(4):
        result[i].to_excel(writer, sheet_name=algorithms[i])