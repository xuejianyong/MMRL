import pandas as pd
import numpy as np

actions = [0,1,2,3,4]
q_table = pd.DataFrame(columns=actions, dtype=np.float64)
print(q_table)
print()
aa = pd.Series([0]*len(actions),index=q_table.columns,name='aa')
print(aa)
print()
q_table.append(aa)
print(q_table)