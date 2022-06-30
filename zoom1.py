import pandas as pd
import numpy as np

# df = pd.DataFrame([[1,2],[3,4]], index = ['row1','row2'],columns =['col1', 'col2'])
# print(df)

# result = df.at['row1','col2']
# print(result)

# result = df.iat[0,1]
# print(result)

# result = df.loc['row1','col1']
# print(result)

data = np.random.randint(10,size=(10,10))
print(data)
df  = pd.DataFrame(data = data)

