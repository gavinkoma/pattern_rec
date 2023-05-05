import pandas as pd

df = pd.read_csv("/Users/gavinkoma/Desktop/pattern_rec/final/prediction.csv")
df = df.drop(df.columns[0],axis = 1)
df_export = pd.read_csv("/Users/gavinkoma/Desktop/pattern_rec/final/dataframe.csv")

vertical_stack = pd.concat([df_export,df],axis = 1).reindex(df.index)
print(vertical_stack.head(10))

print("unique classes: ", vertical_stack['predictions'].unique())








