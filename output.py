# # AINA MARTÍ ARANDA
# # 2021 Final Graduate Project
# # Application and development of a CNN model to optimize an OligoFISSEQ image obtention pipeline

import sys
from termcolor import colored, cprint
import pandas as pd

#get the input arguments
filename=sys.argv[1]
path_folder=sys.argv[2]

lst=[]
cprint("you will have to repeat the following models:", "yellow")
with open(filename, "r") as a_file:
  start=False
  for line in a_file: 
    stripped_line = line.strip()
    list_values=stripped_line.split("\t")
    if list_values[0][0]==">":
      list_values[0]=list_values[0][2:]
      if start==True:
        cprint("\t> copies:"+head[0][2:], "blue", end=" ")
        cprint(" fold:"+head[1][-1], "blue", end=" ")
        cprint(" rounds:"+head[2], "blue")
      start=True
      head=list_values

    else:
      if start:
        for i in range(len(list_values)):
          try:
            list_values[i]=float(list_values[i])
          except:
            list_values[i]=list_values[i]

        lst.append(head+list_values)
      start=False

  if start==True:
        cprint("\t> copies:"+list_values[0][2:], "blue", end=" ")
        cprint(" fold:"+list_values[1][-1], "blue", end=" ")
        cprint(" rounds:"+list_values[2], "blue")

df = pd.DataFrame(lst, columns =['copies', 'fold', 'rounds', 'threshold_VAL', 'LOSS_val','AUC_val'])

print("\n\n-----WHOLE TABLE------\n")
print(df)
df.to_csv(path_folder+"results.csv")
cprint("number of models built: "+str(len(df)), "green")

print("\n\n-----SUMMARY------\n")

print(df.groupby(['rounds', 'copies']).mean())
df.groupby(['rounds', 'copies']).mean().to_csv(path_folder+"results_summary.csv")
