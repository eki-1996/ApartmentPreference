import pandas as pd
from io import StringIO

path = "test.csv"

df = pd.read_csv(path)

posterior_baces = []

for head in list(df.columns):
    if "experiments.1." in head and "posterior" in head:
        posterior_baces.append(df[head])

for posterior_bace in posterior_baces:
    for player_data in posterior_bace:
        if isinstance(player_data, str) and player_data != "":
            all_datas = player_data.split("&")
            for data in all_datas:
                lines = data.split(' ')
                fixed_data = '\n'.join(lines)
                df_reconstructed = pd.read_csv(StringIO(fixed_data))
                print(df_reconstructed)
                # save to csv
                df_reconstructed.to_csv("test_posterior.csv")