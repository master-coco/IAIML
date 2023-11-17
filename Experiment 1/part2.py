#2. Use Pandas to create dictionary in python.
import pandas as pd
mobiles = [
            ("apple", 60000,150000),
            ("samsung",40000,1000000),
            ("google",45000,90000)
            ]
df = pd.DataFrame(mobiles,columns = ['company','min rate','max rate'])
print(df)
# hello bye
dictionary = df.to_dict(orient='records')
print(dictionary)