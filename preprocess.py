import pandas as pd
df = pd.read_csv("dating-full.csv")
length = len(df)
processed = []
c = 0
for i in range(0,(length)):
    race = df['race'][i]
    l1 = len(race)
    race = race.strip("'")
    l2 = len(race)
    if l1>l2:
        c += 1
    processed.append(race)
df['race'] = processed
processed = []
for i in range(0,(length)):
    race_o = df['race_o'][i]
    l1 = len(race_o)
    race_o = race_o.strip("'")
    l2 = len(race_o)
    if l1>l2:
        c += 1
    processed.append(race_o)
df['race_o'] = processed
processed = []
for i in range(0,(length)):
    field = df['field'][i]
    l1 = len(field)
    field = field.strip("'")
    l2 = len(field)
    if l1>l2:
        c += 1
    processed.append(field)
df['field'] = processed
processed = []
print("Quotes removed from ",c," cells")
processed = []
c = 0
total = 0
for i in df['field']:
    if i.islower() == False:
        c += 1
    i = i.lower()
    total += 1
    i
    processed.append(i)
print("Standardized ",c," cells to lower case.")
df['field'] = processed
def sort_encode(field):
    field = field.astype('category')
    field_unique = list(field.unique())
    field_sorted = sorted(field_unique)
    encoding_sch = {field_unique: i for i, field_unique in enumerate(field_sorted)}
    return encoding_sch
print("Value assigned for male in column gender: ",sort_encode(df["gender"])['male'])
print("Value assigned for European/Caucasian-American in column race: ",sort_encode(df["race"])['European/Caucasian-American'])
print("Value assigned for Latino/Hispanic American in column race_o: ",sort_encode(df["race_o"])['Latino/Hispanic American'])
print("Value assigned to law in column field: ",sort_encode(df["field"])['law'])

encode_all = {"gender": sort_encode(df["gender"]), "race": sort_encode(df["race"]), 
              "race_o": sort_encode(df["race_o"]), "field": sort_encode(df["field"])}
df.replace(encode_all, inplace=True)
df2 = df.copy()
def std_importance():
    df2['pref_o_total'] = df2['pref_o_attractive']+df2['pref_o_sincere']+df2['pref_o_intelligence']+df2['pref_o_funny']
    +df2['pref_o_ambitious']+df2['pref_o_shared_interests']
    df2['pref_o_attractive'] = df2['pref_o_attractive']/df2['pref_o_total']
    df2['pref_o_sincere'] = df2['pref_o_sincere']/df2['pref_o_total']
    df2['pref_o_intelligence'] = df2['pref_o_intelligence']/df2['pref_o_total']
    df2['pref_o_funny'] = df2['pref_o_funny']/df2['pref_o_total']
    df2['pref_o_ambitious'] = df2['pref_o_ambitious']/df2['pref_o_total']
    df2['pref_o_shared_interests'] = df2['pref_o_shared_interests']/df2['pref_o_total']
    
    df2['important_total'] = df2['attractive_important']+df2['sincere_important']+df2['intelligence_important']+df2['funny_important']
    +df2['ambition_important']+df2['shared_interests_important']
    df2['attractive_important'] = df2['attractive_important']/df2['important_total']
    df2['sincere_important'] = df2['sincere_important']/df2['important_total']
    df2['intelligence_important'] = df2['intelligence_important']/df2['important_total']
    df2['funny_important'] = df2['funny_important']/df2['important_total']
    df2['ambition_important'] = df2['ambition_important']/df2['important_total']
    df2['shared_interests_important'] = df2['shared_interests_important']/df2['important_total']
    
    for i in range(9,21):
        print("Mean of ",list(df2.columns.values)[i],": ", round(df2.iloc[:, i].mean(),2))
        
std_importance()
df.iloc[:,9:21] = df2.iloc[:,9:21]
df.to_csv('dating.csv', index=False)