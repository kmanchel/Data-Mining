import pandas as pd
import numpy as np

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure 
from bokeh.palettes import Spectral5
from bokeh.transform import dodge
from bokeh.core.properties import value

#(i)(a)
df = pd.read_csv("dating.csv")
df_male = df[df['gender'] == 1]
df_female = df[df['gender'] == 0]


#(ii)(a)
rating_counts = df['attractive_partner'].value_counts().to_frame('counts')
rating_counts.plot.bar(y='counts', rot=0)

#(ii)(b)
decision = df[df['attractive_partner'] == 10]['decision']
decision = decision.value_counts().to_frame('counts_attractive_partner')
decision.plot.pie(title = "when attractive_partner=10",y='counts_attractive_partner',figsize=(5, 5))
print(decision)

df_success = pd.DataFrame(columns=list(df.columns.values)[26:32])
index = 0
scores = np.arange(0,11)
df_success['scores'] = scores
for i in range(26,32):
    success_rates = []
    score = []
    for j in scores:
        success_count = df[(df.iloc[:,i] == j) & (df.iloc[:,52] == 1)]['decision'].count()
        total_count = df[(df.iloc[:,i] == j)]['decision'].count()
        success_rate = success_count/total_count
        success_rates.append(success_rate)
    df_success.iloc[:,index] = success_rates
    index+=1


source_2ii_d = ColumnDataSource(df_success)

p1 = figure(plot_width=400, plot_height=400, title="attractive_partner")
# add a circle renderer with a size, color, and alpha
p1.circle(x = 'scores', y = 'attractive_partner', size=5, color="navy", source = source_2ii_d)
show(p1)

p2 = figure(plot_width=400, plot_height=400, title="sincere_partner")
# add a circle renderer with a size, color, and alpha
p2.circle(x = 'scores', y = 'sincere_partner', size=5, color="navy", source = source_2ii_d)
show(p2)

p3 = figure(plot_width=400, plot_height=400, title="intelligence_parter")
# add a circle renderer with a size, color, and alpha
p3.circle(x = 'scores', y = 'intelligence_parter', size=5, color="navy", source = source_2ii_d)
show(p3)

p4 = figure(plot_width=400, plot_height=400, title="funny_partner")
# add a circle renderer with a size, color, and alpha
p4.circle(x = 'scores', y = 'funny_partner', size=5, color="navy", source = source_2ii_d)
show(p4)

p5 = figure(plot_width=400, plot_height=400, title="ambition_partner")
# add a circle renderer with a size, color, and alpha
p5.circle(x = 'scores', y = 'ambition_partner', size=5, color="navy", source = source_2ii_d)
show(p5)

p6 = figure(plot_width=400, plot_height=400, title="shared_interests_partner")
# add a circle renderer with a size, color, and alpha
p6.circle(x = 'scores', y = 'shared_interests_partner', size=5, color="navy", source = source_2ii_d)
show(p6)