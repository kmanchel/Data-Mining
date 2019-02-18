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

#(i)(b)
male_means = []
female_means = []
attributes = list(df.columns.values)[15:21]
for i in range(15,21):
        male_mean = df_male.iloc[:, i].mean()
        female_mean = df_female.iloc[:, i].mean()
        male_means.append(male_mean)
        female_means.append(female_mean)


#(i)(c)
output_file("Q 2(i).html")
data = {'Attributes': attributes,
            'Female Means': female_means, 'Male Means': male_means}

source = ColumnDataSource(data=data)

b = figure(x_range=attributes, title="Comparison", x_axis_label='Attribute',
           y_axis_label="Mean Value", toolbar_location=None, tools="", plot_width=1000, plot_height=300)
b.vbar(x=dodge('Attributes', -0.225, range=b.x_range), top="Male Means", width=0.35, source=source, color="blue", legend=value("Male"))
b.vbar(x=dodge('Attributes', +0.225, range=b.x_range), top="Female Means", width=0.35, source=source, color="red", legend=value("Female"))
b.x_range.range_padding = 0.1
b.xgrid.grid_line_color = None
b.legend.location = "top_right"
b.legend.orientation = "vertical"
show(b)