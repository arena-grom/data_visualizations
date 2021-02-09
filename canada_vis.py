"""
Simple visualization of Canadian immigration data using Matplotlib. 
Created nov 19 2020 by @arena
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
plt.style.use('ggplot')


# Part 1 - Reading data and Visualizing ---------------------------------------

# importing data
xlsx = pd.ExcelFile('Canada.xlsx')
sheet_names = xlsx.sheet_names

# cleaning up
print(f'Parsing sheet "{sheet_names[1]}" and cleaning.')
df = xlsx.parse(sheet_names[1], skiprows=range(0, 20))
print(df.head())
df = df.dropna(how='all', axis=0)
df = df.dropna(how='all', axis=1)
df.drop(columns=['Type', 'Coverage', 'AREA', 
                 'REG', 'DEV', 'DevName'], inplace=True)
df.rename(columns={'OdName': 'Country', 'AreaName': 
                   'Continent', 'RegName': 'Region'}, inplace=True)

# adding an entry for Total
#total = df.sum()
#total.iloc[:4] = 'Total'
#df = df.append(total, ignore_index=True)

# renaming
df.set_index('Country', inplace=True)
print(f'Slice of the cleaned DataFrame:\n{df.iloc[:5, :13]}')

# checking for missing values
print(f'Check if Missing values exist: {df.isnull().values.any()}')


print('Plotting...')

# #### Line plot --------------------------------------------------------------
swiss = df.iloc[df.index == 'Switzerland', 3:]

plt.plot(swiss.T, c='steelblue')
plt.title('Immigration from Switzerland')
plt.xlabel('Years')
plt.ylabel('Number of people')
plt.show()

# #### Multiple lines
df_im = df.loc[['India', 'Pakistan', 'Bangladesh']].drop(columns=['Continent', 'Region'])

plt.plot(df_im.T)
plt.legend(df_im.index)
plt.title('Immigration from South Asian Countries')
plt.xlabel('Years')
plt.ylabel('Number of people')
plt.show()

# #### Overlapping and Stacked Area Plots -------------------------------------
df_im2 = df.loc[['India', 'Pakistan', 'China', 'France']].drop(columns=['Continent', 'Region'])

df_im2.T.plot(kind='area', stacked=False)
plt.legend(loc='upper left')
plt.suptitle('Immigration from India, Pakistan, China and France', fontsize=14)
plt.title('Overlapping area plot', fontsize=10)
plt.xlabel('Years')
plt.ylabel('Number of people')
plt.show()

df_im2.T.plot(kind='area', stacked=True, alpha=0.8)
plt.legend(loc='upper left')
plt.suptitle('Immigration from India, Pakistan, China and France', fontsize=14)
plt.title('Stacked area plot', fontsize=10)
plt.xlabel('Years')
plt.ylabel('Number of people')
plt.show()

# #### Histogram --------------------------------------------------------------
df_im2.T.plot.hist(figsize=(6, 4), bins=15, alpha=0.5)
plt.title('Immigration from India, Pakistan, China and France')
plt.xlabel('Number of people/year')
plt.ylabel('Years')
plt.legend()
plt.show()

# #### Stacked Histogram 
fig = plt.figure(figsize=(6, 4))

plt.hist(df_im2.T.values, bins=15, label=df_im2.index, 
         rwidth=1.5, stacked=True, alpha=0.85)
plt.title('Immigration from India, Pakistan, China and France')
plt.xlabel('Number of people/year')
plt.ylabel('Years')
plt.legend()
plt.show()

# #### Pie chart --------------------------------------------------------------
continent = df.drop(index=['Total']).groupby('Continent').sum()
continent = continent.drop('World')
continent['sum'] = continent[list(continent.columns)].sum(axis=1)

plt.pie(continent['sum'],
        labels=continent.index,
        autopct='%1.0f%%')
plt.axis("image")
plt.title('Immigration per Continent to Canada')
plt.show()

# ### Pie chart with new colors 
plt.pie(continent['sum'],
        labels=continent.index,
        autopct='%1.0f%%',
        colors=['lightgrey', 'lightblue', 'lightgreen', 
                'pink', 'gold', 'c'],
        explode=(0.06, 0, 0, 0, 0.16, 0.11),
        shadow=True,
        startangle=40)

plt.axis("image")
plt.title('Immigration per Continent to Canada')
plt.show()

# #### Boxplots ---------------------------------------------------------------
# 1 box
china = df.iloc[df.index == 'China', 3:]
chi = china.values[0]

fig = plt.figure(figsize=(5, 4))
plt.boxplot(chi, labels=['China'], widths=.25, patch_artist=True,
            boxprops=dict(facecolor='steelblue', alpha=0.8),
            medianprops=dict(color='firebrick'))

plt.title('Immigration from China to Canada')
plt.ylabel('Number of people/year')
plt.show()

# 3 boxes
fig = plt.figure(figsize=(6, 4))
bp = plt.boxplot(df_im.T.values, labels=df_im.index,
                 widths=0.3,
                 patch_artist=True,
                 medianprops=dict(color='firebrick'))

colors = ['mediumpurple', 'darkgrey', 'khaki']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

plt.title('Immigration from South Asian Countries to Canada')
plt.ylabel('Number of people/year')
plt.xlabel('Country')
plt.show()

# #### Scatterplot by Year ----------------------------------------------------
print('Building DataFrame with Year as Index.')
df_year = df.iloc[:, 3:]
df_year.index = df_year.index.rename('year')
df_year = df_year.T
df_year[['Algeria', 'Australia', 'Switzerland', 'India', 'Pakistan']].tail(5)

# Scatterplot with trendline --------------------------------------------------
x = df_year.index
y = df_year.Total

fig = plt.figure(figsize=(6, 4))
plt.scatter(x, y)
z = np.polyfit(x.astype('int64'), y.values, 1)
p = np.poly1d(z)
plb.plot(df_year.index, p(x), "k--")
# plt.annotate('Increasing trend', xy=(100,100))

plt.title('Total Immigrants to Canada 1980-2013')
plt.ylabel('Number of people/year')
plt.xlabel('Year')
plt.show()

# ### Scatterplot changed marks -----------------------------------------------
x = df_year.index
y = df_year.Total

fig = plt.figure(figsize=(6, 4))
plt.scatter(x, y, c='lightblue', marker='o', edgecolor='k')

z = np.polyfit(x.astype('int64'), y.values, 1)
p = np.poly1d(z)
plb.plot(df_year.index, p(x), "--", color='grey')

plt.title('Total Immigrants to Canada 1980-2013')
plt.ylabel('Number of people/year')
plt.xlabel('Year')
plt.show()

# #### Bar plot with trendline ------------------------------------------------
france = df.iloc[df.index == 'France', 3:]
y = france.values[0]
x = np.arange(len(france.values[0]))

fig = plt.figure(figsize=(8, 5))
plt.bar(x, y, color='steelblue', alpha=0.8)

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plb.plot(x, p(x), "k--")
plt.annotate('Increasing trend', xy=(15, 4000))

plt.xticks(x, france.columns, rotation=90)
plt.title('Immigration from France to Canada per Year')
plt.xlabel('Years')
plt.ylabel('Number of people')
plt.show()

# ### Icelandic immigration barplot -------------------------------------------
iceland = df.iloc[df.index == 'Iceland', 3:]
y = iceland.values[0]
x = np.arange(len(iceland.values[0]))

fig = plt.figure(figsize=(8, 4))
plt.bar(x, y, color='skyblue', edgecolor='k')

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plb.plot(x, p(x), '--', color='grey')

plt.annotate('Increasing trend', xy=(21, 27), color='grey')

plt.xticks(x, iceland.columns, rotation=90)
plt.title('Immigration from Iceland to Canada per Year')
plt.xlabel('Years')
plt.ylabel('Number of people')
plt.show()

# #### Horizontal Bar plot with labels ----------------------------------------
print('Sorting Country DataFrame by number of Immigrants')
df_top = pd.DataFrame(df.sum(axis=1), columns=['Immigrants'])
df_top.sort_values(by=['Immigrants'], inplace=True, ascending=False)

tot = df_top.loc['Total'].values[0]
df_top['Percentage_total'] = round(df_top.Immigrants / tot, 4)

# additional mapping for clarity
df_top.rename(index={'United Kingdom of Great Britain and Northern Ireland': 'UK',
                     'United States of America': 'USA',
                     'Iran (Islamic Republic of)': 'Iran',
                     'Republic of Korea': 'South Korea',
                     'Viet Nam': 'Vietnam'}, inplace=True)


def find_top(n, df_top=df_top):
    df_cut = df_top.iloc[:n + 1, :].drop('Total')
    return df_cut

# ### Horizontal Top 10 Barplot -----------------------------------------------
df_top10 = find_top(10)

fig, ax = plt.subplots(figsize=(8, 4))
y = np.arange(len(df_top10))

ax.barh(np.arange(len(df_top10)), df_top10.Immigrants)
# add annotation
for a in ax.patches:
    ax.text(a.get_width() - 80000,
            a.get_y() + .6,
            str(round((a.get_width() / tot) * 100, 2)) + '%',
            fontsize=12,
            color='white')

ax.set_yticks(y)
ax.set_yticklabels(df_top10.index)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Number of people')
ax.set_title('Top 10 Immigration Countries of Canada')
plt.show()

# ### Horizontal Top 15 Barplot -----------------------------------------------
df_top15 = find_top(15)

fig, ax = plt.subplots(figsize=(8, 6))
y = np.arange(len(df_top15))

ax.barh(np.arange(len(df_top15)), df_top15.Immigrants, color='steelblue')
# add annotation
for a in ax.patches:
    ax.text(a.get_width() - 80000,
            a.get_y() + .6,
            str(a.get_width()),
            fontsize=12,
            color='white')

ax.set_yticks(y)
ax.set_yticklabels(df_top15.index)
ax.invert_yaxis()
ax.set_xlabel('Number of people')
ax.set_ylabel('Country')
ax.set_title('Top 15 Immigration Countries of Canada')
plt.show()


# END
