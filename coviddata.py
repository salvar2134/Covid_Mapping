#import libraries
import os, sys
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from numpy import int64
import requests, io
import urllib.request
import folium
from folium import plugins
import fiona
import branca
from branca.colormap import linear
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import hvplot.pandas

url = 'https://www.worldometers.info/coronavirus/#countries'
response = requests.get(url)

data = response.content.decode('utf-8')


class HTMLTableParser:

    def parse_url(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        return [(table['id'],self.parse_html_table(table))\
            for table in soup.find_all('table')]

    def parse_html_table(self, table):
        n_columns = 0
        n_rows=0
        column_names = []

        # Find number of rows and columns and column titles if we can
        for row in table.find_all('tr'):
            # Determine the number of rows in the table
            td_tags = row.find_all('td')
            if len(td_tags) > 0:
                n_rows+=1
                if n_columns == 0:
                    # Set the number of columns for our table
                    n_columns = len(td_tags)
            # Handle column names if we find them
            th_tags = row.find_all('th') 
            if len(th_tags) > 0 and len(column_names) == 0:
                for th in th_tags:
                    column_names.append(th.get_text())
        
        # Safeguard on Column Titles
        if len(column_names) > 0 and len(column_names) != n_columns:
            raise Exception("Column titles do not match the number of columns")

        columns = column_names if len(column_names) > 0 else range(0,n_columns)
        df = pd.DataFrame(columns = columns, index= range(0,n_rows))
        row_marker = 0
        for row in table.find_all('tr'):
            column_marker = 0
            columns = row.find_all('td')
            for column in columns:
                df.iat[row_marker,column_marker] = column.get_text()
                column_marker += 1
            if len(columns) > 0:
                row_marker += 1

        # Convert to float if possible
        for col in df:
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass 
        return df 

hp = HTMLTableParser()
table = hp.parse_url(url)[0][1]

# check bottom rows
# print(table.tail(10))
# check top rows
# print(table.header(10))

#Drop top unwanted rows
df = table.drop(table.index[[0,1,2,3,4,5,6,7]]).reset_index(drop=True)
#drop tail unwanted rows
df.drop(df.tail(8).index,inplace=True)
#drop new line '\n' charachter
df.replace(['\n'], '', regex=True, inplace=True)
df.replace([','], '', regex=True, inplace=True)
#drop unwanted special characters using a loop
for col in df.columns[2:15]: 
    df[col] = df[col].str.replace("+", "").str.replace(",", "").str.replace("N/A", "").str.replace(" ", "").str.replace(" ", "")
    df1 = df.rename(columns={'Country,Other': 'CNTRY_NAME',
    'Serious,Critical': 'Serious_Critical', 'Tot Cases/1M pop': 'Tot_Case_1M_pop', 'Deaths/1M pop': 'Deaths_1M_pop',
    'Tests/\n1M pop\n': 'Tests_1M_pop'})

#convert object columns in dataframe to numeric
df1.fillna(0, inplace=True)
df1.replace(np.nan, 0, inplace=True)
df1.replace(np.inf, 0, inplace=True)
for col in df1.columns[2:15]:
    df1[col] = pd.to_numeric(df1[col], errors='ignore')

df1.fillna(0, inplace=True)
df1.replace(np.nan, 0, inplace=True)
df1.replace(np.inf, 0, inplace=True)
for col in df1.columns[2:15]:
    df1[col]=df1[col].apply(int)    

df1.sort_values(by=['TotalCases'], inplace=True, ascending=False)
#df1.head(10)

#get country data
url = "https://opendata.arcgis.com/datasets/a21fdb46d23e4ef896f31475217cbb08_1.geojson"
world = gpd.read_file(url)

world.CNTRY_NAME = world.CNTRY_NAME.replace({"United States", "United Kingdom"}, {"USA", "UK"})
#merge two datasets based on Country Name
corona = world.merge(df1, on='CNTRY_NAME', how='left')

#correct data type
corona.fillna(0, inplace=True)
corona.replace(np.nan, 0, inplace=True)
corona.replace(np.inf, 0, inplace=True)
for col in corona.columns[4:17]:
    corona[col]=corona[col].astype(int)

#new merged geopandas dataframe
df_world = pd.merge(df1, world, on='CNTRY_NAME')
crs = {'init': 'epsg:4326'}
corona_gpd = gpd.GeoDataFrame(df_world, crs=crs, geometry='geometry')

# f, ax = plt.subplots(1,1,figsize=(12,8))
# ax = corona_gpd.plot(column='TotalCases', cmap='rainbow', ax=ax, legend=True, 
#  legend_kwds={'label': 'Total Cases by Country'})
 
# ax = corona_gpd.plot(figsize=(15, 15), column='TotalDeaths', 
# cmap=plt.cm.jet, scheme='fisher_jenks', k=9, alpha=1, legend=True, 
# markersize=0.5)
# plt.title('Coronavirus Total Death by Country')

# covid_interact = corona_gpd.hvplot(c="TotalDeaths", cmap='rainbow',
#     width=800,height=450,
#     title="TotalDeaths by Country")

#print(corona.dtypes)
#world.plot()
#plt.show()
#pd.set_option("display.max_rows", None, "display.max_columns", None)
#print(world_CNTRY_NAME)
#hvplot.show(covid_interact)


def embed_map(m):
    from IPython.display import IFrame
    m.save('index.html')
    return IFrame('index.html', width='100%', height='500px')

tiles='https://services.arcgisonline.com/arcgis/rest/services/World_Topo_Map/MapServer/WMTS/tile/1.0.0/World_Topo_Map/default/default028mm/{z}/{y}/{x}.png'
map = folium.Map([0, 0], zoom_start=2, tiles=tiles, attr='Esri')

gjson = corona_gpd.to_crs(epsg='4326').to_json()
df3 = df1.set_index('CNTRY_NAME')['TotalCases'].dropna()
colorscale = branca.colormap.linear.YlOrRd_09.scale(df1.TotalCases.min(), df1.TotalCases.max())

def style_function(feature):
    TotalCases = df3.get(int(feature['id'][-1:]), None)
    return {
        'fillOpacity': 1,
        'weight': 1,
        'fillColor': '#black' if TotalCases is None else colorscale(TotalCases)
    }

colorscale.add_to(map)
colorscale.caption = 'Total Cases by Country'
country = folium.features.GeoJson(gjson, tooltip=folium.features.GeoJsonTooltip(fields=['CNTRY_NAME','TotalCases']), style_function=style_function)

map.add_child(country)
folium.LayerControl().add_to(map)
# save map as html
results ="C:\\Users\\Steven\\projects\\covid"
map.save(os.path.join(results, 'index.html'))
embed_map(map)