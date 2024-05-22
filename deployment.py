import streamlit as st

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from datetime import timedelta
import squarify
import warnings
import matplotlib.cm as cm
import plotly.express as px
import plotly.graph_objects as go
import cufflinks as cf
import io

#
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score , silhouette_samples
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.stats import norm
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from colorama import Fore, Style
from termcolor import colored
import datetime as dt
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from yellowbrick.cluster import InterclusterDistance
from streamlit_option_menu import option_menu


# hierarchical Clustering libraries
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

# Importing plotly and cufflinks in offline mode
import plotly.offline
import plotly.graph_objects as go
import plotly.express as px

# Setting Configurations:
pd.set_option("display.max_rows", None,"display.max_columns", None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Import Warnings:

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
plt.style.use('seaborn')

# Navigation
clust = st.sidebar.slider("Pilih jumlah cluster : ", 2,9,3,1)

data = st.sidebar.selectbox(
   "Data",
   ("Isi Dataset","Data Pelanggan"),
   index=None,
   placeholder="Select...",
)
clus = st.sidebar.selectbox(
   "Evaluation N Cluster",
   ("Elbow Method","Silhouette Coefficient"),
   index=None,
   placeholder="Select...",
)
rfm = st.sidebar.selectbox(
   "Visualization of RFM Segments",
   ("Pie Chart","Histogram","Treemap","Barplot","Line Plot Avg","Treemap Squarify","Word Cloud","Scatter Matrix Pairplot","Scatter 3d Plot"),
   index=None,
   placeholder="Select...",
)
kmean = st.sidebar.selectbox(
   "Visualization of K-Means Segments",
   ("Intercluster Distance Map", "Pie Chart"),
   index=None,
   placeholder="Select...",
)
output = st.sidebar.selectbox(
   "Karakteristik Pelanggan dan Strategi Pemasaran",
   ("RFM Segments","K-Means Segments"),
   index=None,
   placeholder="Select...",
)
output2 = st.sidebar.selectbox(
   "Hasil Segmentasi Pelanggan",
   ("RFM Segments","K-Means Segments"),
   index=None,
   placeholder="Select...",
)

## Read dataset
df = pd.read_csv('dataset-inisialisasi.csv')
df['ID'] = df['ID'].astype('str')
df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)


st.title("Customer Segmentation Kaosdisablon")

#data
df2 = pd.read_csv("daftar-pelanggan.csv")
df2['ID'] = df2['ID'].astype('str')


if data == "Data Pelanggan" :
    st.subheader("Data Pelanggan")
    st.write(df2)

if data == "Isi Dataset" :
    st.subheader("Isi Dataset")
    st.write(df)

#data prepocessing
df['Amount'] = df['Quantity'] * df['UnitPrice']
# Calculating monetary attribute
cus_data = df.groupby('ID')[['Amount']].sum() # Total Amount spent
cus_data.rename(columns={'Amount' : 'Monetary'},inplace=True)
# Calculating frequency attribute
cus_data['Frequency'] = df.groupby('ID')['Invoice'].count()
# Calculating recency attribute
max_date = max(df['Date'])
df['diff'] = max_date - df['Date']
cus_data['Recency'] = df.groupby('ID')['diff'].min().dt.days
cus_data = cus_data.reset_index()
hasil = cus_data.drop(['ID'],axis=1)

## Feature Scaling
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(hasil)
mmhasil = pd.DataFrame(x_scaled)
mmhasil.columns = ['Monetary', 'Frequency', 'Recency']

## Treating Outlier
h_cap = 0.95
h_cap_val = cus_data['Monetary'].quantile(h_cap)
cus_data['Monetary'][cus_data['Monetary'] > h_cap_val] = h_cap_val
l_cap = 0.05
l_cap_val = cus_data['Monetary'].quantile(l_cap)
cus_data['Monetary'][cus_data['Monetary'] < l_cap_val] = l_cap_val

h_cap = 0.95
h_cap_val = cus_data['Monetary'].quantile(h_cap)
l_cap = 0.05
l_cap_val = cus_data['Monetary'].quantile(l_cap)

cus_data['Monetary'] = cus_data['Monetary'].clip(lower=l_cap_val, upper=h_cap_val)

##PCA
preprocessor = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("pca", PCA(n_components=2, random_state=42)),
        ]
    )
X = cus_data.drop('ID',axis=1)
X_scaled = pd.DataFrame(preprocessor.fit_transform(X),columns=['PC_1','PC_2'])



#elbow method
ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(X_scaled)
    ssd.append(kmeans.inertia_)

# Plot Elbow Curve
fig, ax = plt.subplots()
ax.plot(range_n_clusters, ssd)
ax.set_xlabel('Number of clusters (k)')
ax.set_ylabel('SSD')
ax.set_title('Elbow Curve')

if clus == "Elbow Method" :
    st.subheader("Elbow Method")
    st.pyplot(fig)
    st.write("Penurunan nilai SSD (Sum of Squared Distances) akan menunjukkan seberapa baik data dapat dijelaskan oleh jumlah cluster tertentu. Pada titik penurunan nilai SSD yang mulai melambat secara signifikan atau menyerupai bentuk (siku) atau (elbow), maka itulah jumlah cluster yang sering dipilih sebagai jumlah cluster yang optimal. **Hasil dari plot Elbow Method jumlah cluster yang paling optimal adalah 3.**")

##Silhoutte Coefficient
from sklearn.metrics import silhouette_score
# Calculate silhouette scores
sse_ = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k).fit(X_scaled)
    silhouette = silhouette_score(X_scaled, kmeans.labels_)
    sse_.append([k, silhouette])

# Create DataFrame from silhouette scores
df_silhouette = pd.DataFrame(sse_, columns=['Number of Clusters', 'Silhouette Score'])

# Create a Matplotlib figure and axis
fig, ax = plt.subplots()
ax.plot(df_silhouette['Number of Clusters'], df_silhouette['Silhouette Score'])
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Silhouette Score')

if clus == "Silhouette Coefficient" :
    st.subheader("Silhouette Coefficient")
    st.pyplot(fig)
    st.write("**Menunjukkan bahwa nillai yang paling mendekati 1 yaitu dengan jumlah cluster 3.** Tingginya nilai Silhouette Coefficient mengindikasikan bahwa data yang diuji tercluster dengan baik, yaitu memiliki jarak yang besar atau jauh antar satu cluster ke cluster lain, dan jarak yang rendah atau dekat antar objek dalam suatu cluster yang sama.")


## RFM Model
# quartil
quantiles = cus_data.quantile(q=[0.20,0.40,0.60,0.80])
# scoring
# Argumen (x = nilai, p = recency, monetary, frequency, d = quartiles dict)
def RScore(x,p,d):
    if x <= d[p][0.20]:
        return 1
    elif x <= d[p][0.40]:
        return 2
    elif x <= d[p][0.60]: 
        return 3
    elif x <= d[p][0.80]:
        return 4
    else:
        return 5
    
# Argumen (x = nilai, p = recency, monetary, frequency, k = quartiles dict)
def FMScore(x,p,d):
    if x <= d[p][0.20]:
        return 1
    elif x <= d[p][0.40]:
        return 2
    elif x <= d[p][0.60]: 
        return 3
    elif x <= d[p][0.80]:
        return 4
    else:
        return 5
# input to atribut
cus_data['Recency_Score'] = cus_data['Recency'].apply(RScore, args=('Recency',quantiles,))
cus_data['Frequency_Score'] = cus_data['Frequency'].apply(FMScore, args=('Frequency',quantiles,))
cus_data['Monetary_Score'] = cus_data['Monetary'].apply(FMScore, args=('Monetary',quantiles,))
# input new atribut
cus_data['RFM Score'] = (cus_data['Recency_Score'].astype(str) +
                    cus_data['Frequency_Score'].astype(str) + cus_data['Monetary_Score'].astype(str))
# sum rfm
cus_data['RFM_Sum'] = cus_data[['Recency_Score','Frequency_Score','Monetary_Score']].sum(axis=1)
# segmentasi
segments = {'Customer_Segment' : [ 'Lost customers',
                                    'Hibernating customers',
                                    'Cannot Lose Them',
                                    'At Risk',
                                    'About To Sleep',
                                    'Need Attention',
                                    'Promising',
                                    'New Customers',
                                    'Potential Loyalist',
                                    'Loyal',
                                    'Champions'],
            'RFM' : ['(1)-(1)-(1)|(1)-(1)-(2)|(1)-(2)-(1)|(1)-(3)-(1)|(1)-(4)-(1)|(1)-(5)-(1)',
                     '(3)-(3)-(2)|(3)-(2)-(2)|(2)-(3)-(3)|(2)-(3)-(2)|(2)-(2)-(3)|(2)-(2)-(2)|(1)-(3)-(2)|(1)-(2)-(3)|(1)-(2)-(2)|(2)-(1)-(2)|(2)-(1)-(1)',
                     '(1)-(5)-(5)|(1)-(5)-(4)|(1)-(4)-(4)|(2)-(1)-(4)|(2)-(1)-(5)|(1)-(1)-(5)|(1)-(1)-(4)|(1)-(1)-(3)',
                     '(2)-(5)-(5)|(2)-(5)-(4)|(2)-(4)-(5)|(2)-(4)-(4)|(2)-(5)-(3)|(2)-(5)-(2)|(2)-(4)-(3)|(2)-(4)-(2)|(2)-(3)-(5)|(2)-(3)-(4)|(2)-(2)-(5)|(2)-(2)-(4)|(1)-(5)-(3)|(1)-(5)-(2)|(1)-(4)-(5)|(1)-(4)-(3)|(1)-(4)-(2)|(1)-(3)-(5)|(1)-(3)-(4)|(1)-(3)-(3)|(1)-(2)-(5)|(1)-(2)-(4)',
                     '(3)-(3)-(1)|(3)-(2)-(1)|(3)-(1)-(2)|(2)-(2)-(1)|(2)-(1)-(3)|(2)-(3)-(1)|(2)-(4)-(1)|(2)-(5)-(1)',
                     '(5)-(3)-(5)|(5)-(3)-(4)|(4)-(4)-(3)|(4)-(3)-(4)|(3)-(4)-(3)|(3)-(3)-(4)|(3)-(2)-(5)|(3)-(2)-(4)',
                     '(5)-(2)-(5)|(5)-(2)-(4)|(5)-(2)-(3)|(5)-(2)-(2)|(5)-(2)-(1)|(5)-(1)-(5)|(5)-(1)-(4)|(5)-(1)-(3)|(4)-(2)-(5)|(4)-(2)-(4)|(4)-(1)-(3)|(4)-(1)-(4)|(4)-(1)-(5)|(3)-(1)-(5)|(3)-(1)-(4)|(3)-(1)-(3)',
                     '(5)-(1)-(2)|(5)-(1)-(1)|(4)-(2)-(2)|(4)-(2)-(1)|(4)-(1)-(2)|(4)-(1)-(1)|(3)-(1)-(1)',
                     '(5)-(5)-(3)|(5)-(5)-(1)|(5)-(5)-(2)|(5)-(4)-(1)|(5)-(4)-(2)|(5)-(3)-(3)|(5)-(3)-(2)|(5)-(3)-(1)|(4)-(5)-(2)|(4)-(5)-(1)|(4)-(4)-(2)|(4)-(4)-(1)|(4)-(3)-(1)|(4)-(5)-(3)|(4)-(3)-(3)|(4)-(3)-(2)|(4)-(2)-(3)|(3)-(5)-(3)|(3)-(5)-(2)|(3)-(5)-(1)|(3)-(4)-(2)|(3)-(4)-(1)|(3)-(3)-(3)|(3)-(2)-(3)',
                     '(5)-(4)-(3)|(4)-(4)-(4)|(4)-(3)-(5)|(3)-(5)-(5)|(3)-(5)-(4)|(3)-(4)-(5)|(3)-(4)-(4)|(3)-(3)-(5)',
                     '(5)-(5)-(5)|(5)-(5)-(4)|(5)-(4)-(4)|(5)-(4)-(5)|(4)-(5)-(4)|(4)-(5)-(5)|(4)-(4)-(5)']}
def categorizer(rfm):
    
    if (rfm[:3] in ['111', '112', '121', '131', '141', '151']) :
        rfm = 'Lost Customers'
        
    elif (rfm[:3] in ['332', '322', '233', '232', '223', '222', '132', '123', '122', '212', '212', '211']) :
        rfm = 'Hibernating Customers'
        
    elif (rfm[:3] in ['155', '154', '144', '214', '215','115','114','113']) :
        rfm = 'Cannot Lose Them'
    
    elif (rfm[:3] in ['255','254','245','244','253','252','243','242','235','234','225','224','153','152','145','143','142','135','134','133','125','124']) :
        rfm = 'At Risk'
    
    elif (rfm[:3] in ['331','321','312','221','213','231','241','251']) :
        rfm = 'About To Sleep'
    
    elif (rfm[:3] in ['535', '534','443','434','343','334','325','324']) :
        rfm = 'Need Attention'    
   
    elif (rfm[:3] in ['525', '524','523','522','521','515','514','513','425','424','413','414','415','315','314','313']) :
        rfm = 'Promissing'
    
    elif (rfm[:3] in ['512','511','422','421','412','411','311']) :
        rfm = 'New Customers'
                
    elif (rfm[:3] in ['553','551','552','541','542','533','532','531','452','451','442','441','431','453','433','432','423','353','352','351','342','341','333','323']) :
        rfm = 'Potential Loyalist'
    
    elif (rfm[:3] in ['543','444','435','355','354','345','344','335']) :
        rfm = 'Loyal'

    elif (rfm[:3] in ['555','554','544','545','454','455','445']) :
        rfm = 'Campions'

    return rfm  
# input to new atribut
cus_data['Customer_Category'] = cus_data["RFM Score"].apply(categorizer)
# ratarata cust category
cust_cat_agg = cus_data.groupby('Customer_Category').agg({
    'Recency' : 'mean',
    'Frequency' : 'mean',
    'Monetary' : ['mean','count']
}).round(1)
# size avg cust category
Avg_RFM_Sum = cus_data.groupby('Customer_Category').RFM_Sum.mean()
Size_RFM_Sum = cus_data['Customer_Category'].value_counts()
df_customer_segmentation = pd.concat([Avg_RFM_Sum, Size_RFM_Sum], axis=1).rename(columns={'RFM_Sum':'Avg_RFM_Sum', 'Customer_Category':'Size_RFM_Sum'})

#distribution plot
fig = px.pie(df, 
             values = cus_data['Customer_Category'].value_counts(), 
             names = (cus_data["Customer_Category"].value_counts()).index, 
             title = 'Customer Category Distribution')

if rfm == "Pie Chart" :
    st.subheader("Pie Chart Distribution of RFM Segements")
    st.write("Total Pelanggan 663 Pelanggan.")
    st.plotly_chart(fig)
# plot histogram size label
fig = px.histogram(cus_data, 
                   x = cus_data['Customer_Category'].value_counts().index, 
                   y = cus_data['Customer_Category'].value_counts().values, 
                   title = 'The Size of RFM Label',
                   labels = dict(x = "Customer Segments (Categories)", y ="RFM Label Mean Values"))

if rfm == "Histogram" :
    st.subheader("Histogram Visualization of RFM Segements")
    st.write("Total Pelanggan 663 Pelanggan.")
    # Menampilkan plot di Streamlit
    st.plotly_chart(fig)

# visualisasi treemap
fig = px.treemap(df_customer_segmentation, 
                 path=[df_customer_segmentation.index], 
                 values='Size_RFM_Sum', 
                 width=950, height=600)

fig.update_layout(title_text='Customer Segmentation',
                  title_x=0.5, title_font=dict(size=20)
                  )
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))

if rfm == "Treemap" :
    st.subheader("Treemap Visualization of RFM Segements")
    st.write("Total Pelanggan 663 Pelanggan.")
    st.plotly_chart(fig)

## Create plot and resize
segmentation = pd.DataFrame(cus_data.Customer_Category.value_counts(dropna=False).sort_values(ascending=False))
segmentation.reset_index(inplace=True)
segmentation.rename(columns={'index':'Customer Category', 'Customer_Category':'The Number Of Customer'}, inplace=True)
#barplot
# Create a figure and axis explicitly
fig, ax = plt.subplots(figsize=(19, 8))

# Plot the bar plot
sns.barplot(data=segmentation, x='Customer Category', y='The Number Of Customer', palette='Oranges_r', ax=ax)

if rfm == "Barplot" :
    st.subheader("Bar Plot Visualization of RFM Segements")
    st.write("Total Pelanggan 663 Pelanggan.")
    st.pyplot(fig)

#line plot
# Create a figure and axis explicitly
fig, ax = plt.subplots(figsize=(14, 7))

# Plot the line plot
sns.lineplot(x=df_customer_segmentation.index, y=df_customer_segmentation.Avg_RFM_Sum, ax=ax)
ax.set_xticklabels(df_customer_segmentation.index, rotation=30, fontsize=14)

if rfm == "Line Plot Avg" :
    st.subheader("Line Plot Visualization of RFM Segements")
    st.pyplot(fig)

#treemap squarify
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(13, 8)
plt.clf()
squarify.plot(sizes=segmentation['The Number Of Customer'], 
                      label=['Lost customers',
                              'Hibernating customers',
                              'Cannot Lose Them',
                              'At Risk',
                              'About To Sleep',
                              'Need Attention',
                              'Promising',
                              'New Customers',
                              'Potential Loyalist',
                              'Loyal',
                              'Champions'], 
                            alpha=0.8, 
                            color=["red", "#48BCF5", "#DD6AE1", "blue", "cyan", "magenta", '#B20CB7', "#A4E919"])
plt.title("RFM Segments", fontsize=18, fontweight="bold")
plt.axis('off')

# Menyimpan gambar ke dalam BytesIO
buffer_squarify = io.BytesIO()
plt.savefig(buffer_squarify, format='png')
buffer_squarify.seek(0)

if rfm == "Treemap Squarify" :
    st.subheader("Treemap Squarify Visualization of RFM Segements")
    # Menampilkan gambar di Streamlit
    st.image(buffer_squarify)

#word cloud
segment_text = segmentation["Customer Category"].str.split(" ").str.join("_")
all_segments = " ".join(segment_text)

wc = WordCloud(background_color="orange", 
               #max_words=250, 
               max_font_size=256, 
               random_state=42,
               width=800, height=400)
wc.generate(all_segments)
plt.figure(figsize = (16, 15))
plt.imshow(wc)
plt.title("RFM Segments", fontsize=18, fontweight="bold")
plt.axis('off')
# Menyimpan gambar WordCloud ke dalam BytesIO
buffer_word = io.BytesIO()
wc.to_image().save(buffer_word, format='PNG')
buffer_word.seek(0)

if rfm == "Word Cloud" :
    st.subheader("Word Cloud Visualization of RFM Segements")
    # Menampilkan gambar di Streamlit
    st.image(buffer_word)

# Pairplot scatter matrix
fig = px.scatter_matrix(cus_data, 
                        dimensions=['Recency', 'Frequency', 'Monetary'], 
                        color="Customer_Category",
                        width=800, height=600)

if rfm == "Scatter Matrix Pairplot" :
    st.subheader("Scatter Matrix Pairplot Visualization")
    # Menampilkan plot di Streamlit
    st.plotly_chart(fig)

# **Handling with Skewness - np.log**
skew_limit = 0.45 # This is our threshold-limit to evaluate skewness. Overall below abs(1) seems acceptable for the linear models. 
skew_vals = cus_data[['Recency', 'Frequency', 'Monetary']].skew()
skew_cols = skew_vals[abs(skew_vals)> skew_limit].sort_values(ascending=False)

cf.go_offline()
rfm_log = cus_data[skew_cols.index].copy()

for col in skew_cols.index.values:
    rfm_log[col] = rfm_log[col].apply(np.log1p)

    print(rfm_log.skew())
print()
# **Handling with Skewness - Power Transformer**
from sklearn.preprocessing import PowerTransformer
rfm_before_trans = cus_data[skew_cols.index].copy()
pt = PowerTransformer(method='yeo-johnson')
trans= pt.fit_transform(rfm_before_trans)
rfm_trans = pd.DataFrame(trans, columns =skew_cols.index )
# Interpreting Skewness 
for skew in rfm_trans.skew():
    if -0.75 < skew < 0.75:
        print ("A skewness value of", '\033[1m', Fore.GREEN, skew, '\033[0m', "means that the distribution is approx.", '\033[1m', Fore.GREEN, "symmetric", '\033[0m')
    elif  -0.75 < skew < -1.0 or 0.75 < skew < 1.0:
        print ("A skewness value of", '\033[1m', Fore.YELLOW, skew, '\033[0m', "means that the distribution is approx.", '\033[1m', Fore.YELLOW, "moderately skewed", '\033[0m')
    else:
        print ("A skewness value of", '\033[1m', Fore.RED, skew, '\033[0m', "means that the distribution is approx.", '\033[1m', Fore.RED, "highly skewed", '\033[0m')
#scatter 3d
fig = px.scatter_3d(rfm_trans, 
                    x='Recency',
                    y='Frequency',
                    z='Monetary',
                    color='Frequency')

if rfm == "Scatter 3d Plot" :
    st.subheader("Scatter 3D Plot")
    # Menampilkan plot di Streamlit
    st.plotly_chart(fig)

## Run KMeans
# prediction was added
kmeans = KMeans(n_clusters = clust).fit(X_scaled)
kmeans.fit_predict(X_scaled)
labels = kmeans.labels_
rfm_trans['ClusterID']=labels



## Plot Distance
plt.figure(figsize=(10, 7))
# Instantiate the clustering model and visualizer
model = KMeans(clust)
visualizer = InterclusterDistance(model)

visualizer.fit(X_scaled)  # Fit the data to the visualizer

# Simpan plot ke dalam BytesIO
buffer_map = io.BytesIO()
visualizer.show(outpath=buffer_map)  # Menampilkan visualisasi dan menyimpannya ke buffer
buffer_map.seek(0)

if kmean == "Intercluster Distance Map" :
    st.subheader("Intercluster Distance Map")
    st.image(buffer_map)
    st.write("**Klaster yang Berdekatan:** Klaster yang dekat satu sama lain dalam grafik menunjukkan adanya kesamaan atau hubungan yang erat dalam fitur atau pola yang ada di dalamnya.")
    st.write("**Klaster yang Terpisah:** Klaster yang jauh satu sama lain menandakan perbedaan atau kelompok yang lebih terpisah dalam dataset.")


#distribution kmeans
fig = px.pie(df, values = rfm_trans['ClusterID'].value_counts(), 
             names = (rfm_trans['ClusterID'].value_counts()).index, 
             title = 'Predicted Clusters Distribution')

if kmean == "Pie Chart" :
    st.subheader("Pie Chart Cluster Segments")
    st.write("Total Pelanggan 663 Pelanggan.")
    st.plotly_chart(fig)

cus_data['ClusterID'] = labels
def generate_cluster_labels(clust):
    if clust == 2:
        return {0: 'Occasional Spenders', 1: 'Frequent Buyers'}
    elif clust == 3:
        return {0: 'Regular Customers', 1: 'Inactive Customer', 2: 'Premium Shoppers'}
    elif clust == 4:
        return {0: 'Moderate Buyers', 1: 'Occasional Customer', 2: 'High-Value Regular Customers', 3: 'Infrequent Low Spenders'}
    # Tambahkan kondisi elif untuk nilai clust lainnya sampai 10
    elif clust == 5:
        return {0: 'High-Value Regular Buyers', 1: 'Occasional Buyers', 2: 'Infrenquent Low Spenders', 3: 'Moderate Buyers', 4: 'High-Value Infrenquent Shoppers'}
    elif clust == 6:
        return {0: 'Low-Spending Infrenquent Shoppers', 1: 'Moderate Spending Regular Shoppers', 2: 'High-Value Frequent Shoppers', 3: 'Moderate Spending Occasional Shoppers', 4: 'High-Spending Regular Shoppers', 5: 'Very High-Value Infrenquent Shoppers'}
    elif clust == 7:
        return {0: 'Regular Moderate Spenders', 1: 'Infrequent High-Spenders', 2: 'High-Value Occasional Buyers', 3: 'Occasional Moderate Spenders', 4: 'Frequent Big Spenders', 5: 'Regular Low Spenders', 6: 'Frequent Big Spenders'}
    elif clust == 8:
        return {0: 'Occasional High-Spenders', 1: 'Moderate Spenders', 2: 'Low Spenders', 3: 'Frequent Big Spenders', 4: 'Occasional Moderate Spenders', 5: 'Big Spenders', 6: 'Regular Spenders', 7: 'Frequent Low Spenders'}
    elif clust == 9:
        return {0: 'Occasional Spenders', 1: 'Infrequent Bargain Shoppers', 2: 'Frequent High Spenders', 3: 'Moderate Consistent Buyers', 4: 'Regular Moderate Spenders', 5: 'Intermittent Budget Shoppers', 6: 'Steady Moderate Buyers', 7: 'Frequent Big Spenders', 8: 'Consistent Low Spenders'}
    else:
        return {i: f'Cluster {i}' for i in range(clust)}

# Contoh penggunaan:
clust_value = clust  # Ganti nilai ini sesuai dengan nilai klaster yang diinginkan (2-10)
cluster_labels = generate_cluster_labels(clust_value)

# Kemudian, tambahkan kolom Labels berdasarkan nilai klaster
rfm_trans['Labels'] = rfm_trans['ClusterID'].map(cluster_labels)

if output2 == "K-Means Segments" :
    st.subheader("Hasil Customer Segmentation Berdasarkan RFM dan KMeans")
    st.write("Dengan Jumlah Cluster ", clust)
    drop = ['Frequency_Score','Monetary_Score']
    hasill = cus_data.drop(columns=['Recency_Score']+drop)
    st.write(hasill)



if output2 == "RFM Segments" :
    st.subheader("Hasil Customer Segmentation Berdasarkan RFM")
    # Mengambil semua kolom kecuali 'Cluster ID'
    drop = ['Recency_Score','Frequency_Score','Monetary_Score']
    satu = cus_data[cus_data['Customer_Category'] == 'Campions'].drop(columns=['ClusterID']+drop)
    st.write("Pelanggan dengan Label RFM Campion")
    st.write(satu)
    drop = ['Recency_Score','Frequency_Score','Monetary_Score']
    dua = cus_data[cus_data['Customer_Category'] == 'Loyal'].drop(columns=['ClusterID']+drop)
    st.write("Pelanggan dengan Label RFM Loyal")
    st.write(dua)
    drop = ['Recency_Score','Frequency_Score','Monetary_Score']
    tiga = cus_data[cus_data['Customer_Category'] == 'Potential Loyalist'].drop(columns=['ClusterID']+drop)
    st.write("Pelanggan dengan Label RFM Potential Loyalist")
    st.write(tiga)
    drop = ['Recency_Score','Frequency_Score','Monetary_Score']
    empat = cus_data[cus_data['Customer_Category'] == 'New Customers'].drop(columns=['ClusterID']+drop)
    st.write("Pelanggan dengan Label RFM New Customers")
    st.write(empat)
    drop = ['Recency_Score','Frequency_Score','Monetary_Score']
    lima = cus_data[cus_data['Customer_Category'] == 'Promissing'].drop(columns=['ClusterID']+drop)
    st.write("Pelanggan dengan Label RFM Promissing")
    st.write(lima)
    drop = ['Recency_Score','Frequency_Score','Monetary_Score']
    enam = cus_data[cus_data['Customer_Category'] == 'Need Attention'].drop(columns=['ClusterID']+drop)
    st.write("Pelanggan dengan Label RFM Need Attention")
    st.write(enam)
    drop = ['Recency_Score','Frequency_Score','Monetary_Score']
    tujuh = cus_data[cus_data['Customer_Category'] == 'About To Sleep'].drop(columns=['ClusterID']+drop)
    st.write("Pelanggan dengan Label RFM About To Sleep")
    st.write(tujuh)
    drop = ['Recency_Score','Frequency_Score','Monetary_Score']
    delapan = cus_data[cus_data['Customer_Category'] == 'Cannot Lose Them'].drop(columns=['ClusterID']+drop)
    st.write("Pelanggan dengan Label RFM Cannot Lose Them")
    st.write(delapan)
    drop = ['Recency_Score','Frequency_Score','Monetary_Score']
    sembilan = cus_data[cus_data['Customer_Category'] == 'At Risk'].drop(columns=['ClusterID']+drop)
    st.write("Pelanggan dengan Label RFM At Risk")
    st.write(sembilan)
    drop = ['Recency_Score','Frequency_Score','Monetary_Score']
    sepuluh = cus_data[cus_data['Customer_Category'] == 'Hibernating Customers'].drop(columns=['ClusterID']+drop)
    st.write("Pelanggan dengan Label RFM Hibernating Customers")
    st.write(sepuluh)
    drop = ['Recency_Score','Frequency_Score','Monetary_Score']
    sebelas = cus_data[cus_data['Customer_Category'] == 'Lost Customers'].drop(columns=['ClusterID']+drop)
    st.write("Pelanggan dengan Label RFM Lost Customers")
    st.write(sebelas)

if output == "RFM Segments" :
    st.subheader("Karakteristik Pelanggan dan Strategi Pemasaran berdasarkan RFM Segments")
    st.write("**Champion**")
    st.write("•	Karakteristik: Baru-baru ini melakukan pembelian, sering melakukan pemesanan, dan menghabiskan jumlah uang tertinggi.")
    st.write("•	Strategi: Memberikan penghargaan kepada mereka karena loyalitasnya. Mereka merupakan calon early adopters produk baru dan kemungkinan besar akan mempromosikan merek Anda. Dapat memberikan referensi kepada orang lain.")
    st.write("**Loyal**")
    st.write("•	Karakteristik: Melakukan pemesanan secara teratur dan responsif terhadap promosi.")
    st.write("•	Strategi: Meningkatkan penjualan dengan menawarkan produk bernilai lebih tinggi kepada mereka. Meminta ulasan atau testimoni dari mereka juga bisa membantu.")
    st.write("**Potential Loyalists**")
    st.write("•	Karakteristik: Pelanggan baru-baru ini yang telah menghabiskan jumlah yang layak.")
    st.write("•	Strategi: Menawarkan program keanggotaan atau program loyalitas untuk mempertahankan keterlibatan mereka. Rekomendasi yang dipersonalisasi dapat meningkatkan keterlibatan mereka.")
    st.write("**New Customers**")
    st.write("•	Karakteristik: Baru saja melakukan pembelian.")
    st.write("•	Strategi: Memberikan dukungan onboarding, memberi mereka akses lebih awal, dan memulai membangun hubungan yang baik dengan mereka.")
    st.write("**Promissing**")
    st.write("•	Karakteristik: Potensial menjadi pelanggan setia beberapa bulan yang lalu. Meskipun sering melakukan pembelian dengan jumlah yang baik, pembelian terakhir mereka sudah beberapa minggu yang lalu.")
    st.write("•	Strategi: Menawarkan kupon atau diskon untuk membawa mereka kembali ke platform dan mempertahankan keterlibatan mereka. Rekomendasi yang dipersonalisasi juga dapat membantu.")
    st.write("**Need Attention**")
    st.write("•	Karakteristik: Pelanggan inti yang pembelian terakhirnya lebih dari satu bulan yang lalu.")
    st.write("•	Strategi: Menawarkan penawaran terbatas atau rekomendasi yang dipersonalisasi untuk mempertahankan hubungan dengan mereka.")
    st.write("**About To Sleep**")
    st.write("•	Karakteristik: Terakhir kali melakukan pembelian jauh sebelumnya, tetapi dalam 4 minggu terakhir mereka telah mengunjungi situs atau membuka pesan.")
    st.write("•	Strategi: Membuat subjek pesan yang sangat dipersonalisasi atau memberikan diskon spesifik pada produk tertentu untuk membangkitkan kembali minat mereka.")
    st.write("**Cannot Lose Them But Losing**")
    st.write("•	Karakteristik: Melakukan pembelian besar dan sering, namun sudah lama tidak kembali.")
    st.write("•	Strategi: Mencoba memenangkan mereka kembali melalui perpanjangan layanan atau produk baru. Fokus pada personalisasi tinggi untuk mempertahankan mereka.")
    st.write("**At Risk**")
    st.write("•	Karakteristik: Mirip dengan 'Cannot Lose Them' namun dengan nilai moneter dan frekuensi yang lebih rendah.")
    st.write("•	Strategi: Menyediakan sumber daya yang membantu di situs web atau mengirim pesan yang dipersonalisasi untuk mempertahankan hubungan dengan mereka.")
    st.write("**Hibernating Customers**")
    st.write("•	Karakteristik: Pelanggan yang sebelumnya melakukan pembelian yang lebih kecil dan jarang, tetapi sudah lama tidak melakukan pembelian.")
    st.write("•	Strategi: Mengikutsertakan mereka dalam komunikasi pesan standar, tetapi memeriksa secara teratur agar konten Anda tidak dianggap sebagai spam. Tidak perlu mengalokasikan sumber daya berlebih di segmen ini.")
    st.write("**Lost Customers**")
    st.write("•	Karakteristik: Melakukan pembelian terakhir sudah lama dan tidak ada interaksi sama sekali dalam 4 minggu terakhir.")
    st.write("•	Strategi: Membangkitkan kembali minat mereka melalui kampanye outreach, namun jika tidak berhasil, lebih baik fokus pada segmen lain yang lebih responsif.")
if output == "K-Means Segments" :
    st.subheader("Karakteristik Pelanggan dan Strategi Pemasaran berdasarkan K-Means Segments")
    st.write("Dengan Jumlah Cluster ", clust)
    st.markdown("---")
    if clust == 2 :
        duaa = """
        \n**Cluster 0: "Occasional Spenders" (Pembeli Sesekali)**
        \n•	Karakteristik Pelanggan:
            Cluster ini memiliki nilai frekuensi yang rendah dan nilai monetary yang relatif rendah, menunjukkan bahwa pelanggan dalam cluster ini cenderung berbelanja secara sporadis atau sesekali.
        \n•	Strategi Pemasaran:
            okus pada meningkatkan keterlibatan dengan menawarkan promosi khusus saat pelanggan melakukan pembelian, memperkuat hubungan dengan komunikasi yang relevan dan bermanfaat, serta menyediakan insentif untuk meningkatkan frekuensi pembelian.
        \n**Cluster 1:  "Frequent Buyers" (Pembeli Rutin)**
        \n•	Karakteristik Pelanggan:
            Cluster ini memiliki nilai frekuensi yang tinggi dan nilai monetary yang tinggi, menunjukkan bahwa pelanggan dalam cluster ini cenderung berbelanja secara teratur dan memberikan kontribusi finansial yang besar bagi perusahaan.
        \n•	Strategi Pemasaran:
            Pertahankan keterlibatan dengan terus memberikan nilai tambah dan pengalaman belanja yang memuaskan, berikan insentif untuk memperkuat loyalitas mereka, kirimkan promosi eksklusif untuk mengapresiasi dukungan mereka
        """
        st.markdown(duaa)
    if clust == 3 :
        tigaa = """
        \n**Cluster 2: "Premium Shoppers"**
        \n•	Karakteristik Pelanggan:
            Memiliki nilai Frequency dan Monetary yang tinggi, menunjukkan bahwa pelanggan dalam cluster ini adalah pembeli yang aktif dan memberikan kontribusi finansial yang besar bagi perusahaan. Mereka mungkin sering membeli produk dengan nilai yang tinggi.
        \n•	Strategi Pemasaran:
            Fokus pada pembangunan loyalitas dengan program khusus, seperti program loyalitas premium dengan insentif tambahan, diskon eksklusif, dan akses ke penawaran khusus.
        \n**Cluster 0: "Regular Customers"**
        \n•	Karakteristik Pelanggan:
            Memiliki tingkat aktivitas belanja yang cukup tinggi dan memberikan kontribusi finansial yang signifikan bagi perusahaan. Mereka cenderung melakukan pembelian secara teratur, meskipun tidak sebanyak "Premium Shoppers".
        \n•	Strategi Pemasaran:
            Pertahankan keterlibatan dengan menawarkan promosi berbasis frekuensi dan program reward untuk memberikan insentif kepada pelanggan yang sering berbelanja. Lakukan kampanye retensi yang bertujuan untuk memastikan bahwa pelanggan merasa dihargai dan ingin terus berbelanja.
        \n**Cluster 1: "Inactive Customers"**
        \n•	Karakteristik Pelanggan:
            Memiliki nilai Recency yang tinggi, menunjukkan bahwa pelanggan dalam cluster ini cenderung tidak melakukan pembelian baru-baru ini. Selain itu, nilai Frequency dan Monetary mereka juga relatif rendah dibandingkan dengan cluster lainnya, menunjukkan bahwa mereka tidak memberikan kontribusi finansial yang signifikan bagi perusahaan. 
        \n•	Strategi Pemasaran:
            Fokus pada reaktivasi dengan kampanye pemasaran khusus yang menawarkan penawaran khusus untuk pembelian berikutnya atau diskon eksklusif untuk pembelian ulang. Lakukan survei kepuasan pelanggan untuk memahami alasan di balik ketidakaktifan mereka dan identifikasi cara untuk meningkatkan pengalaman pembelian.
        """
        st.markdown(tigaa)
    if clust == 4 :
        empatt = """
        \n**Cluster 0: "Moderate Buyers"**
       \n •	Karakteristik Pelanggan:
            Pembelian mereka memiliki nilai tengah, dengan frekuensi yang moderat dan recency yang relatif baru. Mereka cenderung berbelanja secara stabil dengan pengeluaran yang cukup konsisten.
       \n •	Strategi Pemasaran:
            Fokus pada mempertahankan keterlibatan dengan memberikan insentif kepada pelanggan untuk berbelanja lebih sering, seperti program loyalitas atau promosi diskon berdasarkan frekuensi pembelian.
        \n**Cluster 1: "Occasional Buyers"**
       \n •	Karakteristik Pelanggan:
            Pembelian mereka terjadi secara sporadis dengan frekuensi yang rendah dan recency yang beragam. Mereka mungkin tidak sering berbelanja, namun ketika mereka melakukannya, mereka cenderung mengeluarkan jumlah yang lebih tinggi.
       \n •	Strategi Pemasaran:
            Menawarkan promosi khusus atau diskon untuk memancing pelanggan kembali ke platform, serta menyediakan rekomendasi produk yang relevan untuk meningkatkan keterlibatan mereka.
        \n**Cluster 2: "High-Value Regular Customers"**
       \n •	Karakteristik Pelanggan:
            Mereka adalah pelanggan dengan nilai tinggi yang berbelanja secara teratur dengan frekuensi yang tinggi dan recency yang relatif baru. Kontribusi finansial yang signifikan mereka membuat mereka menjadi aset berharga bagi perusahaan.
       \n •	Strategi Pemasaran:
            Fokus pada mempertahankan kepuasan pelanggan dengan memberikan pengalaman yang personal dan eksklusif, serta memperkenalkan program loyalitas atau reward yang eksklusif.
        \n**Cluster 3: "Infrequent Low Spenders"**
       \n •	Karakteristik Pelanggan:
            Mereka adalah pelanggan yang jarang berbelanja dengan frekuensi yang rendah dan recency yang sudah lama. Meskipun pengeluaran mereka relatif kecil, mereka tetap memiliki potensi untuk meningkatkan keterlibatan.
       \n •	Strategi Pemasaran:
            Mencoba untuk meningkatkan frekuensi pembelian dengan menawarkan insentif, seperti diskon spesial untuk pembelian berikutnya atau promosi eksklusif yang hanya tersedia untuk mereka.
        """
        st.markdown(empatt)
    if clust == 5 :
        limaa = """
        \n**Cluster 0: "High-Value Regular Buyers"**
       \n •	Karakteristik Pelanggan:
            Pelanggan dalam cluster ini memiliki tingkat pembelian yang tinggi, frekuensi yang cukup sering, dan recency yang relatif baru. Mereka merupakan pelanggan reguler yang memberikan kontribusi finansial yang signifikan.
       \n •	Strategi Pemasaran:
            Fokus pada mempertahankan keterlibatan dengan memberikan insentif bagi pelanggan untuk tetap berbelanja secara teratur, seperti program loyalitas dengan reward yang menarik atau promosi eksklusif.
        \n**Cluster 1: "Occasional Buyers"**
       \n •	Karakteristik Pelanggan:
            Pelanggan dalam cluster ini berbelanja secara sporadis dengan frekuensi yang rendah dan recency yang beragam. Meskipun demikian, mereka dapat memberikan kontribusi finansial yang cukup.
       \n •	Strategi Pemasaran:
            Menawarkan promosi khusus atau diskon yang dapat menarik pelanggan kembali ke platform, serta menyediakan rekomendasi produk yang sesuai dengan preferensi mereka.
        \n**Cluster 2: "Infrequent Low Spenders"**
       \n •	Karakteristik Pelanggan:
            Pelanggan dalam cluster ini jarang berbelanja dengan frekuensi yang rendah dan recency yang sudah lama. Meskipun pengeluaran mereka relatif kecil, mereka masih memiliki potensi untuk meningkatkan keterlibatan.
       \n •	Strategi Pemasaran:
            Mencoba meningkatkan frekuensi pembelian dengan menawarkan insentif, seperti diskon untuk pembelian berikutnya atau promosi khusus yang hanya ditujukan kepada mereka.
        \n**Cluster 3: "Moderate Buyers"**
       \n •	Karakteristik Pelanggan:
            Pelanggan dalam cluster ini memiliki tingkat pembelian yang moderat, frekuensi yang cukup, dan recency yang relatif baru. Mereka merupakan pembeli yang stabil dengan kontribusi finansial yang cukup signifikan.
       \n •	Strategi Pemasaran:
            Mempertahankan keterlibatan dengan memberikan insentif kepada pelanggan untuk tetap berbelanja secara teratur, serta menyediakan promosi diskon atau reward untuk meningkatkan loyalitas.
        \n**Cluster 4: "High-Value Infrequent Shoppers"**
        \n•	Karakteristik Pelanggan:
            Pelanggan dalam cluster ini memiliki tingkat pembelian yang sangat tinggi, namun frekuensi belanja yang rendah dan recency yang cukup lama. Mereka dapat memberikan kontribusi finansial yang besar, meskipun tidak berbelanja secara teratur.
        \n•	Strategi Pemasaran:
            Fokus pada mempertahankan hubungan dengan memberikan perlakuan istimewa, seperti akses eksklusif ke produk atau layanan, serta program loyalitas dengan reward yang menarik.
        """
        st.markdown(limaa)
    if clust == 6 :
        enamm = """
        \n**Cluster 0: "Low-Spending Infrequent Shoppers"**
       \n •	Karakteristik Pelanggan:
            Pelanggan dalam cluster ini memiliki nilai pembelian yang rendah, frekuensi belanja yang jarang, dan recency yang sudah lama. Mereka cenderung tidak aktif dalam berbelanja.
       \n •	Strategi Pemasaran:
            Membuat promosi khusus atau diskon yang menarik untuk mendorong pelanggan dalam cluster ini agar lebih sering berbelanja. Menawarkan insentif tambahan untuk pembelian berulang dapat membantu meningkatkan keterlibatan mereka.
        \n**Cluster 1: "Moderate Spending Regular Shoppers"**
       \n •	Karakteristik Pelanggan:
            Pelanggan dalam cluster ini memiliki nilai pembelian yang moderat, frekuensi belanja yang cukup sering, dan recency yang relatif baru. Mereka merupakan pembeli reguler dengan kontribusi finansial yang stabil.
       \n •	Strategi Pemasaran:
            Fokus pada mempertahankan keterlibatan dengan memberikan insentif kepada pelanggan untuk terus berbelanja secara teratur. Program loyalitas dengan reward yang menarik atau promosi berbasis frekuensi pembelian dapat membantu meningkatkan loyalitas pelanggan.
        \n**Cluster 2: "High-Value Frequent Shoppers"**
       \n •	Karakteristik Pelanggan:
            Pelanggan dalam cluster ini memiliki nilai pembelian yang sangat tinggi, frekuensi belanja yang tinggi, dan recency yang relatif baru. Mereka adalah pembeli yang aktif dan memberikan kontribusi finansial yang besar bagi perusahaan.
       \n •	Strategi Pemasaran:
            Menyediakan pengalaman belanja yang personal dan eksklusif untuk mempertahankan hubungan dengan pelanggan dalam cluster ini. Program loyalitas dengan reward yang eksklusif atau promosi produk yang relevan dapat membantu mempertahankan loyalitas mereka.
        \n**Cluster 3: "Moderate Spending Occasional Shoppers"**
       \n •	Karakteristik Pelanggan:
            Pelanggan dalam cluster ini memiliki nilai pembelian yang moderat, frekuensi belanja yang sporadis, dan recency yang relatif baru. Mereka cenderung berbelanja secara teratur namun tidak sering.
       \n •	Strategi Pemasaran:
            Membuat promosi khusus atau diskon untuk menarik pelanggan dalam cluster ini kembali ke platform. Menawarkan insentif untuk pembelian berikutnya atau rekomendasi produk yang relevan dapat membantu meningkatkan keterlibatan mereka.
        \n**Cluster 4: "High-Spending Regular Shoppers"**
       \n •	Karakteristik Pelanggan:
            Pelanggan dalam cluster ini memiliki nilai pembelian yang tinggi, frekuensi belanja yang cukup sering, dan recency yang relatif baru. Mereka adalah pembeli reguler yang memberikan kontribusi finansial yang signifikan.
       \n •	Strategi Pemasaran:
            okus pada mempertahankan kepuasan pelanggan dengan memberikan pengalaman belanja yang personal dan eksklusif. Program loyalitas dengan reward yang menarik atau promosi diskon untuk pembelian berikutnya dapat membantu mempertahankan keterlibatan mereka.
        \n**Cluster 5: "Very High-Value Infrequent Shoppers"**
       \n •	Karakteristik Pelanggan:
            Pelanggan dalam cluster ini memiliki nilai pembelian yang sangat tinggi, namun frekuensi belanja yang rendah, dan recency yang cukup lama. Meskipun tidak sering berbelanja, mereka memberikan kontribusi finansial yang besar.
       \n •	Strategi Pemasaran:
            Fokus pada menjaga hubungan dengan memberikan perlakuan istimewa kepada pelanggan dalam cluster ini. Menawarkan akses eksklusif ke produk atau layanan, serta program loyalitas dengan reward yang menarik dapat membantu mempertahankan loyalitas mereka.
        """
        st.markdown(enamm)
    if clust == 7 :
        tujuhh = """
        \n**Cluster 0: "Regular Moderate-Spenders"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini berbelanja secara teratur dengan jumlah belanja yang moderat.
       \n •	Strategi Pemasaran:
            Tawarkan program loyalitas atau diskon reguler untuk mempertahankan kebiasaan belanja mereka.
        \n**Cluster 1: "Infrequent High-Spenders"**
       \n •	Karakteristik Pelanggan:
            Meskipun jarang berbelanja, pembeli dalam cluster ini menghabiskan jumlah yang signifikan setiap kali mereka berbelanja.
       \n •	Strategi Pemasaran:
            Berikan penawaran khusus atau diskon untuk memancing pembelian lebih lanjut dari mereka.
        \n**Cluster 2: "High-Value Occasional Buyers"**
       \n •	Karakteristik Pelanggan:
            Meskipun jarang berbelanja, pembeli dalam cluster ini menghabiskan jumlah yang besar setiap kali mereka berbelanja.
       \n •	Strategi Pemasaran:
            Fokus pada membangun hubungan dan memberikan insentif untuk pembelian lebih lanjut.
        \n**Cluster 3: "Occasional Moderate-Spenders"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini berbelanja secara tidak teratur dengan jumlah belanja yang moderat.
       \n •	Strategi Pemasaran:
            Tawarkan diskon atau penawaran spesial untuk mendorong pembelian lebih sering.
        \n**Cluster 4: "Frequent Big-Spenders"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini berbelanja secara sering dengan jumlah belanja yang besar setiap kali.
       \n •	Strategi Pemasaran:
            Berikan penawaran eksklusif atau insentif pembelian lanjutan untuk mempertahankan keaktifan mereka.
        \n**Cluster 5: "Regular Low-Spenders"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini berbelanja secara teratur namun dengan jumlah belanja yang lebih rendah.
       \n •	Strategi Pemasaran:
            Tawarkan diskon atau promosi khusus untuk meningkatkan nilai belanja mereka.
        \n**Cluster 6: "Frequent Big-Spenders"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini berbelanja secara sering dengan jumlah belanja yang besar setiap kali.
       \n •	Strategi Pemasaran:
            Fokus pada pengalaman pelanggan yang eksklusif dan penawaran khusus untuk mempertahankan tingkat pembelian mereka.
        """
        st.markdown(tujuhh)
    if clust == 8 :
        delapann = """
        \n**Cluster 0: "Occasional High-Spenders"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini jarang berbelanja, tetapi ketika mereka melakukannya, mereka menghabiskan jumlah yang besar.
       \n •	Strategi Pemasaran:
            Tawarkan penawaran khusus atau diskon untuk memancing pembelian lebih lanjut.
        \n**Cluster 1: "Moderate Spenders"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini memiliki tingkat pembelian dan frekuensi yang moderat.
       \n •	Strategi Pemasaran:
            Fokus pada memberikan pengalaman pembelian yang memuaskan untuk mempertahankan kecenderungan pembelian mereka.
        \n**Cluster 2: "Low-Spenders"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini memiliki nilai belanja yang relatif rendah.
       \n •	Strategi Pemasaran:
            Tawarkan insentif atau promosi untuk meningkatkan frekuensi pembelian mereka.
        \n**Cluster 3: "Frequent Big-Spenders"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini berbelanja secara sering dengan jumlah belanja yang besar setiap kali.
       \n •	Strategi Pemasaran:
            Berfokus pada memberikan pengalaman belanja yang eksklusif dan penawaran khusus untuk mempertahankan keaktifan pembelian mereka.
        \n**Cluster 4: "Occasional Moderate-Spenders"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini tidak sering berbelanja, namun ketika mereka melakukannya, mereka menghabiskan jumlah yang moderat.
       \n •	Strategi Pemasaran:
            Tawarkan diskon atau promosi untuk memancing pembelian lebih lanjut.
        \n**Cluster 5: "Big Spenders"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini memiliki nilai pembelian tertinggi.
       \n •	Strategi Pemasaran:
            Fokus pada memberikan pengalaman belanja yang mewah dan eksklusif untuk mempertahankan kecenderungan pembelian mereka.
        \n**Cluster 6: "Regular Spenders"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini berbelanja dengan frekuensi yang moderat namun konsisten.
       \n •	Strategi Pemasaran:
             Tawarkan program loyalitas atau diskon reguler untuk mempertahankan keaktifan pembelian mereka.
        \n**Cluster 7: "Frequent Low-Spenders"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini sering berbelanja dengan nilai belanja yang relatif rendah setiap kali.
       \n •	Strategi Pemasaran:
            Fokus pada meningkatkan nilai belanja per transaksi dengan menawarkan penawaran atau insentif tambahan.
        """
        st.markdown(delapann)
    if clust == 9 :
        sembilann = """
        \n**Cluster 0: "Occasional Spenders"**
        \n•	Karakteristik Pelanggan:
            Pembeli dalam cluster ini berbelanja secara teratur, namun dengan nilai belanja yang cenderung lebih rendah.
        \n•	Strategi Pemasaran:
            Tawarkan insentif untuk meningkatkan nilai belanja per transaksi, seperti diskon tambahan untuk pembelian dalam jumlah tertentu.
        \n**Cluster 1: "Infrequent Bargain Shoppers"**
        \n•	Karakteristik Pelanggan:
            Pembeli dalam cluster ini jarang berbelanja, dan ketika mereka melakukannya, mereka cenderung membeli produk dengan harga yang lebih terjangkau.
        \n•	Strategi Pemasaran:
            Fokus pada penawaran diskon atau promo yang menarik untuk mendorong pembelian dan meningkatkan frekuensi belanja.
        \n**Cluster 2: "Frequent High-Spenders"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini sering berbelanja dengan nilai belanja yang tinggi.
       \n •	Strategi Pemasaran:
            Berfokus pada memberikan pengalaman belanja yang eksklusif dan penawaran khusus untuk mempertahankan keaktifan pembelian mereka.
        \n**Cluster 3: "Moderate Consistent Buyers"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini memiliki kebiasaan belanja yang moderat dan konsisten.
       \n •	Strategi Pemasaran:
            Tawarkan program loyalitas atau diskon reguler untuk mempertahankan keaktifan pembelian mereka.
        \n**Cluster 4: "Regular Moderate-Spenders"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini berbelanja secara teratur dengan nilai belanja yang moderat.
       \n •	Strategi Pemasaran:
            Berikan penawaran khusus atau insentif untuk meningkatkan nilai belanja per transaksi.
        \n**Cluster 5: "Intermittent Budget Shoppers"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini berbelanja secara tidak teratur, dan cenderung memilih produk dengan harga terjangkau.
       \n •	Strategi Pemasaran:
            Fokus pada penawaran diskon atau promo yang menarik untuk mendorong pembelian dan meningkatkan frekuensi belanja.
        \n**Cluster 6: "Steady Moderate Buyers"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini memiliki kebiasaan belanja yang moderat dan konsisten.
       \n •	Strategi Pemasaran:
            Tawarkan program loyalitas atau diskon reguler untuk mempertahankan keaktifan pembelian mereka.
        \n**Cluster 7: "Frequent Big-Spenders"**
       \n •	Karakteristik Pelanggan:
            Pembeli dalam cluster ini sering berbelanja dengan nilai belanja yang besar.
       \n •	Strategi Pemasaran:
            Fokus pada memberikan pengalaman belanja yang mewah dan eksklusif untuk mempertahankan kecenderungan pembelian mereka.
        \n**Cluster 8: "Consistent Low-Spenders"**
        \n•	Karakteristik Pelanggan:
            Pembeli dalam cluster ini memiliki kebiasaan belanja yang konsisten, namun dengan nilai belanja yang cenderung rendah.
        \n•	Strategi Pemasaran:
            Tawarkan insentif untuk meningkatkan nilai belanja per transaksi, seperti diskon tambahan untuk pembelian dalam jumlah tertentu.
        """
        st.markdown(sembilann)
import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from pydantic_settings import BaseSettings

report = ProfileReport(rfm_trans, title="Report", html={'style': {'full_width':True}}, explorative=True, missing_diagrams={'bar': True})


   # Simpan laporan ke dalam file HTML
report.to_file("report.html")
   
   # Baca isi file HTML
html_file = open("report.html", "r")
html_content = html_file.read()
   
   # Menampilkan laporan HTML dalam aplikasi Streamlit sebagai komponen HTML
st.components.v1.html(html_content, width=1000, height=1000, scrolling=True)
