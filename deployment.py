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

data = st.sidebar.radio("Data",['None','Isi Dataset','Data Pelanggan'])
clus = st.sidebar.radio("Evaluation N Cluster",['None','Elbow Method','Silhouette Coefficient'])
rfm = st.sidebar.radio("Visualization of RFM Segments", ['None','Pie Chart','Histogram','Treemap','Barplot','Line Plot Avg','Treemap Squarify','Word Cloud','Scatter Matrix Pairplot','Scatter 3d Plot'])
kmean = st.sidebar.radio("Visualization of K-Means Segments", ['None','Intercluster Distance Map','Pie Chart'])
output = st.sidebar.radio("Karakteristik Pelanggan dan Strategi Pemasaran",['None','RFM Segments','K-Means Segments'])
output2 = st.sidebar.radio("Hasil Segmentasi Pelanggan", ['None','RFM Segments','K-Means Segments'])

## Read dataset
df = pd.read_csv('dataset-inisialisasi.csv')
df['ID'] = df['ID'].astype('str')
df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)


st.title("Customer Segmentation")
st.header("Kaosdisablon")
st.text("Jl. KNPI No 22 Ciseureuh Purwakarta")
st.markdown("[Instagram](https://www.instagram.com/kaosdisablon/)")
st.markdown("---")

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
                        width=1000, height=800)

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
        return {0: 'Lost/Need Attention', 1: 'Current Customer'}
    elif clust == 3:
        return {0: 'Lost/Need Attention', 1: 'Current Customer', 2: 'Top/Best Customer'}
    elif clust == 4:
        return {0: 'Lost/Need Attention', 1: 'Current Customer', 2: 'Top/Best Customer', 3: 'Cluster 3'}
    # Tambahkan kondisi elif untuk nilai clust lainnya sampai 10
    elif clust == 5:
        return {0: 'Lost/Need Attention', 1: 'Current Customer', 2: 'Top/Best Customer', 3: 'Cluster 3', 4: 'Cluster 4'}
    elif clust == 6:
        return {0: 'Lost/Need Attention', 1: 'Current Customer', 2: 'Top/Best Customer', 3: 'Cluster 3', 4: 'Cluster 4', 5: 'Cluster 5'}
    elif clust == 7:
        return {0: 'Lost/Need Attention', 1: 'Current Customer', 2: 'Top/Best Customer', 3: 'Cluster 3', 4: 'Cluster 4', 5: 'Cluster 5', 6: 'Cluster 6'}
    elif clust == 8:
        return {0: 'Lost/Need Attention', 1: 'Current Customer', 2: 'Top/Best Customer', 3: 'Cluster 3', 4: 'Cluster 4', 5: 'Cluster 5', 6: 'Cluster 6', 7: 'Cluster 7'}
    elif clust == 9:
        return {0: 'Lost/Need Attention', 1: 'Current Customer', 2: 'Top/Best Customer', 3: 'Cluster 3', 4: 'Cluster 4', 5: 'Cluster 5', 6: 'Cluster 6', 7: 'Cluster 7', 8: 'Cluster 8'}
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
    st.write("•	Karakteristik: Terakhir kali melakukan pembelian jauh sebelumnya, tetapi dalam 4 minggu terakhir mereka telah mengunjungi situs atau membuka email.")
    st.write("•	Strategi: Membuat subjek email yang sangat dipersonalisasi atau memberikan diskon spesifik pada produk tertentu untuk membangkitkan kembali minat mereka.")
    st.write("**Cannot Lose Them But Losing**")
    st.write("•	Karakteristik: Melakukan pembelian besar dan sering, namun sudah lama tidak kembali.")
    st.write("•	Strategi: Mencoba memenangkan mereka kembali melalui perpanjangan layanan atau produk baru. Fokus pada personalisasi tinggi untuk mempertahankan mereka.")
    st.write("**At Risk**")
    st.write("•	Karakteristik: Mirip dengan 'Cannot Lose Them' namun dengan nilai moneter dan frekuensi yang lebih rendah.")
    st.write("•	Strategi: Menyediakan sumber daya yang membantu di situs web atau mengirim email yang dipersonalisasi untuk mempertahankan hubungan dengan mereka.")
    st.write("**Hibernating Customers**")
    st.write("•	Karakteristik: Pelanggan yang sebelumnya melakukan pembelian yang lebih kecil dan jarang, tetapi sudah lama tidak melakukan pembelian.")
    st.write("•	Strategi: Mengikutsertakan mereka dalam komunikasi email standar, tetapi memeriksa secara teratur agar konten Anda tidak dianggap sebagai spam. Tidak perlu mengalokasikan sumber daya berlebih di segmen ini.")
    st.write("**Lost Customers**")
    st.write("•	Karakteristik: Melakukan pembelian terakhir sudah lama dan tidak ada interaksi sama sekali dalam 4 minggu terakhir.")
    st.write("•	Strategi: Membangkitkan kembali minat mereka melalui kampanye outreach, namun jika tidak berhasil, lebih baik fokus pada segmen lain yang lebih responsif.")
if output == "K-Means Segments" :
    st.subheader("Karakteristik Pelanggan dan Strategi Pemasaran berdasarkan K-Means Segments")
    st.write("Dengan Jumlah Cluster ", clust)
    st.markdown("---")
    if clust == 2 :
        duaa = """
        \n**Cluster 0: "Diverse Engagement & Retention Focus"**
        \n•	Karakteristik Pelanggan:
            Terdiri dari campions, promising, yang memerlukan perhatian, pelanggan baru, potensial loyalist, loyal, hibernating, about to sleep.
        \n•	Strategi Pemasaran:
            Fokus pada berbagai metode retensi dan peningkatan keterlibatan dengan pendekatan yang luas.
        \n**Cluster 1: "High-Risk Reactivation & Retention"**
        \n•	Karakteristik Pelanggan:
            Melibatkan pelanggan yang hibernating, about to sleep, memerlukan perhatian, pelanggan baru, potensial, at risk, cannot lose, promising, kehilangan, loyal, campions.
        \n•	Strategi Pemasaran:
            Strategi yang terfokus pada reaktivasi dan retensi tingkat tinggi, memperhitungkan risiko yang beragam dalam kehilangan pelanggan.
        """
        st.markdown(duaa)
    if clust == 3 :
        tigaa = """
        \n**Cluster 0: "Versatile Engagement & Growth Potential"**
        \n•	Karakteristik Pelanggan:
            Terdiri dari campions, promising, yang memerlukan perhatian, pelanggan baru, potensial loyalist, dan loyal.
        \n•	Strategi Pemasaran:
            Pendekatan yang beragam untuk meningkatkan keterlibatan, pertumbuhan, dan mengoptimalkan potensi pelanggan.
        \n**Cluster 1: "Risk Management & Retention Focus"**
        \n•	Karakteristik Pelanggan:
            Melibatkan campions, promising, loyal, yang berisiko kehilangan, tidak bisa kehilangan.
        \n•	Strategi Pemasaran:
            Fokus pada manajemen risiko kehilangan pelanggan, dengan penekanan pada strategi retensi yang efektif.
        \n**Cluster 2: "Reactivation & Diverse Risk Mix"**
        \n•	Karakteristik Pelanggan:
            Termasuk pelanggan yang hampir tidur, hibernating, yang memerlukan perhatian, pelanggan baru, potensial, yang berisiko kehilangan, tidak bisa kehilangan, promising, loyal, campions.
        \n•	Strategi Pemasaran:
            Pendekatan yang luas untuk membangkitkan kembali minat, menyesuaikan strategi dengan berbagai tingkat risiko dan keterlibatan pelanggan.
        """
        st.markdown(tigaa)
    if clust == 4 :
        empatt = """
        \n**Cluster 0: "Revival & Engagement Focus"**
       \n •	Karakteristik Pelanggan:
            Rentang pelanggan yang luas, mulai dari hibernating hingga yang berisiko kehilangan.
       \n •	Strategi Pemasaran:
            Memfokuskan usaha untuk membangkitkan kembali minat dan meningkatkan keterlibatan, menggunakan promosi yang dipersonalisasi untuk menggerakkan mereka kembali.
        \n**Cluster 1: "Active & Developmental Focus"**
       \n •	Karakteristik Pelanggan:
            Merangkul campions, pelanggan potensial, dan yang membutuhkan perhatian tambahan.
       \n •	Strategi Pemasaran:
            Berfokus pada pengembangan hubungan yang lebih dalam, menawarkan program loyalitas dan promosi yang dirancang khusus.
        \n**Cluster 2: "High-Value Retention & Risk Management"**
       \n •	Karakteristik Pelanggan:
            Terdiri dari pelanggan yang memiliki nilai tinggi tetapi rentan terhadap risiko kehilangan.
       \n •	Strategi Pemasaran:
            Menjaga pelanggan tingkat lanjut dengan fokus pada retensi dan pengelolaan risiko kehilangan.
        \n**Cluster 3: "Loyalty & Potential Focus"**
       \n •	Karakteristik Pelanggan:
            Terdiri dari campions, pelanggan potensial yang loyal, dan yang membutuhkan perhatian.
       \n •	Strategi Pemasaran:
            Memperkuat loyalitas dengan menawarkan promosi dan program keanggotaan yang lebih memikat, serta mengimplementasikan strategi pemasaran yang lebih terfokus.
        """
        st.markdown(empatt)
    if clust == 5 :
        limaa = """
        \n**Cluster 0: "High-Value & Risk Management"**
       \n •	Karakteristik Pelanggan:
            Meliputi campions, pelanggan potensial, yang loyal, berisiko kehilangan, dan tidak boleh hilang.
       \n •	Strategi Pemasaran:
            Fokus pada manajemen risiko dan mempertahankan pelanggan bernilai tinggi dengan penekanan pada personalisasi dan strategi retensi yang kuat.
        \n**Cluster 1: "Comprehensive Engagement & Attention"**
       \n •	Karakteristik Pelanggan:
            Menyeliputi campions, pelanggan potensial yang baru, yang memerlukan perhatian, loyal, hibernating, dan hampir tidur.
       \n •	Strategi Pemasaran:
            Memiliki pendekatan yang holistik untuk keterlibatan, dengan strategi yang beragam untuk merangkul dan mempertahankan pelanggan.
        \n**Cluster 2: "Loyalty & Potential Focus"**
       \n •	Karakteristik Pelanggan:
            Terdiri dari campions, pelanggan potensial yang baru, dan yang membutuhkan perhatian, yang loyal.
       \n •	Strategi Pemasaran:
            Menekankan pada pengembangan loyalitas dan memanfaatkan potensi pelanggan yang ada.
        \n**Cluster 3: "Revival & Retention Focus"**
       \n •	Karakteristik Pelanggan:
            Melibatkan pelanggan yang hibernating, hampir tidur, yang memerlukan perhatian, baru, berisiko kehilangan, yang tidak boleh hilang, dan potensial.
       \n •	Strategi Pemasaran:
            Menekankan pada penghidupan kembali minat dan retensi melalui promosi yang dipersonalisasi dan strategi retensi yang kuat.
        \n**Cluster 4: "Engagement & Attention Driven"**
        \n•	Karakteristik Pelanggan:
            Terdiri dari campions, pelanggan potensial yang membutuhkan perhatian lebih, dan yang loyal.
        \n•	Strategi Pemasaran:
            Fokus pada keterlibatan yang mendalam dan perhatian terhadap pelanggan yang berpotensi berkembang menjadi campions atau pelanggan setia.
        """
        st.markdown(limaa)
    if clust == 6 :
        enamm = """
        \n**Cluster 0: "Engaged & Potential Loyalty"**
       \n •	Karakteristik Pelanggan:
            Meliputi campions, yang menunjukkan potensi loyalitas, loyal, yang memerlukan perhatian.
       \n •	Strategi Pemasaran:
            Fokus pada mempertahankan keterlibatan yang kuat, menawarkan insentif untuk memperkuat keterikatan, dan memberikan perhatian ekstra kepada pelanggan.
        \n**Cluster 1: "Diverse Engagement & Risk"**
       \n •	Karakteristik Pelanggan:
            Menyeliputi pelanggan yang hampir tidur, hibernating, yang memerlukan perhatian, baru, berisiko kehilangan, dan pelanggan berpotensi.
       \n •	Strategi Pemasaran:
            Memiliki pendekatan yang beragam untuk keterlibatan, dengan fokus pada mengelola risiko dan menarik kembali minat yang terkait.
        \n**Cluster 2: "Loyalty & Retention Focus"**
       \n •	Karakteristik Pelanggan:
            Terdiri dari campions, yang menunjukkan potensi loyalitas, yang setia, berisiko kehilangan.
       \n •	Strategi Pemasaran:
            Menekankan pada strategi retensi dan pengembangan loyalitas yang kuat.
        \n**Cluster 3: "Comprehensive Attention & Engagement"**
       \n •	Karakteristik Pelanggan:
            Meliputi campions, pelanggan potensial yang memerlukan perhatian, yang baru, yang loyal, yang hibernating, hampir tidur.
       \n •	Strategi Pemasaran:
            Menekankan pada keterlibatan menyeluruh, memberikan perhatian khusus, dan retensi pelanggan.
        \n**Cluster 4: "Risk & Value Management"**
       \n •	Karakteristik Pelanggan:
            Terdiri dari pelanggan berisiko kehilangan, potensial, loyal, dan campions.
       \n •	Strategi Pemasaran:
            Fokus pada manajemen risiko dan mempertahankan pelanggan dengan nilai tinggi melalui strategi retensi yang kuat.
        \n**Cluster 5: "Loyalty & Attention Balance"**
       \n •	Karakteristik Pelanggan:
            Melibatkan campions, pelanggan potensial, yang memerlukan perhatian, yang loyal.
       \n •	Strategi Pemasaran:
            Menemukan keseimbangan antara keterlibatan dan perhatian, mempertahankan loyalitas sambil memberikan perhatian yang diperlukan.
        """
        st.markdown(enamm)
    if clust == 7 :
        tujuhh = """
        \n**Cluster 0: "Diverse Engagement & Risks"**
       \n •	Karakteristik Pelanggan:
            Melibatkan hampir semua jenis pelanggan dengan tingkat keterlibatan dan risiko yang beragam.
       \n •	Strategi Pemasaran:
            Memerlukan pendekatan yang beragam dalam mengelola keterlibatan dan risiko untuk setiap jenis pelanggan dalam segmen ini.
        \n**Cluster 1: "Potential Loyalists & New Customer Focus"**
       \n •	Karakteristik Pelanggan:
            Termasuk campions, yang menunjukkan potensi loyalitas, dan pelanggan baru.
       \n •	Strategi Pemasaran:
            Fokus pada membangun hubungan awal yang kuat dengan pelanggan baru dan menciptakan program loyalitas untuk menarik minat pelanggan potensial.
        \n**Cluster 2: "Loyal Attention & Promising Engagement"**
       \n •	Karakteristik Pelanggan:
            Meliputi campions yang memerlukan perhatian dan loyal.
       \n •	Strategi Pemasaran:
            Menawarkan strategi retensi yang kuat dan perhatian khusus kepada pelanggan setia dalam segmen ini.
        \n**Cluster 3: "Comprehensive Attention & Engagement"**
       \n •	Karakteristik Pelanggan:
            Melibatkan campions, pelanggan potensial yang memerlukan perhatian, loyal, hibernating, dan hampir tidur.
       \n •	Strategi Pemasaran:
            Memerlukan pendekatan holistik yang meliputi semua jenis pelanggan dalam segmen ini.
        \n**Cluster 4: "Risk & Value Management"**
       \n •	Karakteristik Pelanggan:
            Terdiri dari pelanggan yang berisiko kehilangan, potensial, loyal, dan campions.
       \n •	Strategi Pemasaran:
            Fokus pada manajemen risiko dan mempertahankan pelanggan dengan nilai tinggi melalui strategi retensi yang kuat.
        \n**Cluster 5: "Potential Loyalty & New Customer Focus"**
       \n •	Karakteristik Pelanggan:
            Termasuk campions, pelanggan potensial yang memerlukan perhatian, dan pelanggan baru.
       \n •	Strategi Pemasaran:
            Berfokus pada membangun hubungan awal yang kuat dengan pelanggan baru dan menciptakan program loyalitas untuk menarik minat pelanggan potensial.
        \n**Cluster 6: "Retention-Focused Champions"**
       \n •	Karakteristik Pelanggan:
            Melibatkan campions yang berfokus pada retensi, pelanggan potensial, loyal, dan yang berisiko kehilangan.
       \n •	Strategi Pemasaran:
            Menerapkan strategi retensi yang kuat dan memberikan perhatian khusus kepada pelanggan setia dalam segmen ini.
        """
        st.markdown(tujuhh)
    if clust == 8 :
        delapann = """
        \n**Cluster 0: "Engaged Core"**
       \n •	Karakteristik Pelanggan:
            Termasuk campions, loyal, dan promising.
       \n •	Strategi Pemasaran:
            Fokus pada mempertahankan keterlibatan tinggi yang sudah ada dan merencanakan untuk pertumbuhan lebih lanjut melalui program loyalitas dan promosi.
        \n**Cluster 1: "Varied Engagement with High Potentials"**
       \n •	Karakteristik Pelanggan:
            Memiliki campion, pelanggan yang hampir tidur, hibernating, loyal, yang memerlukan perhatian, pelanggan baru, potensial loyalist, dan promising.
       \n •	Strategi Pemasaran:
            Memerlukan pendekatan yang disesuaikan untuk menarik kembali yang hampir tidur, membangun kembali hubungan dengan yang hibernating, dan memperkuat keterlibatan pelanggan baru dan potensial.
        \n**Cluster 2: "Focused New & Potential Loyalists"**
       \n •	Karakteristik Pelanggan:
            Terdiri dari campions, promising, pelanggan yang memerlukan perhatian, pelanggan baru, dan potensial loyalist.
       \n •	Strategi Pemasaran:
            Menitikberatkan pada membangun hubungan awal yang kuat dengan pelanggan baru, menarik minat pelanggan potensial, dan memberikan perhatian khusus kepada yang memerlukan.
        \n**Cluster 3: "Risk Management & Loyal Focus"**
       \n •	Karakteristik Pelanggan:
            Meliputi pelanggan yang berisiko kehilangan, tidak bisa kehilangan mereka, loyal, dan promising.
       \n •	Strategi Pemasaran:
            Fokus pada manajemen risiko, mempertahankan pelanggan dengan nilai tinggi, dan meningkatkan keterlibatan dengan pelanggan yang loyal.
        \n**Cluster 4: "Comprehensive Customer Engagement"**
       \n •	Karakteristik Pelanggan:
            Termasuk campions, promising, pelanggan yang memerlukan perhatian, pelanggan baru, potensial loyalist, dan loyal.
       \n •	Strategi Pemasaran:
            Memerlukan pendekatan holistik yang mencakup semua jenis pelanggan untuk memastikan keterlibatan yang kuat dan retensi yang baik.
        \n**Cluster 5: "Engagement & Loyal Focus"**
       \n •	Karakteristik Pelanggan:
            Melibatkan campions, promising, pelanggan yang memerlukan perhatian, dan loyal.
       \n •	Strategi Pemasaran:
            Menekankan strategi retensi yang kuat dan memberikan perhatian khusus kepada pelanggan setia dalam segmen ini.
        \n**Cluster 6: "Potential Re-engagement & Varied Risks"**
       \n •	Karakteristik Pelanggan:
            Terdiri dari pelanggan yang hampir tidur, hibernating, pelanggan yang memerlukan perhatian, pelanggan baru, potensial, yang berisiko kehilangan, tidak bisa kehilangan, dan promising.
       \n •	Strategi Pemasaran:
            Memerlukan pendekatan berbeda untuk menarik kembali yang hampir tidur, membangun kembali hubungan dengan yang hibernating, dan meminimalkan risiko kehilangan untuk pelanggan dengan potensi tinggi.
        \n**Cluster 7: "Focused Risk Management & Loyalty"**
       \n •	Karakteristik Pelanggan:
            Meliputi pelanggan yang berisiko kehilangan, tidak bisa kehilangan mereka, promising, dan loyal.
       \n •	Strategi Pemasaran:
            Fokus pada manajemen risiko dan retensi pelanggan dengan nilai tinggi melalui personalisasi tinggi dan strategi retensi yang kuat.
        """
        st.markdown(delapann)
    if clust == 9 :
        sembilann = """
        \n**Cluster 0: "Retention Risk Mix"**
        \n•	Karakteristik Pelanggan:
            Termasuk pelanggan yang hampir tidur, hibernating, yang berisiko kehilangan, tidak bisa kehilangan, dan pelanggan yang sudah hilang.
        \n•	Strategi Pemasaran:
            Fokus pada retensi pelanggan dengan risiko tinggi, mengembalikan minat pelanggan yang terlupakan, dan mendorong keterlibatan yang lebih kuat.
        \n**Cluster 1: "Engaged Diversity & Growth Potential"**
        \n•	Karakteristik Pelanggan:
            Melibatkan campions, promising, yang memerlukan perhatian, pelanggan baru, potensial loyalist, dan loyal.
        \n•	Strategi Pemasaran:
            Memerlukan pendekatan yang beragam untuk meningkatkan keterlibatan, pertumbuhan, dan potensi pelanggan.
        \n**Cluster 2: "Loyalty-Driven Risk Mitigation"**
       \n •	Karakteristik Pelanggan:
            Terdiri dari pelanggan yang berisiko kehilangan, tidak bisa kehilangan, loyal, dan promising.
       \n •	Strategi Pemasaran:
            Memfokuskan pada strategi retensi untuk mengurangi risiko kehilangan pelanggan berharga.
        \n**Cluster 3: "Re-engagement & Mixed Risk"**
       \n •	Karakteristik Pelanggan:
            Melibatkan pelanggan yang hampir tidur, campion, hibernating, loyal, yang memerlukan perhatian, pelanggan baru, potensial loyalist, dan promising.
       \n •	Strategi Pemasaran:
            Memerlukan pendekatan yang berbeda untuk mendorong kembali keterlibatan dan mengatasi risiko yang beragam di dalam segmen ini.
        \n**Cluster 4: "Loyal Risk Management"**
       \n •	Karakteristik Pelanggan:
            Termasuk pelanggan yang berisiko kehilangan, tidak bisa kehilangan, loyal, dan promising.
       \n •	Strategi Pemasaran:
            Fokus pada manajemen risiko kehilangan pelanggan yang loyal dengan mempertahankan keterlibatan mereka.
        \n**Cluster 5: "Diverse Loyalty Focus"**
       \n •	Karakteristik Pelanggan:
            Melibatkan campions, promising, yang memerlukan perhatian, pelanggan baru, dan potensial loyalist.
       \n •	Strategi Pemasaran:
            Mencakup strategi yang beragam untuk mempertahankan keterlibatan, membangun hubungan, dan mengembangkan potensi pelanggan.
        \n**Cluster 6: "Varied Retention & Engagement Mix"**
       \n •	Karakteristik Pelanggan:
            Terdiri dari pelanggan yang hampir tidur, hibernating, yang memerlukan perhatian, pelanggan baru, potensial, yang berisiko kehilangan, tidak bisa kehilangan, promising, dan loyal.
       \n •	Strategi Pemasaran:
            Memerlukan pendekatan yang luas untuk mengatasi berbagai tingkat retensi dan keterlibatan pelanggan.
        \n**Cluster 7: "Focused Loyalty & Attention"**
       \n •	Karakteristik Pelanggan:
            Melibatkan campions, promising, yang memerlukan perhatian, pelanggan baru, dan loyal.
       \n •	Strategi Pemasaran:
            Fokus pada retensi pelanggan setia dan memberikan perhatian khusus kepada mereka.
        \n**Cluster 8: "High-Value Loyalty"**
        \n•	Karakteristik Pelanggan:
            Terdiri dari champions, loyal, dan promising.
        \n•	Strategi Pemasaran:
            Fokus pada strategi retensi dan penghargaan bagi pelanggan dengan nilai tinggi dan loyalitas yang konsisten.
        """
        st.markdown(sembilann)