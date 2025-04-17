import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 假设数据存储在CSV文件中
data = pd.read_excel(r'D:\Users\HUAWEI\Desktop\scl90_data.xlsx', sheet_name=0)
X = data.iloc[:, 10:20].values  # 转换为NumPy数组

# 数据标准化
scaler = StandardScaler()
y=data.iloc[:, 10:20]
X_scaled = scaler.fit_transform(X)
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.models import Model # type: ignore

# 定义自编码器
input_dim = 10  # SCL-90的十个因子
encoding_dim = 2  # 嵌入空间维度

# 编码器
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)

# 解码器
decoder = Dense(input_dim, activation="sigmoid")(encoder)

# 自编码器模型
autoencoder = Model(input_layer, decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# 训练自编码器
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=1)
# 提取嵌入表示
encoder_model = Model(input_layer, encoder)
embeddings = encoder_model.predict(X_scaled)
from sklearn.cluster import KMeans

# 使用K-Means聚类
n_clusters = 3 # 假设分为4个群体
kmeans = KMeans(n_clusters=n_clusters)
clusters = kmeans.fit_predict(embeddings)

# 将聚类结果添加到原始数据中
data['Cluster'] = clusters
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 使用t-SNE进行可视化
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings)


# 统计每一簇的个数
unique, counts = np.unique(clusters, return_counts=True)
cluster_counts = dict(zip(unique, counts))

print("每一簇的个数:", cluster_counts)

from sklearn.metrics import calinski_harabasz_score
# 计算K-Means的卡林斯基-哈拉巴斯指数
kmeans_ch_score = calinski_harabasz_score(embeddings, kmeans.labels_)
print(f"K-Means Calinski-Harabasz Index: {kmeans_ch_score}")
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# 计算K-Means的戴维森堡丁指数
kmeans_db_score = davies_bouldin_score(embeddings, kmeans.labels_)
print(f"K-Means Davies-Bouldin Index: {kmeans_db_score}")
from sklearn.metrics import silhouette_score

# 计算K-Means的轮廓系数
kmeans_score = silhouette_score(embeddings, kmeans.labels_)
print(f"K-Means Silhouette Score: {kmeans_score}")
def get_cluster_data(cluster_id):
    
    cluster_data = data[y['Cluster'] == cluster_id]
    return cluster_data
# 假设第一列是样本编号
sample_ids = data.iloc[:, 0]  # 获取第一列作为样本编号

# 创建一个字典来存储每个簇的样本编号
cluster_samples = {}
for cluster_id in range(n_clusters):
    cluster_samples[f'Cluster_{cluster_id}'] = sample_ids[clusters == cluster_id].tolist()

#每个簇一个工作表
with pd.ExcelWriter(r'D:\Users\HUAWEI\Desktop\cluster_samples_sheets.xlsx') as writer:
    for cluster_id in range(n_clusters):
        cluster_data = sample_ids[clusters == cluster_id].to_frame(name='Sample_ID')
        cluster_data.to_excel(writer, sheet_name=f'Cluster_{cluster_id}', index=False)

print("已创建包含多个工作表的Excel文件，每个工作表对应一个簇")

# 绘制聚类结果
plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    plt.scatter(embeddings_tsne[clusters == i, 0], embeddings_tsne[clusters == i, 1], label=f'Cluster {i}')
plt.title('SCL-90 Clustering Results')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()
plt.show()
from tensorflow.keras.models import save_model



from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import MeanSquaredError


autoencoder.compile(optimizer='adam', loss='mean_squared_error')


autoencoder.compile(optimizer='adam', loss=MeanSquaredError())


autoencoder.save('scl90_autoencoder.h5') 

loaded_model = load_model('scl90_autoencoder.h5', custom_objects={'MeanSquaredError': MeanSquaredError})

# 保存嵌入数据
np.save('embeddings.npy', embeddings)  # 保存到当前目录

np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_var.npy', scaler.var_)  # 必须保存
np.save('scaler_scale.npy', scaler.scale_)
print("embeddings.npy 已生成！")