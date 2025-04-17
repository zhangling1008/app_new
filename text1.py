import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import streamlit as st
import qrcode
from io import BytesIO
import base64
import socket
import sqlite3
from datetime import datetime
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 初始化数据库
def init_db():
    conn = sqlite3.connect('scl90_data.db')
    c = conn.cursor()
    
    # 创建问卷结果表
    c.execute('''CREATE TABLE IF NOT EXISTS responses 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME,
                  responses TEXT,
                  cluster INTEGER,
                  factor_scores TEXT)''')
    
    # 创建用户信息表（可选）
    c.execute('''CREATE TABLE IF NOT EXISTS user_info 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  response_id INTEGER,
                  age INTEGER,
                  gender TEXT,
                  education TEXT,
                  FOREIGN KEY(response_id) REFERENCES responses(id))''')
    
    conn.commit()
    conn.close()

# 初始化数据库
init_db()

# 2. 加载模型和数据
@st.cache_resource
def load_models():
    scaler = StandardScaler()
    autoencoder = load_model('scl90_autoencoder.h5')
    embeddings = np.load('embeddings.npy')
    
    # 加载标准化参数
    scaler.mean_ = np.load('scaler_mean.npy')
    scaler.scale_ = np.load('scaler_scale.npy')
    
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(embeddings)
    return autoencoder, scaler, kmeans

autoencoder, scaler, kmeans = load_models()


# 4. 获取本地IP地址
def get_local_ip():
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return local_ip
    except:
        return "localhost"

# 5. 完整的SCL-90问卷题目
questions = [
    "1. 头痛", "2. 神经过敏", "3. 不必要的想法", "4. 头晕", "5. 对异性兴趣减退",
    "6. 责备他人", "7. 思想被控制感", "8. 责怪他人制造麻烦", "9. 记忆力差", 
    "10. 担心衣饰仪态", "11. 易烦恼激动", "12. 胸痛", "13. 害怕空旷场所",
    "14. 精力下降", "15. 自杀念头", "16. 幻听", "17. 发抖", "18. 不信任他人",
    "19. 胃口差", "20. 易哭泣", "21. 与异性相处不自在", "22. 受骗感", 
    "23. 无缘无故害怕", "24. 控制不住发脾气", "25. 怕单独出门", "26. 自责",
    "27. 腰痛", "28. 完成任务困难", "29. 孤独感", "30. 苦闷", "31. 过分担忧",
    "32. 对事物不感兴趣", "33. 害怕", "34. 感情易受伤害", "35. 他人知道自己的想法",
    "36. 不被理解", "37. 感到他人不友好", "38. 做事必须很慢", "39. 心悸",
    "40. 恶心胃不适", "41. 感到不如他人", "42. 肌肉酸痛", "43. 被监视感",
    "44. 入睡困难", "45. 做事反复检查", "46. 难以决定", "47. 怕乘交通工具",
    "48. 呼吸困难", "49. 忽冷忽热", "50. 因害怕而回避", "51. 脑子变空",
    "52. 身体发麻刺痛", "53. 喉咙梗塞感", "54. 前途无望", "55. 注意力不集中",
    "56. 身体无力", "57. 紧张易紧张", "58. 手脚发重", "59. 想到死亡",
    "60. 吃得太多", "61. 被注视不自在", "62. 不属于自己的想法", 
    "63. 伤害他人冲动", "64. 早醒", "65. 反复洗手", "66. 睡眠不稳",
    "67. 破坏冲动", "68. 特殊想法", "69. 对他人神经过敏", "70. 人多不自在",
    "71. 做事困难", "72. 阵发恐惧", "73. 公共场合进食不适", "74. 经常争论",
    "75. 独处紧张", "76. 成绩未被恰当评价", "77. 孤单感", "78. 坐立不安",
    "79. 无价值感", "80. 熟悉变陌生", "81. 大叫摔东西", "82. 怕当众昏倒",
    "83. 被占便宜感", "84. 性方面苦恼", "85. 该受惩罚", "86. 急于做事",
    "87. 身体严重问题", "88. 与人疏远", "89. 罪恶感", "90. 脑子有毛病"
]

# 6. 保存数据到数据库
def save_to_db(responses, cluster, factor_scores):
    conn = sqlite3.connect('scl90_data.db')
    c = conn.cursor()
    
    # 保存问卷结果
    c.execute('''INSERT INTO responses 
                 (timestamp, responses, cluster, factor_scores) 
                 VALUES (?, ?, ?, ?)''',
              (datetime.now(), 
               str(responses), 
               int(cluster), 
               str(factor_scores)))
    
    # 获取刚插入的ID
    response_id = c.lastrowid
    
    # 保存用户信息（可选）
    if 'user_info' in st.session_state:
        user_info = st.session_state.user_info
        c.execute('''INSERT INTO user_info 
                     (response_id, age, gender, education) 
                     VALUES (?, ?, ?, ?)''',
                  (response_id, 
                   user_info.get('age'), 
                   user_info.get('gender'), 
                   user_info.get('education')))
    
    conn.commit()
    conn.close()
    return response_id

# 7. 创建问卷界面
st.title("SCL-90心理健康自评量表")


with st.sidebar:
   
    # 管理员登录（简化版）
    st.subheader("管理员登录")
    admin_pass = st.text_input("密码", type="password")
    if admin_pass == "admin123":  # 简单密码，实际使用中应该更安全
        st.session_state.admin = True
        st.success("管理员模式已激活")
    elif 'admin' in st.session_state:
        del st.session_state.admin

# 用户信息收集（可选）
with st.expander("个人信息（可选）"):
    user_info = {
        'age': st.number_input("年龄", min_value=10, max_value=100, value=25),
        'gender': st.selectbox("性别", ["男", "女", "其他", "不愿透露"]),
        'education': st.selectbox("教育程度", 
                                ["初中及以下", "高中/中专", "大专/本科", "硕士及以上", "不愿透露"])
    }
    st.session_state.user_info = user_info

# 主问卷区域
with st.form("scl90_form"):
    st.subheader("请根据最近一周的感觉评分（1-5分）：")
    st.caption("1=没有，2=很轻，3=中等，4=偏重，5=严重")
    
    responses = []
    cols = st.columns(5)  # 分5列显示
    for i, q in enumerate(questions):
        with cols[i % 5]:
            responses.append(
                st.radio(
                    q,
                    options=[1, 2, 3, 4, 5],
                    horizontal=True,
                    key=f"q{i}"
                )
            )
    
    submitted = st.form_submit_button("提交评估")
    
    if submitted:
        if len(responses) != 90:
            st.error("请确保回答了所有90个问题")
        else:
            try:
                # 计算因子得分
                factors = {
                    '躯体化': [0,3,11,26,39,41,47,48,51,52,55,57],
                    '强迫症状': [2,8,9,27,37,44,45,50,54,64],
                    '人际关系敏感': [5,20,33,35,36,40,60,68,72],
                    '抑郁': [4,13,14,19,21,25,28,29,30,31,53,70,78],
                    '焦虑': [1,16,22,32,38,56,71,77,79,85],
                    '敌对': [10,23,62,66,73,80],
                    '恐怖': [12,24,46,49,69,74,81],
                    '偏执': [7,17,42,67,75,82],
                    '精神病性': [6,15,34,61,76,83,84,86,87,89],
                    '其他': [18,43,58,59,63,65,88]
                }
                
                # 验证索引范围
                for factor, indices in factors.items():
                    for idx in indices:
                        if idx >= len(responses):
                            raise IndexError(f"因子'{factor}'的索引{idx}超出范围")
                
                factor_scores = []
                for factor, indices in factors.items():
                    score = np.mean([responses[i] for i in indices])
                    factor_scores.append(score)
                
                # 分类预测
                scaled = scaler.transform(np.array(factor_scores).reshape(1, -1))
                encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[1].output)
                embedding = encoder.predict(scaled, verbose=0)
                cluster = kmeans.predict(embedding)[0]
                
                # 保存到数据库
                response_id = save_to_db(responses, cluster, factor_scores)
                
                # 显示结果
                st.success("评估完成！")
                st.info(f"您的评估ID: {response_id} (如需后续咨询请保存此ID)")
                
                descriptions = {
                    0: "您的心理健康状况良好",
                    1: "存在轻度心理困扰",
                    2: "建议寻求专业心理帮助"
                }
                
                st.write(f"**评估结果**: {descriptions[cluster]}")
                
                # 可视化部分修改为：
                fig, ax = plt.subplots()
                all_embeddings = np.load('embeddings.npy')

                # 设置中文字体
                plt.rcParams['font.sans-serif'] = ['SimHei']
                plt.rcParams['axes.unicode_minus'] = False

                for i in range(3):
                    points = all_embeddings[kmeans.labels_ == i]
                    ax.scatter(points[:,0], points[:,1], label=f'群体 {i}', alpha=0.5)
                ax.scatter(embedding[0,0], embedding[0,1], c='red', s=100, label='您的位置')

                # 设置中文标题和标签
                ax.set_title("您在群体中的分布位置", fontsize=14)
                ax.set_xlabel("特征维度1", fontsize=12)
                ax.set_ylabel("特征维度2", fontsize=12)
                ax.legend()

                st.pyplot(fig)
                # 显示因子得分
                st.subheader("各因子得分")
                factor_names = list(factors.keys())
                for name, score in zip(factor_names, factor_scores):
                    st.write(f"{name}: {score:.2f}")
                
            except Exception as e:
                st.error(f"处理出错: {str(e)}")
                st.info("请确保已回答所有问题并重新提交")
            # 在显示建议前添加类型检查和范围验证

        if len(responses) != 90:
            st.error("请确保回答了所有90个问题")
        else:
            try:
                # ... [之前的代码保持不变，直到分类预测部分] ...
                
                # 确保 cluster 是整数且在有效范围内
                cluster = int(cluster)  # 确保转换为整数
                if cluster not in [0, 1, 2]:  # 验证范围
                    cluster = 2  # 默认设为最需要关注的群体
                
                # 保存到数据库
                response_id = save_to_db(responses, cluster, factor_scores)
                
                # 显示结果
                st.success("评估完成！")
                st.info(f"您的评估ID: {response_id} (如需后续咨询请保存此ID)")
                
                descriptions = {
                    0: "您的心理健康状况良好",
                    1: "存在轻度心理困扰",
                    2: "建议寻求专业心理帮助"
                }
                
                st.write(f"**评估结果**: {descriptions.get(cluster, '未知状态')}")
                
                # ... [可视化代码保持不变] ...
                
                # 根据不同簇显示建议 - 修复后的版本
                st.subheader("个性化建议")
                
                cluster_advice = {
                    0: [
                        "您的心理健康状况良好，继续保持积极的生活方式。",
                        "建议定期进行自我心理评估，保持心理健康意识。",
                        "保持规律的作息和适度运动有助于维持当前状态。"
                    ],
                    1: [
                        "您目前存在一些轻度心理困扰，建议关注自己的情绪变化。",
                        "可以尝试一些放松技巧，如深呼吸、冥想或瑜伽。",
                        "与信任的朋友或家人分享您的感受可能会有帮助。",
                        "如果这些症状持续存在或加重，建议寻求专业心理咨询。"
                    ],
                    2: [
                        "您的评估结果显示可能需要专业心理支持。",
                        "建议尽快预约心理咨询师或心理健康专业人士。",
                        "请记住寻求帮助是勇敢的行为，心理健康同样重要。",
                        "您可以联系当地的心理健康热线或服务机构获取支持。",
                        "紧急情况下，请拨打心理援助热线寻求即时帮助。"
                    ]
                }
                
                # 使用 get() 方法提供默认值，防止键不存在
                for advice in cluster_advice.get(cluster, cluster_advice[2]):  # 默认使用最严重情况的建议
                    st.markdown(f"- {advice}")
                
                # ... [其余代码保持不变] ...
                
            except Exception as e:
                st.error(f"处理出错: {str(e)}")
                st.info("请确保已回答所有问题并重新提交")

# 管理员数据查看
if 'admin' in st.session_state and st.session_state.admin:
    conn = sqlite3.connect('scl90_data.db')
    df = pd.read_sql('SELECT * FROM responses', conn)
    conn.close()
    
    st.subheader("所有问卷数据")
    st.dataframe(df)
    
    # 添加数据分析功能
    st.subheader("数据分析")
    if not df.empty:
        st.write(f"总收集量: {len(df)}")
        st.bar_chart(df['cluster'].value_counts())
        
        # 导出数据
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "导出数据为CSV",
            data=csv,
            file_name='scl90_responses.csv',
            mime='text/csv'
        )