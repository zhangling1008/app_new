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

# 1. 加载模型和数据
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

# 2. 生成二维码的函数
def generate_qr_code(url):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    buffered = BytesIO()
    img.save(buffered)
    return base64.b64encode(buffered.getvalue()).decode()

# 3. 获取本地IP地址
def get_local_ip():
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return local_ip
    except:
        return "localhost"

# 4. 完整的SCL-90问卷题目
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

# 5. 创建问卷界面
st.title("SCL-90心理健康自评量表")

# 在侧边栏添加二维码
with st.sidebar:
    st.subheader("手机扫码填写问卷")
    
    # 获取当前访问地址
    port = 8501  # Streamlit默认端口
    local_ip = get_local_ip()
    access_url = f"http://{local_ip}:{port}"
    
    # 生成并显示二维码
    qr_img = generate_qr_code(access_url)
    st.markdown(f'<img src="data:image/png;base64,{qr_img}" width="150">', 
               unsafe_allow_html=True)
    st.caption(f"扫描二维码访问: {access_url}")
    st.write("或分享此链接给他人")

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
                
                # 显示结果
                st.success("评估完成！")
                
                descriptions = {
                    0: "您的心理健康状况良好",
                    1: "存在轻度心理困扰",
                    2: "建议寻求专业心理帮助"
                }
                
                st.write(f"​**​评估结果​**​: {descriptions[cluster]}")
                
                # 可视化
                fig, ax = plt.subplots()
                all_embeddings = np.load('embeddings.npy')
                for i in range(3):
                    points = all_embeddings[kmeans.labels_ == i]
                    ax.scatter(points[:,0], points[:,1], label=f'群体 {i}', alpha=0.5)
                ax.scatter(embedding[0,0], embedding[0,1], c='red', s=100, label='您的位置')
                ax.legend()
                ax.set_title("您在群体中的分布位置")
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"处理出错: {str(e)}")
                st.info("请确保已回答所有问题并重新提交") 