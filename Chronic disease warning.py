import pandas as pd  # 导入pandas库，用于数据处理和分析
from sklearn.model_selection import train_test_split  # 导入train_test_split函数，用于将数据集拆分为训练集和测试集
from sklearn.linear_model import LogisticRegression  # 导入LogisticRegression类，用于构建逻辑回归模型
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc  # 导入用于评估模型性能的函数
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
import seaborn as sns  # 导入seaborn库，用于更美观的绘图
import matplotlib.font_manager as fm  # 导入字体管理器

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取CSV文件
data_path = r'C:\Users\GUOXI\Desktop\邮件\heart.csv'  # 指定CSV文件的路径
data = pd.read_csv(data_path)  # 读取CSV文件并存储到data DataFrame中

# 显示数据的前几行
print("数据前几行:")
print(data.head())  # 打印数据的前五行，以检查数据是否正确读取

# 显示列名
print("列名:")
print(data.columns)  # 打印数据集的列名，检查列名是否正确

# 检查数据的基本信息
print("数据基本信息:")
print(data.info())  # 打印数据集的基本信息，包括每列的数据类型和非空值数量

# 数据预处理，例如处理缺失值
print("处理缺失值前的数据形状:", data.shape)  # 打印数据处理缺失值前的形状
data = data.dropna()  # 移除包含缺失值的行
print("处理缺失值后的数据形状:", data.shape)  # 打印数据处理缺失值后的形状

# 特征选择和提取
 # 根据实际的标签列名 'Outcome' 选择特征和标签
X = data.drop('Outcome', axis=1)  # 选择特征列，移除标签列 'Outcome'
y = data['Outcome']  # 选择标签列 'Outcome'

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 将数据集按 80% 训练集和 20% 测试集进行拆分

# 打印训练集和测试集的形状
print("训练集形状:", X_train.shape, y_train.shape)  # 打印训练集的形状
print("测试集形状:", X_test.shape, y_test.shape)  # 打印测试集的形状

# 构建逻辑回归模型
model = LogisticRegression(max_iter=200)  # 初始化逻辑回归模型，并设置最大迭代次数为 200，确保模型收敛

# 训练模型
model.fit(X_train, y_train)  # 使用训练集训练逻辑回归模型

# 在测试集上进行预测
y_pred = model.predict(X_test)  # 在测试集上进行预测，得到预测标签
y_pred_prob = model.predict_proba(X_test)[:, 1]  # 获取正类的预测概率，用于绘制ROC曲线

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)  # 计算模型的预测准确率
print("预测准确率: {:.2f}%".format(accuracy * 100))  # 打印预测准确率，以百分比形式显示

# 生成混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)  # 计算混淆矩阵

# 打印混淆矩阵
print("混淆矩阵:")
print(conf_matrix)

# 生成分类结果报告
class_report = classification_report(y_test, y_pred)  # 生成分类结果报告
print("分类结果报告:")
print(class_report)  # 打印分类结果报告

# 绘制混淆矩阵的热力图
plt.figure(figsize=(10, 7))  # 设置图形大小
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')  # 使用Seaborn绘制热力图
plt.title('混淆矩阵')  # 设置图形标题
plt.xlabel('预测标签')  # 设置X轴标签为
plt.ylabel('真实标签')  # 设置Y轴标签为
plt.show()  # 显示图形

# 绘制测试集特征分布图
plt.figure(figsize=(12, 10))  # 设置图形大小
for i, column in enumerate(X_test.columns, 1):
    plt.subplot(3, 3, i)  # 创建子图
    sns.histplot(X_test[column], kde=True)  # 绘制特征分布图
    plt.title(column)  # 设置子图标题
plt.tight_layout()  # 调整子图布局
plt.show()  # 显示图形

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)  # 计算假阳性率和真阳性率
roc_auc = auc(fpr, tpr)  # 计算AUC值

plt.figure(figsize=(8, 6))  # 设置图形大小
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC曲线 (面积 = %0.2f)' % roc_auc)  # 绘制ROC曲线
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # 绘制对角线
plt.xlim([0.0, 1.0])  # 设置X轴范围
plt.ylim([0.0, 1.05])  # 设置Y轴范围
plt.xlabel('假阳性率')  # 设置X轴标签
plt.ylabel('真阳性率')  # 设置Y轴标签
plt.title('接收者操作特征 (ROC) 曲线')  # 设置图形标题
plt.legend(loc="lower right")  # 设置图例位置
plt.show()  # 显示图形
