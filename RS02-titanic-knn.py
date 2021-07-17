
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score #交叉验证
from sklearn.feature_extraction import DictVectorizer

#数据加载
train_data = pd.read_csv('./train.csv') # 返回DataFrame类型
test_data = pd.read_csv('./test.csv')


# 使用平均年龄来填充年龄中的nan值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
# 使用票价的均值填充票价中的nan值
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)

print(train_data['Embarked'].value_counts()) #S最多
# 使用登录最多的港口来填充登录港口的nan值
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S',inplace=True)
# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]
print('特征值')
print(train_features)
# DictVectorizer对非数字化(Embarked、Sex)的特征采用0/1的方式进行量化，而数值型的特征一般情况维持原值即可
dvec=DictVectorizer(sparse=False)
train_features=dvec.fit_transform(train_features.to_dict(orient='record'))
test_features=dvec.transform(test_features.to_dict(orient='record'))
print(dvec.feature_names_)
# 采用Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_features = ss.fit_transform(train_features)
test_ss_features = ss.transform(test_features)

# KNN生存预测
# 创建KNN分类器
knn = KNeighborsClassifier()
knn.fit(train_ss_features, train_labels)

# KNN预测
pred_labels = knn.predict(test_ss_features)

# 得到KNN准确率(基于训练集)
acc_decision_knn = round(knn.score(train_ss_features, train_labels), 6)
print(u'score准确率为 %.4lf' % acc_decision_knn)

# 使用K折交叉验证统计KNN准确率
print(u'cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(knn, train_ss_features, train_labels, cv=10)))




