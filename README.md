# 机器学习，sklearn

## 01_dataset.ipynb
- 数据集加载:
    - sklearn.datasets.load_*
    - sklearn.datasets.fetch_*
- 数据集划分:
    - sklearn.model_selection.train_test_split

## 02_feature_engineering.ipynb
- 特征提取:
    - 字典特征提取: sklearn.feature_extraction.DictVectorizer
    - 文本特征提取: 
        - 词频: sklearn.feature_extraction.text.CountVectorizer
        - 逆文档频率: sklearn.feature_extraction.text.TfidfVectorizer
- 特征预处理:
    - 归一化: sklearn.preprocessing.MinMaxScaler
    - 标准化: sklearn.preprocessing.StandardScaler
- 特征降维:
    - 低方差过滤: sklearn.feature_selection.VarianceThreshold
    - 主成分分析: sklearn.decomposition.PCA

## 03_pca_example.ipynb
instacart数据集PCA降维实战

## 04_knn.ipynb
- knn算法: sklearn.neighbors.KNeighborsClassifier
- 模型调优: sklearn.model_selection.GridSearchCV

## 05_bayesian.ipynb
- 朴素贝叶斯: sklearn.naive_bayes.MultinomialNB

## 06_decision_tree.ipynb
- 决策树: sklearn.tree.DecisionTreeClassifier
- 树结构: 
    - 导出: sklearn.tree.export_graphviz
    - 查看: http://www.webgraphviz.com/

## 07_random_forest.ipynb
- 随机森林: sklearn.ensemble.RandomForestClassifier

## 08_linear_regression.ipynb
- 线性回归
    - 正规方程: sklearn.linear_model.LinearRegression
    - 随机提取下降: sklearn.linear_model.SGDRegressor
    - 岭回归: sklearn.linear_model.Ridge

## 09_logistic_regression.ipynb
- 逻辑回归: sklearn.linear_model.LogisticRegression

## 10_metrics.ipynb
- 分类的多种评价指标: sklearn.metrics.classification_report
    - 准确率
    - 精确率
    - 召回率
    - F1-score
ROC曲线与AUC指标: sklearn.metrics.roc_auc_score

## 11_kmeans.ipynb
- kmeans聚类算法: sklearn.cluster.KMeans
- 聚类评价指标——轮廓系数: sklearn.metrics.silhouette_score

