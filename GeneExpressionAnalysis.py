#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-learn')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


# In[3]:


actual_data_path = "actual.csv"
independent_data_path = "data_set_ALL_AML_independent.csv"
train_data_path = "data_set_ALL_AML_train.csv"


# In[4]:


actual_data = pd.read_csv(actual_data_path)
independent_data = pd.read_csv(independent_data_path)
train_data = pd.read_csv(train_data_path)


# In[5]:


actual_data.tail(5)


# In[6]:


actual_data.dtypes


# In[7]:


actual_data.head(5)


# In[8]:


actual_data.shape


# In[9]:


actual_data.nunique


# In[15]:


actual_data.describe()


# In[10]:


actual_data.columns


# In[11]:


actual_data.info()


# In[12]:


plt.figure(figsize=(20,20))
sns.pairplot(data=actual_data,hue='cancer')
plt.tight_layout()


# In[13]:


actual_data.hist(bins=60, figsize=(10,5))
plt.suptitle('Patient', x=0.5, y=1.02, ha='center', fontsize='large')
plt.tight_layout()


# In[14]:


from scipy import stats
z_scores = np.abs(stats.zscore(actual_data['patient']))
threshold = 3
actual_data_no_outliers = actual_data[(z_scores < threshold)]


# In[15]:


plt.figure(figsize=(12, 6))
sns.boxplot(data=[actual_data['patient'], actual_data_no_outliers['patient']], orient='h', palette=['red', 'blue'])
plt.title("Effect of Outliers")
plt.xticks([0, 1], ['Original', 'Cleaned'])
plt.show()


# In[16]:


actual_data = actual_data.isnull().sum()
print("Missing Data:")
print(actual_data)


# In[17]:


actual_data['cancer'] = actual_data['cancer'].astype(str)


# In[18]:


print(actual_data['cancer'].dtype)


# In[19]:


actual_data = pd.read_csv(actual_data_path, dtype={'cancer': str})


# In[20]:


cancer_counts = actual_data['cancer'].value_counts()
print("Cancer Types Distribution:")
print(cancer_counts)


# In[21]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=actual_data, x='cancer', y='patient')
plt.title("Cancer Types and Patient Relationship")
plt.show()


# In[22]:


actual_data = pd.get_dummies(actual_data, columns=['cancer'], drop_first=True)


# In[23]:


correlation_matrix = actual_data.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Between Features")
plt.show()


# In[24]:


cancer1_data = actual_data[actual_data['cancer_AML'] == 0]
cancer2_data = actual_data[actual_data['cancer_AML'] == 1]


# In[25]:


aml_stats = cancer1_data.describe()
all_stats = cancer2_data.describe()


# In[26]:


gene_expression1 = cancer1_data['cancer_AML']
gene_expression2 = cancer2_data['cancer_AML']


# In[27]:


t_statistic, p_value = stats.ttest_ind(gene_expression1, gene_expression2)


# In[28]:


if p_value < 0.05:
    print("The difference in gene expression between the two types of cancer is statistically significant.")
else:
    print("The difference in gene expression between the two types of cancer is not statistically significant.")


# In[29]:


X = actual_data.drop(columns=['cancer_AML'])
y = actual_data['cancer_AML']


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[31]:


from sklearn.discriminant_analysis import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[32]:


rf_model = RandomForestClassifier(random_state=42)


# In[33]:


rf_model.fit(X_train, y_train)


# In[34]:


# Make predictions of the model
y_pred = rf_model.predict(X_test)

# Calculate the accuracy and classification of the report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(report)


# In[35]:


from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[37]:


y_pred = rf_model.predict(X_test)


# In[38]:


cm = confusion_matrix(y_test, y_pred)


# In[39]:


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['ALL', 'AML'], yticklabels=['ALL', 'AML'])
plt.xlabel('Estimated')
plt.ylabel('Real')
plt.title("Confusion Matrix")
plt.show()


# In[40]:


from sklearn.decomposition import PCA
X = actual_data.drop(columns=['cancer_AML'])
y = actual_data['cancer_AML']


# In[41]:


pca = PCA(n_components=min(X.shape[0], X.shape[1]))
X_pca = pca.fit_transform(X)


# In[42]:


pca_df = pd.DataFrame({'Principal Component 1': X_pca[:, 0]})
pca_df['cancer_AML'] = y

# Visualize PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='Principal Component 1', y=np.zeros_like(X_pca[:, 0]), hue='cancer_AML', palette='viridis')
plt.title("PCA Results")
plt.xlabel("Principal Component 1")
plt.show()


# In[43]:


# Select a gene name that is present in your dataset (for example, the first ranked gene name)
selected_gene = X.columns[0]

# Profile the expression of the selected gene
gene_expression = X[selected_gene]

# Create a line graph showing the gene expression profile by cancer types
plt.figure(figsize=(10, 6))
sns.lineplot(data=pd.DataFrame({'Gene Expression': gene_expression, 'Cancer Type': y}), x='Cancer Type', y='Gene Expression')
plt.title(f'Gen İfade Profili: {selected_gene}')
plt.xlabel('Types of Cancer')
plt.ylabel('Gene Expression Value')
plt.xticks(rotation=45)
plt.show()


# In[44]:


# Enrichment p-values of some genes as an example
enrichment_p_values = [0.01, 0.05, 0.1, 0.2]

# Draw a bar chart showing the enrichment p-values
plt.figure(figsize=(10, 6))
sns.barplot(x=['Gene 1', 'Gene 2', 'Gene 3', 'Gene 4'], y=enrichment_p_values, palette='viridis')
plt.title('Gene Enrichment Analysis')
plt.xlabel('Genes')
plt.ylabel('Enrichment p-values')
plt.xticks(rotation=45)
plt.show()


# In[45]:


from sklearn.metrics import roc_curve, auc
y_true = np.random.randint(2, size=100)
y_scores = np.random.rand(100)


# In[46]:


fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)


# In[47]:


plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[48]:


independent_data.tail(5)


# In[49]:


independent_data.dtypes


# In[50]:


independent_data.head()


# In[51]:


independent_data.shape


# In[52]:


independent_data.nunique


# In[53]:


independent_data.describe()


# In[54]:


independent_data.columns


# In[55]:


independent_data.info()


# In[56]:


independent_data = pd.get_dummies(independent_data, drop_first=True)


# In[57]:


scaler = StandardScaler()


# In[58]:


scaled_X = scaler.fit_transform(independent_data)


# In[59]:


model = PCA(n_components=2)


# In[60]:


principal_components = model.fit_transform(scaled_X)


# In[61]:


plt.figure(figsize=(10,6))
plt.scatter(principal_components[:,0],principal_components[:,1])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# In[62]:


model.n_components


# In[63]:


model.components_


# In[64]:


df_comp = pd.DataFrame(model.components_,index=['PC1','PC2'],columns=independent_data.columns)


# In[65]:


df_comp


# In[66]:


model.explained_variance_ratio_


# In[67]:


np.sum(model.explained_variance_ratio_)


# In[68]:


pca_30 = PCA(n_components=30)
pca_30.fit(scaled_X)


# In[69]:


pca_30.explained_variance_ratio_


# In[70]:


np.sum(pca_30.explained_variance_ratio_)


# In[102]:


explained_variance = []

for n in range(1,30):
    pca = PCA(n_components=n)
    pca.fit(scaled_X)
    
    explained_variance.append(np.sum(pca.explained_variance_ratio_))


# In[103]:


plt.plot(range(1,30),explained_variance)
plt.xlabel("Number of Components")
plt.ylabel("Variance Explained");


# In[77]:


train_data.tail(5)


# In[78]:


train_data.dtypes


# In[79]:


train_data.head()


# In[80]:


train_data.shape


# In[81]:


train_data.nunique


# In[82]:


train_data.describe()


# In[83]:


train_data.columns


# In[84]:


train_data.info()


# In[85]:


train_data = pd.get_dummies(train_data, drop_first=True)


# In[86]:


scaler = StandardScaler()


# In[87]:


scaled_Y = scaler.fit_transform(train_data)


# In[88]:


model = PCA(n_components=2)


# In[89]:


principal_components1 = model.fit_transform(scaled_Y)


# In[90]:


plt.figure(figsize=(10,6))
plt.scatter(principal_components1[:,1],principal_components1[:,0])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# In[91]:


model.n_components


# In[92]:


model.components_


# In[93]:


df_comp1 = pd.DataFrame(model.components_,index=['PC1','PC2'],columns=train_data.columns)


# In[94]:


df_comp1


# In[95]:


model.explained_variance_ratio_


# In[96]:


np.sum(model.explained_variance_ratio_)


# In[97]:


pca_50 = PCA(n_components=50)
pca_50.fit(scaled_Y)


# In[98]:


pca_50.explained_variance_ratio_


# In[99]:


np.sum(pca_50.explained_variance_ratio_)


# In[100]:


explained_variance = []

for n in range(1,30):
    pca = PCA(n_components=n)
    pca.fit(scaled_Y)
    
    explained_variance.append(np.sum(pca.explained_variance_ratio_))


# In[101]:


y_true = np.random.randint(2, size=100)
y_scores = np.random.rand(100)

