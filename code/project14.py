import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

# TODO: read data and overview of data
data_math = pd.read_csv("../data/Maths.csv")
data_port = pd.read_csv("../data/Portuguese.csv")

# convert all object columns to categorical
for data in (data_math, data_port):
    list_str_obj_cols = data.columns[data.dtypes == "object"].tolist()
    for str_obj_col in list_str_obj_cols:
        data[str_obj_col] = data[str_obj_col].astype("category")

print(data_math.info(verbose=True))
print(data_port.info(verbose=True))

data_math.name = 'data_math'  #that you can print the name in loops
data_port.name = 'data_port'

# number of rows and columns
for data in (data_math, data_port):
    r,c = data.shape
    print(f'{data.name} contains {r} rows (students) and {c} columns.')

# outliers for all numerical columns
columns_1_to_5 = ['famrel', 'freetime', 'Dalc', 'Walc', 'goout', 'health']
columns_0_to_20 = ['G1', 'G2', 'G3']
columns_0_to_4 = ['Medu', 'Fedu', 'failures']
columns_1_to_4 = ['traveltime', 'studytime']
columns_0_to_93 = ['absences']
columns_15_to_22 = ['age']
for j in [data_math, data_port]:
    for i in columns_1_to_5:
        outliers = j[i][j[i] > 5].count() + j[i][j[i] < 1].count()
        print(f'Number of values outside the range for {i} in {j.name}: {outliers}.')
    for i in columns_0_to_20:
        outliers = j[i][j[i] > 20].count() + j[i][j[i] < 0].count()
        print(f'Number of values outside the range for {i} in {j.name}: {outliers}.')
    for i in columns_0_to_4:
        outliers = j[i][j[i] > 4].count() + j[i][j[i] < 0].count()
        print(f'Number of values outside the range for {i} in {j.name}: {outliers}.')
    for i in columns_1_to_4:
        outliers = j[i][j[i] > 4].count() + j[i][j[i] < 1].count()
        print(f'Number of values outside the range for {i} in {j.name}: {outliers}.')
    for i in columns_0_to_93:
        outliers = j[i][j[i] > 93].count() + j[i][j[i] < 0].count()
        print(f'Number of values outside the range for {i} in {j.name}: {outliers}.')
    for i in columns_15_to_22:
        outliers = j[i][j[i] > 22].count() + j[i][j[i] < 15].count()
        print(f'Number of values outside the range for {i} in {j.name}: {outliers}.')

#proof, if G3, Walc and Dalc are normally distributed
alpha = 0.05
for i in [data_math, data_port]:
    for j in ['G3', 'Walc', 'Dalc']:
        s, p = sts.shapiro(i[j])
        if p < alpha:
            print(f'{j} in {i.name} is NOT normally distributed. P-value = {p}')
        else:
            print(f'{j} in {i.name} is normally distributed. P-value = {p}')

# correlation evaluation spearman (G3 and alc)
print(
    f"Spearman Rank Correlation between final math grade and Dalc: {sts.spearmanr(data_math['G3'], data_math['Dalc']).correlation: .3f}")
print(
    f"Spearman Rank Correlation between final math grade and Walc: {sts.spearmanr(data_math['G3'], data_math['Walc']).correlation: .3f}")
print(
    f"Spearman Rank Correlation between final port grade and Dalc: {sts.spearmanr(data_port['G3'], data_port['Dalc']).correlation: .3f}")
print(
    f"Spearman Rank Correlation between final port grade and Walc: {sts.spearmanr(data_port['G3'], data_port['Walc']).correlation: .3f}")

# TODO: data pre-processing
#add a column, where the outcome is binary
for data in (data_math, data_port):
    data['G3 0_1'] = np.where(data['G3']>9, 1, 0)

#feature selection
def FeatureSelection (X, y):
    #test for multicollinearity
    vif_scores = pd.DataFrame()
    vif_scores["Attribute"] = X.columns

    # calculating VIF for each feature
    vif_scores["VIF Scores"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    # print(vif_scores)

    vif_filtered = vif_scores[vif_scores['VIF Scores'] < 10]
    cols = list(vif_filtered['Attribute'].unique())
    # print(cols)
    X = X[cols]

    # Perform Recursive Feature Elimination (RFE) with RandomForestClassifier --> GOOD for SVM and linear regression
    k = 15

    model = RandomForestClassifier(random_state=42)  # You can choose any other machine learning algorithm
    rfe = RFE(estimator=model, n_features_to_select=k)
    rfe.fit_transform(X, y)
    selected_features_rfe = X.columns[rfe.support_].tolist()
    X = X[selected_features_rfe]

    return X

#pre-processing LogR, RF and SVM
data_math_lrs = data_math.copy()
data_port_lrs = data_port.copy()
for data in (data_math_lrs, data_port_lrs):
    data = data.drop(columns= ['G3', 'G2', 'G1'], inplace = True)

num_cols_math_lrs = data_math_lrs.select_dtypes(include=['int', 'float']).columns.values
cat_cols_math_lrs = data_math_lrs.select_dtypes(include=['category']).columns.values
cat_cols_math_lrs = np.delete(cat_cols_math_lrs, np.where(cat_cols_math_lrs == 'G3 0_1'))
#(cat_cols_math_lrs)
num_cols_port_lrs = data_port_lrs.select_dtypes(include=['int', 'float']).columns.values
cat_cols_port_lrs = data_port_lrs.select_dtypes(include=['category']).columns.values
cat_cols_port_lrs = np.delete(cat_cols_port_lrs, np.where(cat_cols_port_lrs == 'G3 0_1'))

data_math_lrs_enc = pd.get_dummies(data_math_lrs, columns=cat_cols_math_lrs, prefix=cat_cols_math_lrs, drop_first=True, dtype=float)
data_port_lrs_enc = pd.get_dummies(data_port_lrs, columns=cat_cols_port_lrs, prefix=cat_cols_port_lrs, drop_first=True, dtype=float)
#print(data_math_lrs_enc)

X_math_lrs = data_math_lrs_enc.drop('G3 0_1', axis = 1)
y_math_lrs = data_math_lrs_enc['G3 0_1']
X_port_lrs = data_port_lrs_enc.drop('G3 0_1', axis = 1)
y_port_lrs = data_port_lrs_enc['G3 0_1']

X_math_lrs = FeatureSelection(X_math_lrs, y_math_lrs)
X_port_lrs = FeatureSelection(X_port_lrs, y_port_lrs)

print()
print('The 15 most important features for LogR, RF, SVM in data_math:')
print(X_math_lrs.columns)
print()
print('The 15 most important features for LogR, RF, SVM in data_port:')
print(X_port_lrs.columns)

#evaluation metrics
def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

def evaluation_metrics (clf, y, X, ax, legend_entry='my legendEntry'):
    y_test_pred = clf.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_test_pred).ravel()

    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    # Get the roc curve using a sklearn function
    y_test_predict_proba = clf.predict_proba(X)[::, 1]
    fp_rates, tp_rates, _ = roc_curve(y, y_test_predict_proba)

    # Calculate the area under the roc curve using a sklearn function
    roc_auc = auc(fp_rates, tp_rates)

    # Plot on the provided axis
    if legend_entry != ' ':
        ax.plot(fp_rates, tp_rates, label = legend_entry)
    else:
        ax.plot(fp_rates, tp_rates)

    return [accuracy, precision, recall, specificity, f1, roc_auc]

#pre-processing LinR
data_math_lin = data_math.copy()
data_port_lin = data_port.copy()

num_cols_lr = data_math_lin.select_dtypes(include=['int', 'float']).columns.values
cat_cols_lr = data_math_lin.select_dtypes(include=['category']).columns.values

data_math_lin_enc = pd.get_dummies(data=data_math_lin, columns=cat_cols_lr, drop_first=True, dtype=float)
data_port_lin_enc = pd.get_dummies(data=data_port_lin, columns=cat_cols_lr, drop_first=True, dtype=float)

X_math_lin = data_math_lin_enc.drop(columns=['G3', 'G2', 'G1', 'G3 0_1'], axis=1)
y_math_lin = data_math_lin_enc['G3']
X_port_lin = data_port_lin_enc.drop(columns=['G3', 'G2', 'G1', 'G3 0_1'], axis=1)
y_port_lin = data_port_lin_enc['G3']

X_math_lin = FeatureSelection(X_math_lin, y_math_lin)
X_port_lin = FeatureSelection(X_port_lin, y_port_lin)

print()
print('The 15 most important features for LinR in data_math:')
print(X_math_lin.columns)
print()
print('The 15 most important features for LinR in data_port:')
print(X_port_lin.columns)

# TODO: visualization
# visualisation of label(G3)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].set_xlim(left=-2, right=22)
axes[1].set_xlim(left=-2, right=22)
hist_G3_m = sns.histplot(
    x=data_math['G3'], data=data_math, bins=20, ax=axes[0], color='steelblue'
).set(xlabel='Grade', ylabel='counts', title='data_math')

hist_G3_p = sns.histplot(
    x=data_port['G3'], data=data_port, bins=20, ax=axes[1], color='steelblue'
).set(xlabel='Grade', ylabel='counts', title='data_port')
plt.suptitle('Distribution of G3', fontsize=18)
plt.savefig('../output/G3_distribution.png')

# visualisation of the alcohol features (Dalc and Walc)
fig, axes = plt.subplots(1, 4, figsize=(12, 6))

fig_Dalc_m = sns.histplot(data_math, x='Dalc', ax=axes[0], color='steelblue')
Dalc_m_axes = fig_Dalc_m.set(xlabel='workday alcohol consumption', ylabel='counts', title='Dalc in data_math')

fig_Walc_m = sns.histplot(data_math, x='Walc', ax=axes[1], color='steelblue')
Walc_m_axes = fig_Walc_m.set(xlabel='weekend alcohol consumption', ylabel='counts', title='Walc in data_math')

fig_Dalc_p = sns.histplot(data_port, x='Dalc', ax=axes[2], color='steelblue')
Dalc_p_axes = fig_Dalc_p.set(xlabel='workday alcohol consumption', ylabel='counts',
                             title='Dalc in data_port')

fig_Walc_p = sns.histplot(data_port, x='Walc', ax=axes[3], color='steelblue')
Walc_p_axes = fig_Walc_p.set(xlabel='weekend alcohol consumption', ylabel='counts',
                             title='Walc in data_port')

plt.suptitle('Distribution of alcohol consumption', fontsize=18, y=1)
plt.tight_layout()
plt.savefig("../output/alc_distributions.png")

# Boxplot Final Grade by Walc and Dalc
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
bplot1 = sns.boxplot(x="Walc", y="G3", data=data_math, ax=axes[0, 0], color='steelblue')
axes1 = bplot1.set(xlabel='Walc', ylabel='G3', title='data_math by Walc')
bplot2 = sns.boxplot(x="Dalc", y="G3", data=data_math, ax=axes[0, 1], color='steelblue')
axes2 = bplot2.set(xlabel='Dalc', ylabel='G3', title='data_math by Dalc')
bplot3 = sns.boxplot(x="Walc", y="G3", data=data_port, ax=axes[1, 0], color='steelblue')
axes3 = bplot3.set(xlabel='Walc', ylabel='G3', title='data_port by Walc')
bplot4 = sns.boxplot(x="Dalc", y="G3", data=data_port, ax=axes[1, 1], color='steelblue')
axes4 = bplot4.set(xlabel='Dalc', ylabel='G3', title='data_port by Dalc')
plt.subplots_adjust(hspace=0.35)
plt.suptitle('Final grades by alcohol consumption', fontsize=20)
plt.savefig("../output/G3 by alc.png")

# proportion of failed according to Walc und Dalc
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
a = 0
for k in [data_math, data_port]:
    b = 0
    for j in ['Walc', 'Dalc']:
        count = []
        fail_count = []
        for i in range(1, 6):
            count.append(len(k[(k[j] == i)]))
        for i in range(1, 6):
            fail_count.append(len(k[(k[j] == i) & (k['G3 0_1'] == 0)]))
        percentage = []
        for i in range(0, 5):
            percentage.append(fail_count[i] / count[i] * 100)
        perc_plot = sns.barplot(k, x=[1, 2, 3, 4, 5], y=percentage, ax=axes[a, b], color='steelblue')
        axs = perc_plot.set(xlabel=j, ylabel='Percentage of failed G3',
                            title=f'{k.name} depending on {j}')
        b += 1
    a += 1
plt.subplots_adjust(hspace=0.35)
plt.suptitle('Percentages of failed G3', fontsize=18)
plt.savefig("../output/fail_percentages.png")

#Visualisation of G3 0_1
fig, axes = plt.subplots(1, 2, figsize = (12,6))
sns.countplot(x = data_math['G3 0_1'], data=data_math, ax = axes[0], color='steelblue').set(xlabel='pass(1)/fail(0)', title='data_math')
sns.countplot(x = data_port['G3 0_1'], data=data_port, ax = axes[1], color='steelblue').set(xlabel='pass(1)/fail(0)', title='data_port')

plt.suptitle('Distribution of pass and fail', fontsize=18)
plt.savefig('../output/distribution_passfail.png')


#print number of fails/passes
print('Number of fails in data_math:', data_math['G3 0_1'].value_counts()[0])
print('Number of passes in data_math:', data_math['G3 0_1'].value_counts()[1])
print('Number of fails in data_port:', data_port['G3 0_1'].value_counts()[0])
print('Number of passes in data_math:', data_port['G3 0_1'].value_counts()[1])

#visualisation correlation between Walc/Dalc and G3
plot, axs = plt.subplots(1, 4, figsize=(18, 9))

plot1 = sns.scatterplot(data_math, x=data_math['G3'], y=data_math['Dalc'], ax=axs[0], color='lightblue', alpha=0.5)
axs1 = plot1.set(xlabel='G3', ylabel='Dalc', title='correlation between math G3 and Dalc')

plot2 = sns.scatterplot(data_math, x=data_math['G3'], y=data_math['Walc'], ax=axs[1], color='steelblue', alpha=0.5)
axs2 = plot2.set(xlabel='G3', ylabel='Walc', title='correlation between math G3 and Walc')

plot3 = sns.scatterplot(data_port, x=data_port['G3'], y=data_port['Dalc'], ax=axs[2], color='blue', alpha=0.5)
axs3 = plot3.set(xlabel='G3', ylabel='Dalc', title='correlation between port G3 and Dalc')

plot4 = sns.scatterplot(data_port, x=data_port['G3'], y=data_port['Walc'], ax=axs[3], color='grey', alpha=0.5)
axs4 = plot4.set(xlabel='G3', ylabel='Walc', title='correlation between port G3 and Walc')


plt.tight_layout()
plt.savefig("../output/pearson and spearman correlation.png")

#heatplot
data_math_lin_enc.name='data_math'
data_port_lin_enc.name='data_port'
for data in (data_math_lin_enc, data_port_lin_enc):
    # correlation only with numerical columns --> encoded data
    corr = data.corr(method = 'spearman', numeric_only=True)
    corr = abs(corr)
    mask = np.triu(np.ones_like(corr))
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr, annot=False, cmap='YlGnBu', mask=mask)
    plt.title('Correlation Heatmap', fontsize = 30)
    plt.savefig(f'../output/heatmap_{data.name}.png')

#TODO: Linear Regression (LinR)
X_math_lin.name = 'Mathematics'
X_port_lin.name = 'Portuguese'

def linear_regression (X, y):
    # test train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=2023)

    # Standardize the numerical features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # convert pandas DataFrame to numpy array
    X_train, X_test, y_train, y_test = (
        np.array(X_train),
        np.array(X_test),
        np.array(y_train),
        np.array(y_test),
    )

    # initialize the model
    LR = LinearRegression()

    # fit the model
    LR.fit(X=X_train, y=y_train)

    # get intercept and coefficients of the model
    print()
    print(X.name, 'Intercept:', LR.intercept_)
    print(X.name, 'Coefficients:', LR.coef_)

    # Make predictions using the fitted model
    y_pred = LR.predict(X_test)

    # Performance analysis
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("R2 score: {:.3f}, RMSE: {:.3f}".format(r2, rmse))

    columns = X.columns.values.tolist()

    return LR.coef_, columns

coef_math, columns_math = linear_regression(X_math_lin, y_math_lin)
coef_port, columns_port = linear_regression(X_port_lin, y_port_lin)

# visualization of feature importance - despite the bad performance
importance_math_lr = np.abs(coef_math)
importance_scores_math_lr = (importance_math_lr / np.sum(importance_math_lr)).flatten()
importance_math_pd = pd.DataFrame(importance_scores_math_lr, columns=['importance'])
importance_math_pd['features'] = columns_math
importance_math_sort = importance_math_pd.sort_values(by=['importance'], ascending=True)

importance_port_lr = np.abs(coef_port)
importance_scores_port_lr = (importance_port_lr / np.sum(importance_port_lr)).flatten()
importance_port_pd = pd.DataFrame(importance_scores_port_lr, columns=['importance'])
importance_port_pd['features'] = columns_port
importance_port_sort = importance_port_pd.sort_values(by=['importance'], ascending=True)

fig, ax = plt.subplots(2, 1, figsize=(18, 16))
ax[0].bar(importance_math_sort['features'], importance_math_sort['importance'])
ax[0].set_xticks(np.arange(len(importance_scores_math_lr)), importance_math_sort['features'], rotation=90)
ax[0].set_title('Feature importance in data_math')
ax[1].bar(importance_port_sort['features'], importance_port_sort['importance'])
ax[1].set_xticks(np.arange(len(importance_scores_port_lr)), importance_port_sort['features'], rotation=90)
ax[1].set_title('Feature importance in data_port')
plt.subplots_adjust(hspace=0.4)
plt.savefig('../output/feature_importance_linReg')

#TODO: Logistic Regression (LogR)
#Stratified k-fold
X_math_lrs.name='data_math'
X_port_lrs.name='data_port'

def performclf (X, y, clf, ax): #can be used for Random Forest too
    n_splits = 5
    skf = StratifiedKFold(n_splits, shuffle=True, random_state = 42)

    #prepare the data frame to store the performance
    df_performance = pd.DataFrame(columns = ['data', 'fold','clf','accuracy','precision','recall',
                                             'specificity','F1','roc_auc'])
    df_normcoef = pd.DataFrame(index = X.columns, columns = np.arange(n_splits))

    fold =  0

    # Loop over all splits
    for train_index, test_index in skf.split(X,y):

        # Get the relevant subsets for training and testing
        X_test  = X.iloc[test_index]
        y_test  = y.iloc[test_index]
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]

        # Standardize the numerical features using training set statistics
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        #fit
        clf.fit(X_train_sc, y_train)

        # Get the importance values
        try:
            coef = np.abs(clf.coef_)
        except:
            coef = np.abs(clf.feature_importances_)
        df_this_coefs = pd.DataFrame(zip(X_train.columns, np.transpose(coef)), columns=['features', 'coef'])
        df_normcoef.loc[:,fold] = df_this_coefs['coef'] / np.sum(coef).flatten()
        #get mean and std of coefs
        df_mean_std = pd.DataFrame(index=X.columns, columns=['mean', 'std'])
        df_mean_std['mean'] = df_normcoef.mean(axis=1)
        df_mean_std['std'] = df_normcoef.std(axis=1)
        df_mean_std = df_mean_std.sort_values(by='mean', ascending=False)

        # Evaluate your classifiers - ensure to use the correct inputs
        eval_metrics = evaluation_metrics(clf, y_test, X_test_sc, axs[ax],legend_entry='fold ' + str(fold))
        df_performance.loc[len(df_performance),:] = [X.name, fold,'LogR']+eval_metrics

        # increase counter for folds
        fold += 1
    return df_performance, df_mean_std

fig,axs = plt.subplots(1,2,figsize=(9, 4))
clf_LogR = LogisticRegression(random_state=42)

#perform Logistic Regression for data_math
df_performance_LogR_math, df_mean_std_LogR_math = performclf(X_math_lrs, y_math_lrs, clf_LogR, 0)
df_perf_ms_LogR_math = pd.DataFrame(index=df_performance_LogR_math[['accuracy','precision','recall', 'specificity', 'F1', 'roc_auc']].columns, columns=['mean', 'std'])
df_perf_ms_LogR_math['mean'] = df_performance_LogR_math[['accuracy','precision','recall', 'specificity', 'F1', 'roc_auc']].mean()
df_perf_ms_LogR_math['std'] = df_performance_LogR_math[['accuracy','precision','recall', 'specificity', 'F1', 'roc_auc']].std()
print()
print('Performance of Logistic Regression:')
print('data_math:')
print(df_perf_ms_LogR_math)

#perform Logistic Regression for data_port
df_performance_LogR_port, df_mean_std_LogR_port = performclf(X_port_lrs, y_port_lrs, clf_LogR, 1)
df_perf_ms_LogR_port = pd.DataFrame(index=df_performance_LogR_port[['accuracy','precision','recall', 'specificity', 'F1', 'roc_auc']].columns, columns=['mean', 'std'])
df_perf_ms_LogR_port['mean'] = df_performance_LogR_port[['accuracy','precision','recall', 'specificity', 'F1', 'roc_auc']].mean()
df_perf_ms_LogR_port['std'] = df_performance_LogR_port[['accuracy','precision','recall', 'specificity', 'F1', 'roc_auc']].std()
print()
print('data_port:')
print(df_perf_ms_LogR_port)

#plot ROC curves
data_name = ['data_math', 'data_port']
for i, ax in enumerate(axs):
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    add_identity(ax, color='r', ls="--",label = 'random\nclassifier')
    ax.set_title(data_name[i])
    ax.legend()

plt.suptitle('ROC curves Logistic regression', fontsize = 16)
plt.tight_layout()
plt.savefig('../output/roc_curves_LogR_kfold.png')

#plot normalized feature importance
df_mean_std_LogR_math = df_mean_std_LogR_math.head(5)
df_mean_std_LogR_port= df_mean_std_LogR_port.head(5)
fig, ax = plt.subplots(1,2, figsize=(12,6))
#math
ax[0].bar(df_mean_std_LogR_math.index, df_mean_std_LogR_math['mean'], yerr = df_mean_std_LogR_math['std'])
ax[0].set_title('data_math', fontsize=18)
#port
ax[1].bar(df_mean_std_LogR_port.index, df_mean_std_LogR_port['mean'], yerr =df_mean_std_LogR_port['std'])
ax[1].set_title('data_port', fontsize=18)
for i, ax in enumerate(ax):
    ax.set_xlabel('Top 5 features')
    ax.set_ylabel('Normalized feature importance')

plt.suptitle('Feature Importance of Logistic Regression', fontsize = 18)
plt.tight_layout()
plt.savefig('../output/importance_LogR.png')

#TODO: Random Forest (RF)
fig,axs = plt.subplots(1,2,figsize=(9, 4))
clf_RF = RandomForestClassifier(random_state=42)

#perform Logistic Regression for data_math
df_performance_RF_math, df_mean_std_RF_math = performclf(X_math_lrs, y_math_lrs, clf_RF, 0)
df_perf_ms_RF_math = pd.DataFrame(index=df_performance_RF_math[['accuracy','precision','recall', 'specificity', 'F1', 'roc_auc']].columns, columns=['mean', 'std'])
df_perf_ms_RF_math['mean'] = df_performance_RF_math[['accuracy','precision','recall', 'specificity', 'F1', 'roc_auc']].mean()
df_perf_ms_RF_math['std'] = df_performance_RF_math[['accuracy','precision','recall', 'specificity', 'F1', 'roc_auc']].std()
print()
print('Performance of Random Forest:')
print('data_math:')
print(df_perf_ms_RF_math)

#perform Logistic Regression for data_port
df_performance_RF_port, df_mean_std_RF_port = performclf(X_port_lrs, y_port_lrs, clf_RF, 1)
df_perf_ms_RF_port = pd.DataFrame(index=df_performance_RF_port[['accuracy','precision','recall', 'specificity', 'F1', 'roc_auc']].columns, columns=['mean', 'std'])
df_perf_ms_RF_port['mean'] = df_performance_RF_port[['accuracy','precision','recall', 'specificity', 'F1', 'roc_auc']].mean()
df_perf_ms_RF_port['std'] = df_performance_RF_port[['accuracy','precision','recall', 'specificity', 'F1', 'roc_auc']].std()
print()
print('data_port:')
print(df_perf_ms_RF_port)

#plot ROC curves
data_name = ['data_math', 'data_port']
for i, ax in enumerate(axs):
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    add_identity(ax, color='r', ls="--",label = 'random\nclassifier')
    ax.set_title(data_name[i])
    ax.legend()

plt.suptitle('ROC curves Random Forest', fontsize = 16)
plt.tight_layout()
plt.savefig('../output/roc_curves_RF_kfold.png')

#plot normalized feature importance
df_mean_std_RF_math = df_mean_std_RF_math.head(5)
df_mean_std_RF_port= df_mean_std_RF_port.head(5)
fig, ax = plt.subplots(1,2, figsize=(12,6))
#math
ax[0].bar(df_mean_std_RF_math.index, df_mean_std_RF_math['mean'], yerr = df_mean_std_RF_math['std'])
ax[0].set_title('data_math', fontsize=18)
#port
ax[1].bar(df_mean_std_RF_port.index, df_mean_std_RF_port['mean'], yerr =df_mean_std_RF_port['std'])
ax[1].set_title('data_port', fontsize=18)
for i, ax in enumerate(ax):
    ax.set_xlabel('Top 5 features')
    ax.set_ylabel('Normalized feature importance')

plt.suptitle('Feature Importance of Random Forest', fontsize = 18)
plt.tight_layout()
plt.savefig('../output/importance_RF.png')

#TODO: Support Vector Machine (SVM)
#Split the data in train and test
X_math_train, X_math_test, y_math_train, y_math_test = train_test_split(X_math_lrs, y_math_lrs, test_size=0.2, random_state=42)
X_port_train, X_port_test, y_port_train, y_port_test = train_test_split(X_port_lrs, y_port_lrs, test_size=0.2, random_state=42)

#Scale the data
sc = StandardScaler()
X_math_train_sc = sc.fit_transform(X_math_train)
X_math_test_sc = sc.transform(X_math_test)
X_port_train_sc = sc.fit_transform(X_port_train)
X_port_test_sc = sc.transform(X_port_test)
# print(X_math_test_sc)

# Code for SVM
classifier_math = SVC(kernel='rbf', random_state=0,
                      probability=True)
classifier_math.fit(X_math_train_sc, y_math_train)
# Predicting the Test set results
y_pred_svm_math = classifier_math.predict(X_math_test_sc)
cm_svm_math = confusion_matrix(y_math_test, y_pred_svm_math)
# print(cm_svm_math)

classifier_port = SVC(kernel='rbf', random_state=0, probability=True)
classifier_port.fit(X_port_train_sc, y_port_train)
# Predicting the Test set results
y_pred_svm_port = classifier_port.predict(X_port_test_sc)
cm_svm_port = confusion_matrix(y_port_test, y_pred_svm_port)
# determining the precision,recall and f1-score

print('--*--*--*--*--*--*--*--*--')
print('mathematics')
report_svm_math = classification_report(y_math_test, y_pred_svm_math)
print(report_svm_math)
print('portuguese')
report_svm_port = classification_report(y_port_test, y_pred_svm_port)
print(report_svm_port)

print('--*--*--*--*--*--*--*--*--')

#calculate the evaluation metrics with the previous function
#--> ROC curves
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
df_performance = pd.DataFrame(columns=['clf', 'accuracy', 'precision', 'recall',
                                       'specificity', 'F1', 'roc_auc'])
svm_eval_math = evaluation_metrics(classifier_math, y_math_test, X_math_test_sc, axs[0], ' ')
df_performance.loc[len(df_performance), :] = ['classifier_math'] + svm_eval_math
svm_eval_port = evaluation_metrics(classifier_port, y_port_test, X_port_test_sc, axs[1], ' ')
df_performance.loc[len(df_performance), :] = ['classifier_port'] + svm_eval_port

print()
print('Performance of SVM Classifier:')
print(df_performance)
print()

subject = ['Math', 'Port']
for i, ax in enumerate(axs):
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    add_identity(ax, color="r", ls="--", label='random\nclassifier')
    ax.set_title(subject[i])
    ax.legend()
plt.suptitle('ROC curves SVM', fontsize=16)

plt.tight_layout()
plt.savefig('../output/roc_curves_SVM.png')
