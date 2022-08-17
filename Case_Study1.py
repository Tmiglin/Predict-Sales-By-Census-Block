# %%
# =============================================================================
# # A Darden brand wishes to expand into areas without existing locations. 
# # They have asked you to evaluate their current business in order to choose where to expand.
# # You have pulled census and housing data at the county level, and joined with the average sales for that brand in each county. 
# # You have split the data into two datasets:
# 
# # 1.	Build a model to predict sales for a given county
# # 2.	How did you select your variables?
# # 3.	How well does your model perform?
# # 4.	Which counties do you recommend expanding into first?
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,RepeatedKFold,GridSearchCV
from sklearn import metrics

train = pd.read_csv("DRI_AdvancedAnalytics_CaseStudy\case1_existing.csv")
predict = pd.read_csv("DRI_AdvancedAnalytics_CaseStudy\case1_new.csv")
predict['sales'] = 0
predict['training_set'] = False
train['training_set'] = True
fulldf = pd.concat([predict,train])
fulldf.dtypes

# =============================================================================
# Initial Observations:
#     - households and number of househoulds seem to be the same data point, assuming the data point came from two sources
#     is it better to take the average of the two?
#     - All of these fields are numeric with different scales, data needs to be scaled
#     - CountyID is the unique identifier
# =============================================================================
# %%
# Data Transformations necessary for modeling (Analysis of Fields)

fulldf['homes'] = ((fulldf['households']+fulldf['number_of_homes'])/2).astype('int64')
fulldf.drop('number_of_homes',axis=1,inplace=True)
fulldf.drop('households',axis=1,inplace=True)
train = fulldf[fulldf['training_set']==True]
corr = train.corr()
sns.heatmap(corr)
# %% 
# Check skewness in predictors and target variable
skewness = {i:train[i].skew() for i in list(train.columns)}

plt.figure()
sns.distplot(train['population']).set_title('Distribution of population field')
pop = np.log(train['population'])
pop.skew()
plt.figure()
sns.distplot(pop).set_title('Distribution of population field after transformation')

# train.drop('families',axis=1,inplace=True)
# train.drop('homes',axis=1,inplace=True)

newskewness = {i:{} for i in list(train.columns)}
for i,v in skewness.items():
    if v > 1 or v < -1:
        try:
            newskewness[i]['base'] = train[i].skew()
            newskewness[i]['log'] = np.log(train[i]).skew()
            newskewness[i]['sqrt'] = np.sqrt(train[i]).skew()
        except:
            pass

def feature_transform(df):
    df['population'] = np.log(df['population'])
    df['families'] = np.log(df['families'])
    df['homes'] = np.log(df['homes'])
    df['median_home_value'] = np.log(df['median_home_value'])
    df['pct_12_24_yrs'] = np.sqrt(df['pct_12_24_yrs'])
    df['pct_65_plus'] = np.sqrt(df['pct_65_plus'])
    df['vacancy_rate'] = np.log(df['vacancy_rate'])
    
    return df

train = feature_transform(train)

# =============================================================================
# The goal of this cell is to fix the skewness of each fields distribution
# We want each field to have a chance to have a positive correlation with
# the sales field
# =============================================================================
# %%
# Bring Data to a common scale and check 
model_vars = train.drop('county_id',axis=1).drop('training_set',axis=1)
scaler = MinMaxScaler()

scaled_pieces = scaler.fit_transform(model_vars)
scaled_df = pd.DataFrame(scaled_pieces, columns=model_vars.columns)
newcorr = scaled_df.corr()
k=8
cols = newcorr.nlargest(k, 'sales')['sales'].index
cm = np.corrcoef(train[cols].values.T)
sns.heatmap(
    cm, 
    cbar=True, 
    annot=True, 
    square=True, 
    fmt='.2f', 
    annot_kws={'size': 10}, 
    yticklabels=cols.values, 
    xticklabels=cols.values
)

# =============================================================================
# Looking at the new correlation numbers, the same fields look to be having
# a positive correlation, however not many fields are having a major
# positive impact. Since the population based fields show multicollinearity
# it is not in the best practice to use all three.
# =============================================================================
# %%
scaled_df = scaled_df.drop('families',axis=1).drop('homes',axis=1)
y = scaled_df['sales']
X = scaled_df[list(scaled_df.columns)[:-1]]

grid = {}
grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
grid['l1_ratio'] = np.arange(0, 1, 0.01)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
model = ElasticNet()

search = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
results = search.fit(X,y)

# =============================================================================
# We are using an elastic net regression because it allows a balance of both 
# penalties, which can result in better performance than a model with 
# either one or the other penalty on some problems. The hyperparameters
# are tuned with a gridsearch over k-fold-validations
# =============================================================================
# %%
feature_search = ElasticNet(alpha = results.best_params_['alpha'],l1_ratio = results.best_params_['l1_ratio'])
feature_search.fit(X,y)

# =============================================================================
# Lets see how the model reacts with all of the features being used with the best
# pair of hyperparameters. We will then look at the coefficients from the regression ]
# and eliminate any of the coefficients that don't have a positive relationship
# =============================================================================

coef = pd.Series(feature_search.coef_, index = X.columns)
positive_features = [i for i in coef.index if coef[i]>0.01]
X_trim = scaled_df[positive_features]
results_trim = search.fit(X_trim,y)
print(f"Best RMSE : {-results.best_score_}")
print(f"Best Parameters : {results.best_params_}")

X_train,X_test,y_train,y_test = train_test_split(X_trim,y,test_size=.25)

final_model = ElasticNet(alpha = results_trim.best_params_['alpha'],l1_ratio = results_trim.best_params_['l1_ratio'])
final_model.fit(X_train,y_train)
y_pred = final_model.predict(X_test)

# =============================================================================
# 0.066 final RMSE for the final model, not the most ideal score
# however, the feature set struggles a bit.
# =============================================================================
print(f"RMSE : {np.sqrt(metrics.mean_squared_error(y_pred,y_test))}")
# %%

labels = fulldf[fulldf['training_set']==False]['county_id']
newlocdf = feature_transform(fulldf[fulldf['training_set']==False])[list(X_trim.columns)]
# We will treat negative infinity as converging to zero
newlocdf['median_home_value'][newlocdf['median_home_value']<-1E308] = 0 
newlocdf_scaled = scaler.fit_transform(newlocdf)
newlocinput = pd.DataFrame(newlocdf_scaled,columns = list(newlocdf.columns))


new_pred = final_model.predict(newlocinput)
sales_opp = pd.DataFrame(new_pred.T,columns=['Sales'])
sales_opp['county_id'] = labels

# Can use this line of code to return the most fruitful county opportunities
sales_opp.sort_values('Sales',ascending=False).head(20)
