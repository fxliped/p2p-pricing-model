import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, log_loss, 
                             classification_report, 
                             RocCurveDisplay)
from sklearn.model_selection import RandomizedSearchCV


# --- Data Loading ---
listings = pd.read_csv('data/listings.csv')
pricing = pd.read_csv('data/pricing_log.csv')
outcomes = pd.read_csv('data/outcomes.csv')

# --- EDA ---
df = listings.merge(pricing, on = 'listing_id').merge(outcomes, how = 'left', on = 'listing_id')

df.head()
df.info()
df.isnull().sum()
df['sold_14d'].value_counts(normalize = True)
df['sold_14d'].value_counts().plot(kind = 'bar')
plt.show()

# Need to check distribution for days_to_sell (3168 nulls)
    # Won't matter because we are dropping it, per project specifications



# -- Data Cleaning & Feature Engineering --



# First Visualize the Variables

# Keep only features available at listing time
    # But keep response variable, sold_14d

df.drop(columns=['days_to_sell', 'gmv_14d', 'platform_revenue_14d', 'seller_payout_14d'], inplace=True)

num_cols = df.select_dtypes(include = 'number').columns.drop('sold_14d')
cat_cols = df.select_dtypes(include = 'object').columns

for col in num_cols:
    sns.histplot(data = df, x = col, hue = 'sold_14d', kde = True, bins = 30)
    plt.title(f'{col} by sold_14d'.upper())
    plt.show()

for col in cat_cols:
    sell_rate = df.groupby(col)['sold_14d'].mean().sort_values()
    sell_rate.plot(kind = 'barh')
    plt.title(f'sell rate by {col}'.upper())
    plt.show()

# seller_rating skewed heavily to 5, otherwise relatively gaussian
# est_value & list_price are skewed right -> needs log transform
    # log transformations still look bad

df['log_price_ratio'] = np.log(df['list_price'] / df['est_value'])

pt = PowerTransformer(method = 'box-cox', standardize = True)
pt.fit_transform(df[['list_price', 'est_value']])
print(f"Optimal Lambda: {pt.lambdas_}")

# Manually apply box-cox formula to skewed variables

#df['transformed_list_price'] = (df['list_price']**pt.lambdas_[0]- 1) / pt.lambdas_[0]
#df['transformed_est_value'] = (df['est_value']**pt.lambdas_[1]- 1) / pt.lambdas_[1]
#df.drop(columns = ['list_price', 'est_value'], axis = 1, inplace = True)

num_cols = df.select_dtypes(include = 'number').columns.drop('sold_14d')
cat_cols = df.select_dtypes(include = 'object').columns

# Visualize variables after transformations
for col in num_cols:
    sns.histplot(data = df, x = col, hue = 'sold_14d', kde = True, bins = 30)
    plt.title(f'{col} by sold_14d'.upper())
    plt.show()

for col in cat_cols:
    sell_rate = df.groupby(col)['sold_14d'].mean().sort_values()
    sell_rate.plot(kind = 'barh')
    plt.title(f'sell rate by {col}'.upper())
    plt.show()




# --- Split Data ---

df['created_at'] = pd.to_datetime(df['created_at'])
train_df = df[df["created_at"] < "2026-03-01"].copy()
test_df = df[df["created_at"] >= "2026-03-01"].copy()
cat_cols = cat_cols.drop('created_at')

print(f"Train Size: {len(train_df)}, Test Size: {len(test_df)}")
print(f"Train Sell Rate: {train_df['sold_14d'].mean():.3f}")
print(f"Test Sell Rate: {test_df['sold_14d'].mean():.3f}")

# -- Feature Scaling & Normalization --

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output = False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])


# -- Feature Selection (Train only) --

# RandomForest importance

X_train = train_df.drop(['sold_14d', 'created_at'], axis = 1)
X_train_encoded = pd.get_dummies(X_train, columns = list(cat_cols))
X_test = test_df.drop(['sold_14d', 'created_at'], axis = 1)
X_test_encoded = pd.get_dummies(X_test.copy(), columns = list(cat_cols))

y_train = train_df['sold_14d']
y_test = test_df['sold_14d']

X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded, join = 'left', axis = 1, fill_value = 0
)

rf_selector = RandomForestClassifier(n_estimators = 100, random_state = 42)

rf_selector.fit(X_train_encoded, y_train)

importances = pd.Series(
    rf_selector.feature_importances_, 
    index = X_train_encoded.columns
).sort_values(ascending = False)
importances.sort_values().plot(kind='barh')
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

# Most important appear to be: quality_score, seller_id, desc_len,
# listing_id, propensity, seller_rating, take_rate, transformed_est_value, 
# transformed_list_price, photo_count, brand_tier, multiplier


# VIF (detect multicollinearity)

X_vif = X_train[num_cols].dropna()
vif = pd.DataFrame({
    'feature': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) 
            for i in range(X_vif.shape[1])] 
})
print(vif.sort_values('VIF', ascending = False))

# NEXT:
    # ACTUALLY SELECT THE BEST FEATURES
# Due to high VIFs for list_price and est_value, lets drop est_value
train_df_clean = train_df.drop(columns=['take_rate', 'multiplier', 'listing_id'])

# Recheck
    # Have lower VIFs w/o the transformed variables, but we perform worse w/o them
X_vif = (train_df_clean
         .replace([np.inf, -np.inf], np.nan)
         .dropna()
         .select_dtypes(include='number')
         .astype(float))
vif = pd.DataFrame({
    'feature': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) 
            for i in range(X_vif.shape[1])] 
})
print(vif.sort_values('VIF', ascending = False))


# told we must keep: `category`, `brand_tier`, `condition`
# - `quality_score`, `est_value`, `list_price`


# --- Model Selection ---


# Logistic Regression
lr_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        class_weight = 'balanced', max_iter=1000, random_state = 42
    ))
])

# Random Forest
rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators = 200, class_weight = 'balanced', random_state = 42
    ))
])

# --- Assess Performance ---

for name, model in [('Logistic Regression', lr_model), 
                    ('Random Forest', rf_model)]:
    model.fit(X_train, y_train) # why not X_train_encoded? (ColumnTransformer and pipeline handles encoding internally)
    y_prob = model.predict_proba(X_test)[:, 1] # what exactly does this do? (returns 2D array with shape (n_samples, 2))
    y_pred = model.predict(X_test)

    print(f"\n--- {name} ---")
    print(f"  AUC-ROC  : {roc_auc_score(y_test, y_prob):.4f}") # why do the .4f? (4 decimal spots appears to be convention for metrics)
    print(f"  Log Loss : {log_loss(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred))

# Logistic Regression AUC-ROC: 0.6010
# Random Forest AUC-ROC: 0.5748

importances.sort_values(ascending = False).head(10)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, model) in zip(axes, [('LR', lr_model), ('RF', rf_model)]):
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.set_title(f'ROC Curve - {name}')
plt.tight_layout()
plt.show()

# -- Sanity Checks -- 

# Check 1: higher price => lower p(sell) holding other features fixed
baseline = X_test.copy().iloc[[0]]
prices = np.linspace(
    X_test['list_price'].quantile(0.1),
    X_test['list_price'].quantile(0.9),
    50
)

probs = []
for p in prices:
    row = baseline.copy()
    row['list_price'] = p
    row['log_price_ratio'] = np.log(p / row['est_value'].values[0])
    probs.append(rf_model.predict_proba(row)[0][1])

plt.plot(prices, probs)
plt.xlabel('List price')
plt.ylabel('P(sold_14d)')
plt.title('Sanity check: price vs. sell probability')
plt.show()

# Elasticity: How sensitive demand is to a 1% change in price
segment = X_test[test_df['category'] == 'Women_Tops'].copy()
price_c = segment['list_price'].copy()
price_c1 = price_c * 1.01

seg_p0 = segment.copy()
seg_p1 = segment.copy()
seg_p1['list_price'] = price_c1
seg_p1['log_price_ratio'] = np.log(
    price_c1 / segment['est_value']
)

p0 = rf_model.predict_proba(seg_p0)[:, 1]
p1 = rf_model.predict_proba(seg_p1)[:, 1]

elasticity = ((p1 - p0) / p0) / 0.01
print(f"\nAverage elasticity (Women_Tops): {elasticity.mean():.4f}")
# -0.4755 elasticity

# -- Tune Hyperparameters / Cross Validation --

param_dist = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 10, None],
    'classifier__min_samples_leaf': [1, 5, 10]
}

search = RandomizedSearchCV(
    rf_model, param_dist,
    n_iter = 10, cv = 5,
    scoring = 'roc_auc',
    random_state = 42, n_jobs = -1
)
search.fit(X_train, y_train)
print("Best params: ", search.best_params_)
print("Best CV AUC: ", search.best_score_)
# Best CV AUC: 0.6026