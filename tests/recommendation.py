import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# --- Data Loading ---
listings = pd.read_csv('data/listings.csv')
pricing = pd.read_csv('data/pricing_log.csv')
outcomes = pd.read_csv('data/outcomes.csv')

df = listings.merge(pricing, on = 'listing_id').merge(outcomes, how = 'left', on = 'listing_id')


df.drop(columns=['days_to_sell', 'gmv_14d', 'platform_revenue_14d', 'seller_payout_14d'], inplace=True)
df['log_price_ratio'] = np.log(df['list_price'] / df['est_value'])


num_cols = df.select_dtypes(include = 'number').columns.drop('sold_14d')
cat_cols = df.select_dtypes(include = 'object').columns



# --- Split Data ---

df['created_at'] = pd.to_datetime(df['created_at'])
train_df = df[df["created_at"] < "2026-03-01"].copy()
test_df = df[df["created_at"] >= "2026-03-01"].copy()
cat_cols = cat_cols.drop('created_at')


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



# Random Forest
rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators = 200, class_weight = 'balanced', random_state = 42
    ))
])



all_cats = sorted(df["category"].unique())
cat_cols = ["category", "condition"]
num_cols = ["brand_tier", "quality_score", "est_value", "list_price", "log_price_ratio"]
interact_cols = [f"lpr_{c}" for c in all_cats]
all_num = num_cols + interact_cols

# cross-validated Random Forest model
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators = 200, max_depth = 5, min_samples_leaf = 5,
        class_weight = 'balanced', random_state = 42
    ))
])

clf.fit(X_train, y_train)
y_prob = clf.predict_proba(X_test)[:, 1] 
y_pred = clf.predict(X_test)
print(f"  \nAUC-ROC  : {roc_auc_score(y_test, y_prob):.4f}")

# try each multiplier for every listing to get potential sell price
# get predicted sell prob from this candidate_price
# get expected platform revenue from each
arms = {"A": 0.70, "B": 0.85, "C": 1.00, "D": 1.15, "E": 1.30}
price_min, price_max = 20, 400


#for row in X_test.iterrows():
def best_arm(row: pd.Series):
    """Determine best arm by expected platform revenue for each listing"""
    cands = []
    for arm, multiplier in arms.items():
        candidate_price = int(max(price_min, min(price_max, round(row['est_value'] * multiplier, 2))))
        d = pd.DataFrame([row])
        d['list_price'] = candidate_price
        d['log_price_ratio'] = np.log(candidate_price / row['est_value'])

        predicted_sell = clf.predict_proba(d)[0, 1]
        expected_rev = predicted_sell * (candidate_price * row['take_rate'])
        cands.append({"arm": arm, "candidate_price": candidate_price, 
                      "p_sell": predicted_sell, "exp_rev": expected_rev})
    cands_df = pd.DataFrame(cands)
    max_rev = cands_df["exp_rev"].max()
    # within-1% tie-break: prefer lower price
    eligible = cands_df[cands_df["exp_rev"] >= max_rev * 0.99]
    chosen = eligible.loc[eligible["candidate_price"].idxmin()]

    return {
        "listing_id": row["listing_id"],
        "category": row["category"],
        "est_value": row["est_value"],
        "take_rate": row["take_rate"],
        "logged_arm": row["arm"],
        "logged_price": row["list_price"],
        "logged_multiplier": row["multiplier"],
        "propensity": row["propensity"],
        "price_ratio": row["log_price_ratio"],
        "rec_arm": chosen["arm"],
        "rec_price": chosen["candidate_price"],
        "rec_p_sell": chosen["p_sell"],
        "rec_exp_rev": chosen["exp_rev"],
        "_cands": cands_df,
    }

policy_rows = [best_arm(row) for _, row in X_test.iterrows()]
rdf = pd.DataFrame([{k: v for k, v in r.items() if k != "_cands"} for r in policy_rows])
print(rdf.head())

differ_pct = (rdf["rec_arm"] != rdf["logged_arm"]).mean()
print(f"% test listings where rec ≠ logged arm: {differ_pct:.1%}")
# 87.6%
print("\nRecommended arm distribution:")
print(rdf["rec_arm"].value_counts().to_string())
# # just E = 2055?


# Example recommendations
print("\n── 5 Example Recommendations ──")
examples = (
    rdf.groupby("category")
       .sample(1, random_state=42)
       .reset_index(drop=True)
       .head(5)
)
for _, r in examples.iterrows():
    cands_df = next(pr["_cands"] for pr in policy_rows
                    if pr["listing_id"] == r["listing_id"])
    rev_c = cands_df[cands_df["arm"] == "C"]["exp_rev"].values[0]
    print(
        f"\n  listing {r['listing_id']:>5} | {r['category']:<14} | "
        f"est=${r['est_value']:>6.2f} | take={r['take_rate']:.3f}"
    )
    print(f"    Logged: arm={r['logged_arm']} price=${r['logged_price']}")
    print(f"    Rec:    arm={r['rec_arm']} price=${r['rec_price']:.0f}  "
          f"p_sell={r['rec_p_sell']:.3f}  E[rev]={r['rec_exp_rev']:.3f}")
    print(f"    Rationale: arm {r['rec_arm']} maximises E[platform_rev]; "
          f"{'within 1% tie → lower price chosen' if r['rec_arm'] != rdf.loc[rdf['rec_exp_rev'].idxmax(), 'rec_arm'] else 'strictly best expected revenue'}")


# Mini hand-check (3 listing IDs) -- check the best_arm() function is working as expected
    # B/C/D arm candidates printed with their prices
print("\n── Mini Hand-Check (3 Listings) ──")
HAND_CHECK_IDS = [2038, 1994, 2407]
arm_subset = ["B", "C", "D"]

for lid in HAND_CHECK_IDS:
    row = X_test[X_test["listing_id"] == lid].iloc[0]
    pr = best_arm(row)
    cdf = pr["_cands"]
    print(f"\n  listing_id={lid} | {row['category']} | "
          f"est_value=${row['est_value']:.2f} | take_rate={row['take_rate']:.3f}")
    print(f"  Logged arm: {row['arm']}  Logged price: ${row['list_price']}")
    print(f"  {'Arm':>3}  {'Candidate price':>16}  {'p_sell':>8}  {'E[platform_rev]':>16}")
    for _, c in cdf[cdf["arm"].isin(arm_subset)].iterrows():
        mark = "  <-- CHOSEN" if c["arm"] == pr["rec_arm"] else ""
        print(f"  {c['arm']:>3}  ${c['candidate_price']:>15.0f}  {c['p_sell']:>8.4f}  {c['exp_rev']:>16.4f}{mark}")
    if pr["rec_arm"] not in arm_subset:
        chosen_c = cdf[cdf["arm"] == pr["rec_arm"]].iloc[0]
        print(f"  {chosen_c['arm']:>3}  ${chosen_c['candidate_price']:>15.0f}  "
              f"{chosen_c['p_sell']:>8.4f}  {chosen_c['exp_rev']:>16.4f}  <-- CHOSEN")






