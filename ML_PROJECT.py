'''
The task here is to predict median house values in Californian districts,
given a number of features from these districts.
'''

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tarfile
import urllib.request
import ssl
import certifi
import tempfile
import shutil

'''************************************
START: LOADING DATA AND STORING IT IN A FILE
**************************************'''


def load_housing_data():
    url = "https://github.com/ageron/data/raw/main/housing.tgz"

    ''' The next 3 line are work around certificate verification issue '''
    with urllib.request.urlopen(url, context=ssl.create_default_context(cafile=certifi.where())) as response:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            shutil.copyfileobj(response, tmp_file)

    with tarfile.open(tmp_file.name) as housing_tarball:
        housing_tarball.extractall(path="datasets")

    return pd.read_csv(Path("datasets/housing/housing.csv"))


housing = load_housing_data()

# TAKING A QUICK LOOK AT THE DATA STRUCTURE AND RELATED STATISTICS
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

'''************************************
END: LOADING DATA AND STORING IT IN A FILE
**************************************'''

'''****************************************************************
 CODE TO SAVE THE FIGURES AS HIGH-RES PNGS FOR THE BOOK
****************************************************************'''
IMAGES_PATH = Path() / "images" / "end_to_end_project"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing.hist(bins=50, figsize=(12,8))
save_fig("attribute_histogram_plots")
plt.show()
'''****************************************************************
END- CODE TO SAVE THE FIGURES AS HIGH-RES PNGS FOR THE BOOK
****************************************************************'''

'''****************************************************************
DIFFERENT WAYS TO CREATE TRAIN AND TEST SAMPLES
****************************************************************'''
import numpy as np


def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = shuffle_and_split_data(housing, 0.2)
print(len(test_set))
np.random.seed(42)

# OR
from zlib import crc32


def is_id_in_test(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2 ** 32


def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


housing_with_id = housing.reset_index()  # adds an 'index' column
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

# OR
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")

# OR
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# print(test_set["total_bedrooms"].isnull().sum())
'''****************************************************************
END- DIFFERENT WAYS TO CREATE TRAIN AND TEST SAMPLES
****************************************************************'''

'''****************************************************************
EXTRA CODE - SHOWS HOW TO COMPUTE 10.7% PROB OF GETTING BAD SAMPLE
****************************************************************'''
from scipy.stats import binom

sample_size = 1000
ratio_female = .511
proba_too_small = binom(sample_size, ratio_female).cdf(485 - 1)
proba_too_large = 1 - binom(sample_size, ratio_female).cdf(535)
print(proba_too_small + proba_too_large)

# OR

np.random.seed(42)
samples = (np.random.rand(100_000, sample_size) < ratio_female).sum(axis=1)
print(((samples <485) | (samples>535)).mean())
'''****************************************************************
END-  SHOWS HOW TO COMPUTE 10.7% PROB OF GETTING BAD SAMPLE
****************************************************************'''

'''****************************************************************
STRATIFIED SAMPLING USED TO CREATE TRAIN AND TEST SAMPLES
****************************************************************'''
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
save_fig("housing_income_cat_bar_plot")
plt.show()

from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = strat_splits[0]  # could any num from 0-9

# OR
strat_train_set, strat_test_set = train_test_split(housing,
                                                   test_size=0.2,
                                                   stratify=housing["income_cat"],
                                                   random_state=42)

print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
'''****************************************************************
END - STRATIFIED SAMPLING USED TO CREATE TRAIN AND TEST SAMPLES
****************************************************************'''

'''****************************************************************
COMPARES STRATIFIED SAMPLING TO RANDOM SAMPLING
****************************************************************'''


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
compare_props = pd.DataFrame({
    "Overall %": income_cat_proportions(housing),
    "Stratified %": income_cat_proportions(strat_test_set),
    "Random %": income_cat_proportions(test_set)
}).sort_index()

compare_props.index.name = "Income Category"
compare_props["Strat. Error %"] = (compare_props["Stratified %"] /
                                   compare_props["Overall %"] - 1)
compare_props["Rand. Error %"] = (compare_props["Random %"] /
                                  compare_props["Overall %"] - 1)

print((compare_props * 100).round(2))

# for set_ in (strat_train_set, strat_test_set):
#     set_.drop("income_cat", axis=1, inplace=True)

'''****************************************************************
END - COMPARES STRATIFIED SAMPLING TO RANDOM SAMPLING
****************************************************************'''

'''****************************************************************
VISUALISE THE DATA
****************************************************************'''
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
save_fig("bad_visualisation_plot")
plt.show()

housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
save_fig("better_visualisation_plot")
plt.show()

housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"]/100, label="population", c="median_house_value",
             cmap="jet", colorbar=True, legend=True, sharex=False, figsize=(10, 7))
save_fig("housing_prices_scatterplot")
plt.show()

filename = "california.png"
if not (IMAGES_PATH / filename).is_file():
    holm3_root = "https://github.com/ageron/handson-ml3/raw/main/"
    url = holm3_root + "images/end_to_end_project/" + filename
    print("Downloading", filename)

    f = open(IMAGES_PATH / filename,'wb')
    f.write((urllib.request.urlopen(url, context=ssl.create_default_context(cafile=certifi.where()))).read())
    f.close()

housing_renamed = housing.rename(columns={"latitude": "Latitude",
                                          "longitude": "Longitude",
                                          "population": "Population",
                                          "median_house_value":"Median house value (USD)"})
housing_renamed.plot(kind="scatter", x="Longitude", y="Latitude",
                     s=housing_renamed["Population"]/100, label="Population",
                     c="Median house value (USD)", cmap="jet", colorbar=True,
                     legend=True, sharex=False, figsize=(10, 7))

california_img= plt.imread(IMAGES_PATH / filename)
axis = -124.55, -113.95, 32.45, 42.05
plt.axis(axis)
plt.imshow(california_img, extent=axis)

save_fig("california_housing_prices_plot")
plt.show()
'''****************************************************************
END- VISUALISE THE DATA
****************************************************************'''

'''****************************************************************
LOOKING FOR CORRELATIONS
****************************************************************'''
corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")
plt.show()

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)
save_fig("income_vs_house_value_scatterplot")
plt.show()

housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

'''****************************************************************
END - LOOKING FOR CORRELATIONS
****************************************************************'''

'''****************************************************************
CLEAN THE DATA
****************************************************************'''

'''housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
print(strat_train_set)

housing.dropna(subset=["total_bedrooms"], inplace=True)
housing.drop("total_bedrooms", axis=1)
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)

null_rows_idx = housing.isnull().any(axis=1)
print(null_rows_idx)
print(housing.loc[null_rows_idx].head())

housing_op1 = housing.copy()
housing_op1.dropna(subset=["total_bedrooms"], inplace=True)
print(housing_op1.loc[null_rows_idx].head())

housing_op2 = housing.copy()
housing_op2.drop("total_bedrooms", axis=1, inplace=True)
print(housing_op2.loc[null_rows_idx].head())

housing_op3 = housing.copy()
median = housing["total_bedrooms"].median()
housing_op3["total_bedrooms"].fillna(median, inplace=True)
print(housing_op3.loc[null_rows_idx].head())'''

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
print(strat_train_set)

null_rows_idx = housing.isnull().any(axis=1)
print(null_rows_idx)
print(housing.loc[null_rows_idx].head())

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')

housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
print(imputer.statistics_)
print(housing_num.median().values)

X = imputer.transform(housing_num)
print(imputer.feature_names_in_)

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
print(housing_tr.loc[null_rows_idx].head())


from sklearn.ensemble import IsolationForest

isolation_forest = IsolationForest(random_state=42)
outlier_pred = isolation_forest.fit_predict(X)
print("OUTLIER PREDICTIONS")
print(outlier_pred)

# Dropping outliers
housing = housing.iloc[outlier_pred==1]
housing_labels = housing_labels.iloc[outlier_pred==1]
'''****************************************************************
END - CLEAN THE DATA
****************************************************************'''

'''****************************************************************
HANDLING TEXT AND CATEGORICAL ATTRIBUTES
****************************************************************'''
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(8))

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:8])
print(ordinal_encoder.categories_)

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)
print(housing_cat_1hot.toarray())

cat_encoder = OneHotEncoder(sparse_output=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)
print(cat_encoder.categories_)

df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
pd.get_dummies(df_test)
print(cat_encoder.transform(df_test))

df_test_unknown = pd.DataFrame({"ocean_proximity": ["<2H OCEAN", "ISLAND"]})
pd.get_dummies(df_test_unknown)
cat_encoder.handle_unknown = "ignore"
print(cat_encoder.transform(df_test_unknown))

print(cat_encoder.feature_names_in_)
print(cat_encoder.get_feature_names_out())

df_output = pd.DataFrame(cat_encoder.transform(df_test_unknown),
                         columns=cat_encoder.get_feature_names_out(),
                         index=df_test_unknown.index)
print(df_output)

'''****************************************************************
END - HANDLING TEXT AND CATEGORICAL ATTRIBUTES
****************************************************************'''

'''****************************************************************
Feature Scaling
****************************************************************'''
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
print(housing_num_min_max_scaled)

from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
print(housing_num_std_scaled)

fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
housing["population"].hist(ax=axs[0], bins=50)
housing["population"].apply(np.log).hist(ax=axs[1], bins=50)
axs[0].set_xlabel("Population")
axs[1].set_xlabel("Log of population")
axs[0].set_ylabel("Number of districts")
save_fig("long_tail_plot")
plt.show()

#  just shows that we get a uniform distribution
percentiles = [np.percentile(housing["median_income"], p)
               for p in range(1, 100)]
flattened_median_income = pd.cut(housing["median_income"],
                                 bins=[-np.inf] + percentiles + [np.inf],
                                 labels=range(1, 100+1))
flattened_median_income.hist(bins=50)
plt.xlabel("Median income percentile")
plt.ylabel("Number of districts")
plt.show()

'''
Note: incomes below the 1st percentile are labeled 1, and incomes above the 99th percentile are labeled 100.
This is why the distribution below ranges from 1 to 100 (not 0 to 100)
'''

from sklearn.metrics.pairwise import rbf_kernel

age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)

ages = np.linspace(housing["housing_median_age"].min(),
                  housing["housing_median_age"].max(),
                  500).reshape(-1, 1)
gamma1 = 0.1
gamma2 = 0.03
rbf1 = rbf_kernel(ages, [[35]], gamma=gamma1)
rbf2 = rbf_kernel(ages, [[35]], gamma=gamma2)

fig, ax1 = plt.subplots()

ax1.set_xlabel("Housing median age")
ax1.set_ylabel("Number of districts")
ax1.hist(housing["housing_median_age"], bins=50)

ax2 = ax1.twinx()
color = "blue"
ax2.plot(ages, rbf1, color=color, label="gamma = 0.1")
ax2.plot(ages, rbf2, color=color, label="gamma = 0.03", linestyle="--")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel("Age similarity, color=color")

plt.legend(loc="upper left")
save_fig("age_similarity_plot")
plt.show()

from sklearn.linear_model import LinearRegression

target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5]

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)
print(predictions)

from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)
print(predictions)

'''****************************************************************
END - FEATURE SCALING
****************************************************************'''

'''****************************************************************
CUSTOM TRANSFORMERS
****************************************************************'''
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])
rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[35.]], gamma=0.1))
age_sim_35 = rbf_transformer.transform(housing[["housing_median_age"]])

print(age_sim_35)

ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
ans = ratio_transformer.transform(np.array([[1., 2.], [3., 4.]]))
print(ans)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        X= check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_

from sklearn.cluster import KMeans
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]],sample_weight=housing_labels)
print(similarities[:3].round(2))

housing_renamed = housing.rename(columns={"latitude": "Latitude", "longitude": "Longitude",
                                         "population": "Population",
                                         "median_house_value": "Median house value (USD)"})
housing_renamed["Max cluster similarity"] = similarities.max(axis=1)

housing_renamed.plot(kind="scatter", x="Longitude", y="Latitude", grid=True,
                     s=housing_renamed["Population"]/100, label="Population",
                     c="Max cluster similarity",
                     cmap="jet", colorbar=True,
                     legend=True, sharex=False, figsize=(10, 7))
plt.plot(cluster_simil.kmeans_.cluster_centers_[:, 1],
         cluster_simil.kmeans_.cluster_centers_[:, 0],
         linestyle="", color="black", marker="X", markersize=20,
         label="Cluster centers")
plt.legend(loc="upper right")
save_fig("district_cluster_plot")
plt.show()

'''****************************************************************
END - CUSTOM TRANSFORMERS
****************************************************************'''
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


'''****************************************************************
TRANSFORMATION PIPELINES
****************************************************************'''
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])

from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

from sklearn import set_config

set_config(display='diagram')
print(num_pipeline)

housing_num_prepared = num_pipeline.fit_transform(housing_num)
print(housing_num_prepared[:2].round(2))

df_housing_num_prepared = pd.DataFrame(housing_num_prepared,
                                       columns=num_pipeline.get_feature_names_out(),
                                       index=housing_num.index)
print(df_housing_num_prepared.head(2))
print(num_pipeline.steps)
print(num_pipeline.set_params(simpleimputer__strategy="median"))

from sklearn.compose import ColumnTransformer

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object))
)
housing_prepared = preprocessing.fit_transform(housing)

housing_prepared_fr = pd.DataFrame(housing_prepared,
                                   columns=preprocessing.get_feature_names_out(),
                                   index=housing.index)
print(housing_prepared_fr.head(2).round(2))



def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]


def ratio_pipeline():
    return make_pipeline(SimpleImputer(strategy="median"),
                         FunctionTransformer(column_ratio, feature_names_out=ratio_name),
                         StandardScaler())


log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(column_ratio, feature_names_out=ratio_name),
    StandardScaler())

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                           "households", "median_income"]),
    ("geo", cluster_simil, ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
],
    remainder=default_num_pipeline)
housing_prepared = preprocessing.fit_transform(housing)
print(housing_prepared.shape)
print(preprocessing.get_feature_names_out())

'''****************************************************************
END - TRANSFORMATION PIPELINES
****************************************************************'''

'''****************************************************************
TRAINING AND EVALUATION ON THE TRAINING SET
****************************************************************'''
from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)
print(lin_reg.steps)

housing_predictions = lin_reg.predict(housing)
print(housing_predictions[:5].round(-2))
print(housing_labels.iloc[:5].values)

error_ratios = housing_predictions[:5].round(-2) / housing_labels.iloc[:5].values -1
print(",".join([f"{100 * ratio:.1f}%" for ratio in error_ratios]))

from sklearn.metrics import mean_squared_error
lin_rsme = mean_squared_error(housing_labels, housing_predictions, squared=False)
print(lin_rsme)

from sklearn.tree import DecisionTreeRegressor
tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)
print(tree_reg.steps)
housing_predictions = tree_reg.predict(housing)
tree_rsme = mean_squared_error(housing_labels, housing_predictions, squared=False)
print(tree_rsme)
'''****************************************************************
END - TRAINING AND EVALUATION ON THE TRAINING SET
****************************************************************'''

'''****************************************************************
BETTER EVALUATION USING CROSS-VALIDATION
****************************************************************'''
from sklearn.model_selection import cross_val_score
tree_rsmes = -cross_val_score(tree_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(tree_rsmes).describe())

lin_rsmes = -cross_val_score(lin_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(lin_rsmes).describe())

from sklearn.ensemble import RandomForestRegressor
forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                                scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(forest_rmses).describe())

forest_reg.fit(housing, housing_labels)
housing_predictions = forest_reg.predict(housing)
forest_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
print(forest_rmse)

'''****************************************************************
END - BETTER EVALUATION USING CROSS-VALIDATION
****************************************************************'''

'''****************************************************************
FINE-TUNE YOUR MODEL: GRID SEARCH
****************************************************************'''
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42))
])

param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
     'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15],
     'random_forest__max_features': [6, 8, 10]},
]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                           scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)
print(str(full_pipeline.get_params().keys())[:1000] + "...")
print(grid_search.best_params_)
print(grid_search.best_estimator_)

cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)

cv_res = cv_res[["param_preprocessing__geo__n_clusters",
                 "param_random_forest__max_features", "split0_test_score",
                 "split1_test_score", "split2_test_score", "mean_test_score"]]
score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
cv_res.columns = ["n_clusters", "max_features"] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)

print(cv_res.head())
'''****************************************************************
END - FINE-TUNE YOUR MODEL: GRID SEARCH
****************************************************************'''

'''****************************************************************
RANDOMIZED SEARCH
****************************************************************'''
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {"preprocessing__geo__n_clusters": randint(low=3, high=50),
                  "random_forest__max_features": randint(low=2, high=20)}

rnd_search = RandomizedSearchCV(full_pipeline, param_distributions=param_distribs,
                                n_iter=10, cv=3, scoring="neg_root_mean_squared_error",
                                random_state=42)
rnd_search.fit(housing, housing_labels)

cv_res = pd.DataFrame(rnd_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)

cv_res = cv_res[["param_preprocessing__geo__n_clusters",
                 "param_random_forest__max_features", "split0_test_score",
                 "split1_test_score", "split2_test_score", "mean_test_score"]]
score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
cv_res.columns = ["n_clusters", "max_features"] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)

print(cv_res.head())