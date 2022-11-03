# ElasticTools
Tools for preparing datasets for Elastic Net regression

## What is ElasticTools?

ElasticTools is a library with which you can easily generate datasets suitable for analysis using Elastic Net regression. The main idea is that you have a dataset you would use for "normal" regression, which you can then convert for use with Elastic Net regression. This is done by converting a column's values to binary predictors for those values. For example:

- the dataset has a column *verb lemma* with values *eat*, *sleep*, *rave*
- this column is converted to the columns *is_eat*, *is_sleep* and *is_rave*
	- a row's value for *is_eat* will be 1 if the value for *verb lemma* was *eat*
	- a row's value for *is_sleep* will be 1 if the value for *verb lemma* was *sleep*
	- a row's value for *is_rave* will be 1 if the value for *verb lemma* was *rave*

ElasticTools handles the following aspects of the dataset conversion process:
- converting a column to columns with binary values
- converting binary predictor columns to binary values
- exporting the resulting feature matrix to a NumPy matrix
- exporting a list of all features for matrix column identification

## Installing ElasticTools

ElasticTools is not available on any Python package manager (yet). To use it, simply copy the `elastictools` folder from this repository to your Python script's directory (preferably using `git clone`). From there, you can simply import the libraries like you would for any other package. More information on what libraries to import is given below.

## Using ElasticTools

### Importing the required libraries

You will need to import three libraries to be able to use ElasticTools:

- `pandas`: used to load your dataset into Python
- `numpy`: used to export the feature matrix
- `elastictools.dataset`: the Elastic Tools dataset class

```python
import pandas as pd
import numpy as np
from elastictools.dataset import Dataset
```

### Defining a dataset

If you want to define an Elastic Tools dataset, you first need to import your data into a pandas dataframe. Usually, this will be a CSV file of some sorts. For example, you could load your dataset as follows:

```python
df = pd.read_csv("dataset.csv")
```

Your dataset is now loaded as a pandas dataframe.

Now, we want to turn this dataframe into an ElasticTools dataset. The constructor for the Dataset class takes the following arguments:

| parameter | type    | description                                      | example |
| --------- | ------- | ------------------------------------------------ | -------| 
| `df` | [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)  | the dataframe which your dataset will be based on | / |
| `response_variable_column` | str  | the name of the column that holds your response variable | `"Variant"` |
| `to_binary_column` | str  | the name of the column that will be converted to binary columns | `"VerbLemma"` |
| `other_columns` (optional) | list(str)  | a list of names of other columns which will be used as predictors | `[ "VerbPosition", "NumberOfWords", "Country" ]` |

⚠ Currently, only binary response variables are supported (= logistic regression).

```python
dataset = Dataset(df, "Variant", "VerbLemma", [ "VerbPosition", "NumberOfWords", "Country" ])
```

### Converting the dataset to a NumPy matrix

To convert the dataset to a NumPy matrix, simply use the `as_matrix` method. This will return a feature matrix of type `numpy.ndarray`.

```python
feature_matrix = dataset.as_matrix()
```

You can save this matrix as an `.npy` file by using `np.save`:
```python
np.save("matrix.npy", feature_matrix)
```

### Retrieving the feature list

In the conversion to a NumPy matrix, the names of the different columns get lost. This is why we use `as_feature_list` to get the complete list of features corresponding to the matrix columns. The indices of the list correspond to the indices of the columns. The method will return a simple `list`.

```python
feature_list = dataset.as_feature_list()
```
⚠ The feature list does not include the name of the response variable column, since this column does not strictly contain a feature.

You can save this feature list as a CSV file using the following snippet:
```python
df_feat = pd.DataFrame(feature_list, columns=['feature'])
df_feat.index.name='index'
df_feat.to_csv("feature_list.csv")
```