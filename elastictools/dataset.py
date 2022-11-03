import pandas as pd
import numpy as np

class Dataset:
	def __init__(self, df):
		self.df = df

	def check_column_exists(self, column):
		if not column in self.df.columns:
			raise Exception(f"Column '{column}' is not part of the supplied dataframe.")

	def check_column_is_binary(self, column):
		values = self.df[column].unique()

		if len(values) < 2:
			raise Exception(f"Column '{column} contains less than two unique values")
		elif len(values) > 2:
			raise Exception(f"Column '{column} contains more than two unique values")

	def as_matrix(self, response_variable_column, to_binary_column, other_columns=[],
						response_variable_1_value=None):
		# We check for the response variable and to binary columns whether they are present
		self.check_column_exists(response_variable_column)
		self.check_column_exists(to_binary_column)

		self.check_column_is_binary(response_variable_column)
		response_variable_values = self.df[response_variable_column].unique()

		# We check whether the presupplied reference "1" value actually appears in the data
		if response_variable_1_value is not None:
			if not response_variable_1_value in response_variable_values:
				raise Exception(f"Reference response value '{response_variable_1_value}' not found")
		else:
			# We use the last value as reference "1" value
			response_variable_1_value = response_variable_values[1]

		# We need to binarise all values of the to_binary column
		# So we check in advance how many features there will be
		self.context_features = self.df[to_binary_column].unique().tolist()
		context_feature_count = len(self.context_features)

		# The total features consists of...
		# - the response variable
		# - the binary features
		# - ...
		total_feature_count = 1 + context_feature_count

		# Create the matrix
		# Size: dataframe rows X total feature count
		feature_matrix = np.zeros((len(self.df), total_feature_count))

		# We go over each row and check what the value is for the to_binary column
		for row_index, row in self.df.iterrows():
			to_binary_value = row[to_binary_column]

			# We check what the index of this value is in the context features list
			# This index corresponds to the index of the column of this value in the matrix
			to_binary_index = self.context_features.index(to_binary_value)

			# We then set the value for that column to 1 (= "this feature is present")
			feature_matrix[row_index][to_binary_index]

			# Finally, if we are dealing with the reference "1" value, set the response variable
			# column to 1
			if row[response_variable_column] == response_variable_1_value:
				feature_matrix[row_index][-1] = 1

		return feature_matrix