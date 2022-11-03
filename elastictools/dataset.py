import pandas as pd
import numpy as np

import warnings

class Dataset:
	def __init__(self, df, response_variable_column, to_binary_column, other_columns=[]):
		self.df = df

		# We check for the response variable and to binary columns whether they are present
		self.check_column_exists(response_variable_column)
		self.check_column_exists(to_binary_column)
		
		# We check whether the response variable column is binary
		self.check_column_is_binary(response_variable_column)

		# We need to binarise all values of the to_binary column
		# So we check in advance what features there will be
		self.context_features = self.df[to_binary_column].unique().tolist()

		self.response_variable_column = response_variable_column
		self.to_binary_column = to_binary_column
		self.other_columns = other_columns

		self.set_other_column_info()

	def check_column_exists(self, column):
		if not column in self.df.columns:
			raise Exception(f"Column '{column}' is not part of the supplied dataframe.")

	def check_column_is_binary(self, column, strict=True):
		values = self.df[column].unique()

		message = False

		if len(values) < 2:
			message = f"Column '{column}' contains less than two unique values"
		elif len(values) > 2:
			message = f"Column '{column}' contains more than two unique values"

		if strict and message:
			raise Exception(message)
		elif not strict and message:
			warnings.warn(message)

	def set_other_column_info(self):
		self.other_column_info = {}
		for other_column in self.other_columns:
			values = self.df[other_column].unique()
			is_binary = len(values) == 2
			reference_value = None
			if is_binary:
				reference_value = values[1]

			self.other_column_info[other_column] = { "is_binary": is_binary,
													 "reference_value": reference_value }

	def as_matrix(self, response_variable_1_value=None):
		response_variable_values = self.df[self.response_variable_column].unique()

		# We check whether the presupplied reference "1" value actually appears in the data
		if response_variable_1_value is not None:
			if not response_variable_1_value in response_variable_values:
				raise Exception(f"Reference response value '{response_variable_1_value}' not found")
		else:
			# We use the last value as reference "1" value
			response_variable_1_value = response_variable_values[1]

		# What is the total number of features?
		context_feature_count = len(self.context_features)
		other_columns_count = len(self.other_columns)

		# The total features consists of...
		# - the response variable
		# - the binary features
		# - the other columns
		total_feature_count = context_feature_count + other_columns_count + 1

		# Create the matrix
		# Size: dataframe rows X total feature count
		feature_matrix = np.zeros((len(self.df), total_feature_count))

		# We go over each row and check what the value is for the to_binary column
		for row_index, row in self.df.iterrows():
			to_binary_value = row[self.to_binary_column]

			# We check what the index of this value is in the context features list
			# This index corresponds to the index of the column of this value in the matrix
			to_binary_index = self.context_features.index(to_binary_value)

			# We then set the value for that column to 1 (= "this feature is present")
			feature_matrix[row_index][to_binary_index]

			# We also go over the "other columns", they have values too
			for list_index, other_column in enumerate(self.other_columns):
				# We decide the index of the other column feature in our matrix
				other_column_index = context_feature_count + list_index

				if self.other_column_info[other_column]["is_binary"]:
					if row[other_column] == self.other_column_info[other_column]["reference_value"]:
						feature_matrix[row_index][other_column_index] = 1

			# Finally, if we are dealing with the reference "1" value, set the response variable
			# column to 1
			if row[self.response_variable_column] == response_variable_1_value:
				feature_matrix[row_index][-1] = 1

		return feature_matrix