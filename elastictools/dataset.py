import pandas as pd

class Dataset:
	def __init__(self, df):
		self.df = df

	def check_column_exists(self, column):
		if not column in self.df.columns:
			raise Exception(f"Column '{column}' is not part of the supplied dataframe.")