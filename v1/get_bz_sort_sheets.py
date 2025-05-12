import pandas as pd
import os

# Load all sheets from the Excel file
path           = os.getcwd() + '/'
file_path = path + 'p_value_important_features.xlsx'
#
sheets = pd.read_excel(file_path, sheet_name=None)

# Define compounds to extract
compounds = ['Amino Sugar', 'Carb', 'ConcHC',\
                'Lignin', 'Lipid', 'Other', \
                'Protein', 'Tanin', 'UnsatHC', \
                'SumAllCompounds']

# Prepare df_b and df_z DataFrame
df_z = pd.DataFrame(index=compounds, columns=sheets.keys())
df_b = pd.DataFrame(index=compounds, columns=sheets.keys())

# Extract values
for sheet_name, df in sheets.items():
    feature_col = df.columns[0]
    z_value_col = df.columns[1]
    b_value_col = df.columns[2]
    for comp in compounds:
        matches = df[df[feature_col] == comp]
        if not matches.empty:
            df_z.at[comp, sheet_name] = matches.iloc[0][z_value_col]
            df_b.at[comp, sheet_name] = matches.iloc[0][b_value_col]
        else:
            df_z.at[comp, sheet_name] = None
            df_b.at[comp, sheet_name] = None

df_z.to_csv('/Users/mudu605/Desktop/topic_5_scaling/v1/z_values_key_features.csv')
df_b.to_csv('/Users/mudu605/Desktop/topic_5_scaling/v1/b_values_key_features.csv')