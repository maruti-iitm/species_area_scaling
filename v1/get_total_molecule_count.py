import pandas as pd

# Remove 'alphaDiversity' from the dataset as it's not relevant for this calculation
compound_class_filtered = compound_class_totals.drop(index='alphaDiversity', errors='ignore')

# Recalculate the proportions excluding 'alphaDiversity'
compound_class_filtered['Proportion'] = compound_class_filtered['SampleCount'] / compound_class_filtered['SampleCount'].sum()

# Display the updated proportions
compound_class_filtered.sort_values(by='Proportion', ascending=False).  

"""
SampleCount Proportion
Class
Lignin 87361 0.587443
Tannin 21777 0.146435
ConHC 15292 0.102828
Protein 15019 0.100993
AminoSugar 4444 0.029883
Lipid 2480 0.016676
Carb 1218 0.008190
UnsatHC 654 0.004398
Other 469 0.003154
"""