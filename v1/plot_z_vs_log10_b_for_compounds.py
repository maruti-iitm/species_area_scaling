import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the CSVs
df_b = pd.read_csv('/Users/mudu605/Desktop/topic_5_scaling/v1/b_values_key_features.csv', index_col=0)
df_z = pd.read_csv('/Users/mudu605/Desktop/topic_5_scaling/v1/z_values_key_features.csv', index_col=0)

compounds = [
    'Amino Sugar', 'Carb', 'ConcHC',
    'Lignin', 'Lipid', 'Other',
    'Protein', 'Tanin', 'UnsatHC',
    'SumAllCompounds'
]

plt.figure()
for comp in compounds:
    # Extract b and z values for the compound
    b_vals = df_b.loc[comp].astype(float).values
    z_vals = df_z.loc[comp].astype(float).values
    
    # Mask valid entries
    mask = (~np.isnan(b_vals)) & (~np.isnan(z_vals))
    if mask.sum() < 2:
        continue
    
    log_b = np.log10(b_vals[mask])
    z = z_vals[mask]
    
    # Fit linear regression: Z ~ log10(b)
    X = log_b.reshape(-1, 1)
    y = z
    model = LinearRegression().fit(X, y)
    y_line = model.predict(X)
    r2 = r2_score(y, y_line)
    
    # Plot scatter and regression line
    plt.scatter(log_b, z, label=f"{comp} data", alpha=0.7)
    x_vals = np.linspace(log_b.min(), log_b.max(), 100)
    plt.plot(x_vals, model.predict(x_vals.reshape(-1, 1)), label=f"{comp} fit (R²={r2:.2f})")

plt.xlabel('log10(b)')
plt.ylabel('Z')
plt.title('Z vs log10(b) for All Compounds')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('z_vs_log10b_r2scores.png')
plt.show()