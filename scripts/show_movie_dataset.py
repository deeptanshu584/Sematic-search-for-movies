import pandas as pd
from tabulate import tabulate

# 1. Load Data
df = pd.read_csv("../data/tmdb_5000_movies.csv")

# 2. Select only Title and Overview, and take the first 5 rows
df_sample = df[['original_title', 'overview']].head(5)

# 3. TRICK: Cut off the text if it's too long (so the table doesn't break)
# This lambda function says: "If text is > 50 chars, cut it and add '...'"
df_sample['overview'] = df_sample['overview'].apply(lambda x: str(x)[:50] + "..." if len(str(x)) > 50 else x)

# 4. Print the Beautiful Table
# 'fancy_grid' creates those nice double lines ═
print("\n" + "="*60)
print(" MOVIE DATASET PREVIEW ")
print("="*60)
print(tabulate(df_sample, headers=["Movie Title", "Plot Summary"], tablefmt="fancy_grid"))
print("\n")