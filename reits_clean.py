import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Read in data
df = pd.read_excel("correlation.xlsx", sheet_name='dataselect', parse_dates=False, index_col="Dates")

print(df.head())
print(df.tail())

# 清理列
drop_volumns = list(filter(lambda c: c.startswith('LOG'),df.columns.values.tolist()))
df = df.drop(columns=drop_volumns)
# 清理行
df = df[df.index.notnull()]
df = df.drop('variance')
# 没有REITs
df_without_reits = df.drop(columns=['FNERTR Index'])

print(df.head())
print(df.tail())

# Calculate expected returns and sample covariance
def efficient_frontier(df):
    mu = expected_returns.mean_historical_return(df)
#     S = risk_models.sample_cov(df)
    S = risk_models.CovarianceShrinkage(df).ledoit_wolf()

    # Optimise for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    print(cleaned_weights)
    ef.portfolio_performance(verbose=True)

efficient_frontier(df)
efficient_frontier(df_without_reits)

df.to_csv("reits_clean.csv", date_format='%d-%m-%Y')
