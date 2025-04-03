import pandas as pd

asking_prices = pd.Series([12000, 25000, 15000, 5000, 18000, 23000, 28000])
fair_prices = pd.Series([15000, 24000, 16000, 7000, 19000, 22000, 30000])

good_deals = asking_prices[asking_prices < fair_prices]

good_deal_indices = good_deals.index.tolist()

print("Indices of good deals:", good_deal_indices)
