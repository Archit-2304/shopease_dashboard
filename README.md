SHOPEASE DASHBOARD HANDOFF

Group ID: 066_100_102
Random seed: 66100102
Raw sample: 2500 rows
Cleaned dashboard data: 2401 rows
Flagged rows excluded: 75 rows

Use cleaned_orders.csv as the source for dashboard filters and charts.

Revenue = Quantity × UnitPrice × (1 − DiscountRate), for Delivered orders only.
Pending and Cancelled orders have Revenue = 0.
The currency is not specified in the supplied dataset; avoid assuming one.

OrderMonth is derived from OrderDate.
Missing ages and ratings have not been filled with invented values.
Summaries are calculated from the full cleaned sample. Dashboard filters
will change the displayed totals.

Cleaning: standardized text and discounts; parsed dates; removed duplicate
OrderIDs; excluded invalid order dates, quantities, unit prices, discounts,
and totals that did not match the calculated amount.
