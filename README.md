# Permutation Activities Location Importance

Supplementary material for the article *"Uncovering the Hidden Significance of Activity Locations in Predictive Process Monitoring"*

## Getting Started

To use this script, follow the steps below:

1. Clone this repository to your local machine:

```git clone https://github.com/your-username/location-permutation-importance.git```

2. Measure the importance of the location of activities by running ```CrossValidation_ProcessPermutation.py``` with the following flags:

- `--address`: Path to the dataset file.
- `--constrain`: Boolean flag to avoid generating unrealistic traces after permutation (default: `True`).
- `--Multi_activity`: Boolean flag for considering multi-activity in itemsets (default: `True`).
- `--top_k`: Number of top frequent itemsets to consider (default: `10`).
- `--min_support`: Minimum support for frequent itemset mining (default: `0.5`).

## Results

The script outputs CSV files containing location permutation importance scores and generates plots depicting the results. The generated files are named based on the input dataset.
