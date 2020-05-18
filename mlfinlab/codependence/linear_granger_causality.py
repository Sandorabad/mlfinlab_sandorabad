"""
Different econometric measures of connectedness based on Granger-causality networks,
described in the paper: Econometric measures of connectedness and systemic risk in the
finance and insurance sectors.
Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1963216
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


# pylint: disable=invalid-name

def granger_causality_test(data: pd.DataFrame, reverse: bool, lag: int = 3,
                           significance_level: float = 0.05) -> pd.DataFrame:
    """
    This method runs a Granger-causality test for defining the indicator function of causality.

    The function is described in https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1963216 (p.11)

    :param data: (pd.DataFrame) A data frame which contains time series of different assets returns.
    :param reverse: (bool) True to define the reverse direction of Granger causality.
    :param lag: (int) A value coming from an earlier point in time.
    :param significance_level: (float) A threshold for below which the null hypothesis is rejected.
    :return: (pd.DataFrame) A data frame which stores information whether a Granger causality exists
    between assets or not.
    """

    pairs_list = []  # Store the pairs of the corresponding assets
    save_index = []  # Store the indexes
    causality_indicator_list = []  # Store the indicator value
    for i in range(data.shape[1]):
        if reverse is None:
            j_range = np.arange(data.shape[1])
        elif reverse is False:  # Granger cause
            j_range = np.arange(i + 1, data.shape[1])
        else:  # Reverse Granger cause
            j_range = np.arange(i + 1)
        for j in j_range:
            if i != j:
                # Run a Granger causality test
                test_results = grangercausalitytests(data[[data.columns[i], data.columns[j]]],
                                                     maxlag=lag,
                                                     addconst=True,
                                                     verbose=False)
                # Check whether the Granger causality test accepts the null hypothesis or not
                if test_results[3][0]['ssr_ftest'][1] < significance_level:
                    # When j Granger causes i, then the indicator receives the value 1
                    causality_indicator: int = 1
                else:
                    # When j does not Granger cause i, then the indicator receives the value 0
                    causality_indicator = 0
                # Store the index and the pairs
                save_index.append(pairs_list)
                # Store the output of indicator function
                causality_indicator_list.append(causality_indicator)

                # Store causal pairs
                if reverse is (False or None):
                    pairs_list = (data.columns[i], data.columns[j])
                else:
                    pairs_list = (data.columns[j], data.columns[i])

    # Create a DataFrame which consists of Granger causality information
    gc_indicator = pd.DataFrame(causality_indicator_list, columns=['Indicator'],
                                index=pd.MultiIndex.from_tuples(save_index))

    return gc_indicator


def degree_of_granger_causality(data: pd.DataFrame, reverse: bool, lag: int = 3,
                                significance_level: float = 0.05) -> float:
    """
    Denote by the degree of Granger causality (DGC), the fraction of statistically significant
    Granger-causality relationships among all N(N-1) pairs of N assets.

    This function is described in https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1963216 (p.11)

    :param data: (pd.DataFrame) A data frame which contains time series of different assets returns.
    :param reverse: (bool) True to define the reverse direction of Granger causality.
    :param lag: (int) A value coming from an earlier point in time.
    :param significance_level: (float) Α threshold for below which the null hypothesis is rejected.
    :return: (float) A value which indicates the degree of Granger causality.
    """

    # Run Granger causality tests for both directions
    gc_result = \
        granger_causality_test(data, reverse=False, lag=lag, significance_level=significance_level)
    gc_result_reverse = \
        granger_causality_test(data, reverse=True, lag=lag, significance_level=significance_level)

    # Calculate the degree of Granger causality based on the directionality
    if reverse is False:
        degree_gc = gc_result['GC indicator'].sum(axis=0) / \
                    (len(data.columns) * (len(data.columns) - 1))
    else:
        degree_gc = gc_result_reverse['GC indicator'].sum(axis=0) / \
                    (len(data.columns) * (len(data.columns) - 1))

    return degree_gc

def number_of_connections():
    """
    This method assesses the systemic importance of single institutions.
    """
    return 1

def sector_conditional_connections():
    """
    This method assesses the systemic importance of single institutions, but the connections
    condition on the type of financial institution.
    """
    return 1

def closeness():
    """
    This method measures the shortest path between a financial institution and all other
    institutions reachable from it, averaged across all other financial institutions.
    """
    return 1

def eigenvector_centrality():
    """
    This method measures the importance of a financial institution in a network by assigning
    relative scores to financial institutions based on how connected they are to the rest of
    the network.
    """
    return 1
