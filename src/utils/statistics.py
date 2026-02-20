import numpy as np


def freedman_diaconis_bins(x):
    """
    Returns the number of bins according to the Freedman-Diaconis rule.

    The Freedman-Diaconis rule is a statistical method used to determine the
    optimal bin width (h) for a histogram, designed to minimize the difference
    between the empirical histogram and the underlying probability density function.
    It is highly robust to outliers and skewed data because it uses the interquartile
    range (IQR) rather than the standard deviation.

    Formula:
        Bin Width = 2 * IQR(x) / n^(1/3)

    Where:
        IQR(x) : Interquartile Range of the data (75th percentile - 25th percentile)
        n      : Total number of observations in the sample

    Key Characteristics:
        - Robustness: Because it relies on the IQR, this rule is less sensitive
          to extreme outliers than rules based on data range or standard deviation.
        - Optimal Density: It is designed to minimize the integrated mean squared
          error of the density estimate.
        - Application: Particularly effective for large datasets or data with
          heavy-tailed distributions.
        - Implementation: Often implemented in statistical software (e.g., nclass.FD
          in R, 'fd' in MATLAB histcounts, numpy.histogram_bin_edges(data, bins='fd')).

    Comparison:
        While simpler rules like Sturges' rule (best for normal distributions) or
        Scott's rule exist, the Freedman-Diaconis rule generally provides a better
        balance for non-normal or skewed data by adapting to the scale of the data.

    Parameters
    ----------
    x : array-like
        Input data for which the number of histogram bins is computed.

    Returns
    -------
    int
        Number of bins for a histogram based on the Freedman-Diaconis rule.
    """
    x = np.asarray(x)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    n = len(x)
    bin_width = 2 * iqr / (n ** (1 / 3))
    if bin_width == 0:  # if all values are equal
        return 1
    return int(np.ceil((x.max() - x.min()) / bin_width))
