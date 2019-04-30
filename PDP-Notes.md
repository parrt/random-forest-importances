# Partial dependence through stratification and approximating partial derivatives

## Existing approaches

There are a few basic approaches to identifying the partial effect of a single variable on the target or response variable:

* Stratification
* Beta coefficients from linear model without regularization
* Marginal plots
* PDP/ICE or [Accumulated Local Effects](https://arxiv.org/abs/1612.08468) (ALE) plots

Stratification is great because it's simple to implement and is obviously correctly isolating the effect of a single variable, $x_i$, on the target, $y$. We are grouping samples by all features except for $x_i$ then considering the relationship between that feature and $y$.   Unfortunately, it quickly breaks down when there are many variables because it's impossible to find enough samples that are equivalent across all of those variables.

PDP/ICE plots require that we adjust samples so that we move each $x_i$ through all values of that variable, which can create nonsensical or highly biased plots. Further, we are plotting the predictions, rather than using the raw data.

[ALE plots](https://christophm.github.io/interpretable-ml-book/ale.html#disadvantages-7). This requires that we have an approximation of the distance between categorical variables for ordering them.  This is dicey at best. The ALE computations are much more complicated than the other approaches.

## The new idea

Our approach does not assume nor require an external model and has three overall steps: (i) stratification of observations into similar regions, (ii) piecewise-linear approximation to obtain the partial derivative of $y$ with respect to a feature of interest, $x_i$, and (iii) integration of the partial derivative to get the partial relationship / effect plot for $x_i$.
 
First, stratify observations into groups that are similar (rather than equal) in all features except the single feature of interest, $x_i$.  For each group, fit an OLS line (without regularization) through $x_i$ versus $y$ and associate the slope (the single $\beta$ coefficient) with that group's range of $x_i$ values.  Observations across feature space partitions will have overlapping $x_i$ regions because partitioning does not take $x_i$ into consideration.

The approximate partial derivative of $y$ with respect to $x_i$ at any given point within $x_i$'s range is the average of all slopes whose associated range overlaps with that point.  The partial derivative tells us how $y$ is changing in response to changes in $x_i$  holding all other variables constant. Integrating this partial derivative then shows the relationship between $x_i$ and $y$. For example, if the partial derivative for $x_i$ is approximately some constant $k$ across the entire $x_i$ range, then the relationship between $x_i$ and $y$ is a line with slope $k$.  The $k$ constant is analogous to OLS coefficient $\beta_i$ where $\Delta y = \beta_i \Delta x_i$.

Grouping observations into similar regions of feature space is critical to the success of this algorithm.  There is always too little data in practice to group observations by exact equality and measuring similarity between categorical variables is always problematic. The algorithm internally trains a random forest using all $x_j$ for $j \neq i$ and target $y$. Observations that end up in the same leaf are considered similar. The trees must be limited in depth or forced to have a minimum number of samples per leaf (7 by default).  Multiple observations per leaf are required in order to estimate the slope between $x_i$ and $y$ in each leaf.

Random forests provide another important advantage beyond grouping similar observations in feature space. Because of bootstrapping, each tree in the forest is trained on a different subset of the original data. Averaging slopes across leaves in a single tree and leaves across trees to approximate the derivative of $x_i$, effectively controls for all other variables. This process is similar to how randomization controls for confounding variables in experimental design.

Categorical (and boolean) variables are handled differently than numeric variables. Instead of the slope, the algorithm tracks the average $y$ value for each category of $x_i$ and then subtracts the average value of the first category... something's not right here...

## Weaknesses

* if model minus feature x is bad then plot for x is meaningless as RF
can't do a grouping into similar buckets. in contrast, PDP uses all x to make
predictions.

## Advantages

* RF's can deal with categorical variables as a label encoded, so we don't have to worry about one-hot encoding issues
 
* Model-agnostic as it uses no external model; internally uses a random forest and multiple linear models

* Only considers points where we have data; then we use linear model to
  approximate the partial derivate. plot is then the integration.

* We aren't using model to compute points, only locally to get slopes.

* No nonsensical samples; not touching data

* LM works great if all numeric data; this works with categorical

* Seems to isolate partial dependencies better

## Interactions

$y = x_1 x_2$