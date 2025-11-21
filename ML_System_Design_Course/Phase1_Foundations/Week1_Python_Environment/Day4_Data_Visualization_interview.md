# Day 4: Data Visualization - Interview Questions

> **Topic**: Visual Storytelling
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is the difference between a Histogram and a Bar Chart?
**Answer:**
*   **Histogram**: Visualizes the distribution of a **continuous** variable (bins). Bars touch each other. Order matters (numerical).
*   **Bar Chart**: Visualizes counts of a **categorical** variable. Gaps between bars. Order doesn't strictly matter (can sort by count).

### 2. When would you use a Box Plot vs a Violin Plot?
**Answer:**
*   **Box Plot**: Shows summary statistics (Median, IQR, Outliers). Good for comparing many groups. Hides the underlying distribution shape (modality).
*   **Violin Plot**: Shows the **KDE (Density)** of the data. Reveals if data is bimodal (two peaks). Better for detailed analysis but harder for non-experts to read.

### 3. Explain the "Grammar of Graphics" concept.
**Answer:**
*   A framework (popularized by ggplot2) that breaks plots into components: **Data**, **Aesthetics** (mapping data to x, y, color), **Geometries** (points, bars), **Facets** (subplots), and **Layers**.
*   Allows building complex plots by combining simple building blocks.

### 4. How do you handle Overplotting when visualizing 1 million points?
**Answer:**
*   **Transparency (Alpha)**: Make points semi-transparent.
*   **Sampling**: Plot a random subset.
*   **Hexbin / 2D Histogram**: Aggregate points into bins and color by count.
*   **Datashader**: Rasterizes the data on the server side before rendering.

### 5. What is the difference between `plt.figure()`, `plt.axes()`, and `plt.subplots()` in Matplotlib?
**Answer:**
*   `figure`: The top-level container (the window/page).
*   `axes`: The actual plot region (with x/y axis). A figure can have multiple axes.
*   `subplots`: A helper function to create a figure and a grid of axes in one go.

### 6. When should you use a Log Scale axis?
**Answer:**
*   When data spans **multiple orders of magnitude** (e.g., income, population, loss curves).
*   When the relationship is multiplicative/exponential (Log-log plot makes it linear).

### 7. What is a Heatmap? How is it useful for Feature Selection?
**Answer:**
*   **Heatmap**: Matrix where values are represented by color.
*   **Feature Selection**: Visualizing the **Correlation Matrix**. Highly correlated features (red squares off-diagonal) indicate redundancy (Multicollinearity).

### 8. Explain the difference between Sequential, Diverging, and Qualitative colormaps.
**Answer:**
*   **Sequential**: Low to High (e.g., Light Blue to Dark Blue). For counts/magnitudes.
*   **Diverging**: Has a meaningful midpoint (e.g., Red - White - Blue). For temperature or correlation (-1 to 1).
*   **Qualitative**: Distinct colors (e.g., Red, Green, Blue). For categorical data.

### 9. How do you visualize high-dimensional data (more than 3 dimensions)?
**Answer:**
*   **Dimensionality Reduction**: PCA/t-SNE projected to 2D.
*   **Parallel Coordinates**: Each dimension is a vertical axis. Lines connect values.
*   **Scatter Plot Matrix**: Grid of pairwise scatter plots.

### 10. What is "Data-Ink Ratio"?
**Answer:**
*   Concept by Edward Tufte.
*   **Ratio** = (Ink used for data) / (Total ink used).
*   **Goal**: Maximize it. Remove non-essential ink (gridlines, 3D effects, background colors) that distract from the data.

### 11. How would you visualize the distribution of a categorical variable?
**Answer:**
*   **Bar Chart**: Count per category.
*   **Treemap**: Nested rectangles. Good for hierarchical categories.
*   **Waffle Chart**: Grid of squares.

### 12. What is a Scatter Plot Matrix (Pair Plot)? What does it reveal?
**Answer:**
*   A grid showing scatter plots for every pair of variables.
*   **Reveals**: Linear/Non-linear correlations, clusters, and outliers across all dimensions simultaneously.

### 13. How do you save a Matplotlib figure with high resolution (DPI) for publication?
**Answer:**
*   `plt.savefig('plot.png', dpi=300, bbox_inches='tight')`.
*   `dpi=300`: Print quality.
*   `bbox_inches='tight'`: Removes extra whitespace around the plot.

### 14. What is the difference between Stateful (pyplot) and Object-Oriented Matplotlib APIs?
**Answer:**
*   **Stateful**: `plt.plot()`. Relies on global state (current figure). Easy for simple scripts. Hard to manage for complex subplots.
*   **OO**: `fig, ax = plt.subplots(); ax.plot()`. Explicitly acts on objects. Recommended for all serious work.

### 15. How do you visualize missing data patterns in a dataset?
**Answer:**
*   **Missingno Library**: Matrix plot where white lines indicate missing values.
*   Helps identify if missingness is random or correlated (e.g., sensor A and B fail together).

### 16. What is a QQ Plot? How is it used to test for normality?
**Answer:**
*   **Quantile-Quantile Plot**.
*   Plots quantiles of your data vs quantiles of a theoretical Normal distribution.
*   If points fall on the 45-degree line, the data is Normal.

### 17. When is a Pie Chart appropriate? (Hint: Almost never, but why?)
**Answer:**
*   **Issue**: Humans are bad at comparing angles/areas.
*   **Use**: Only when you have 2-3 categories that sum to a whole (Part-to-whole) and the differences are large. Otherwise, use a Bar Chart.

### 18. How do you create an interactive plot in Python? (Libraries like Plotly/Bokeh).
**Answer:**
*   Use **Plotly Express**: `px.scatter(df, x='a', y='b', hover_data=['c'])`.
*   Generates HTML/JS that allows zooming, panning, and hovering.

### 19. What is a "Stacked Bar Chart" vs "Grouped Bar Chart"? When to use which?
**Answer:**
*   **Stacked**: Bars on top of each other. Good for seeing the **Total** and the composition. Hard to compare individual segment sizes.
*   **Grouped**: Bars side-by-side. Good for **comparing** values between groups.

### 20. How do you visualize time-series data with seasonality?
**Answer:**
*   **Line Chart**: Standard.
*   **Seasonal Decomposition Plot**: Splits series into Trend, Seasonality, and Residuals.
*   **Polar Plot**: Plot time on a circle (0-24h or Jan-Dec) to see cycles overlap.
