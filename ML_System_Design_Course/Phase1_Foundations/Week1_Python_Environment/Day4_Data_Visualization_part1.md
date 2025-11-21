# Day 4 (Part 1): Advanced Data Visualization & Theory

> **Phase**: 6 - Deep Dive
> **Topic**: Visualizing the Un-visualizable
> **Focus**: Big Data, High Dimensions, and Theory
> **Reading Time**: 60 mins

---

## 1. Visualizing Big Data (Datashader)

Plotting 100 Million points in Matplotlib will crash your kernel.

### 1.1 The Pipeline
1.  **Projection**: Map data to a 2D canvas.
2.  **Aggregation**: Bin data into pixels (e.g., 1000x1000 grid). Count points in each pixel.
3.  **Transformation**: Map counts to colors (Log scale, Histogram Equalization).
4.  **Rendering**: Output an image.

### 1.2 Benefits
*   **Zero Overplotting**: You don't draw circles on top of circles. You draw a density map.
*   **Speed**: Can process billions of points in seconds (using Numba/Dask).
*   **Objectivity**: Reveals structure that alpha-blending hides.

---

## 2. The Grammar of Graphics (Leland Wilkinson)

Libraries like Altair and ggplot2 are based on this theory.

### 2.1 Components
*   **Data**: The source.
*   **Aesthetics (Aes)**: Mapping data columns to visual channels (x, y, color, size, shape).
*   **Geometries (Geom)**: The shape to draw (point, line, bar).
*   **Facets**: Splitting plot into subplots based on a category.
*   **Statistics**: Transformations (binning, smoothing) applied before plotting.

### 2.2 Why it matters?
It allows you to reason about charts compositionally. "I want to map `horsepower` to `x`, `mpg` to `y`, and `cylinders` to `color`, using `points`."

---

## 3. High-Dimensional Visualization

How to see 100 dimensions?

### 3.1 Parallel Coordinates
*   **Method**: Draw N vertical axes. A point is a line connecting values on each axis.
*   **Insight**: Clusters appear as "bundles" of lines. Correlations appear as crossing (negative) or parallel (positive) lines between axes.

### 3.2 Andrews Curves
*   **Method**: Map each data point $X = (x_1, x_2, \dots)$ to a Fourier series function:
    $f(t) = x_1/\sqrt{2} + x_2 \sin(t) + x_3 \cos(t) + \dots$
*   **Insight**: Similar points define similar curves.
*   **Cons**: Order of features matters.

---

## 4. Tricky Interview Questions

### Q1: What is the "Lie Factor" in visualization?
> **Answer**: Defined by Edward Tufte.
> $\text{Lie Factor} = \frac{\text{Size of effect in graphic}}{\text{Size of effect in data}}$.
> *   Example: A bar chart starting at 50 instead of 0 exaggerates the difference between 55 and 60. The visual difference is 100%, the data difference is ~9%. Lie Factor = 11.

### Q2: When should you use a Log Scale?
> **Answer**:
> 1.  **Power Laws**: When data spans multiple orders of magnitude (Income, City Populations).
> 2.  **Ratios**: When multiplicative changes are more important than additive ones (Stock prices).
> 3.  **Skew**: To un-skew a distribution for better visibility.

### Q3: Explain "Data-Ink Ratio".
> **Answer**: Tufte's principle.
> $\text{Data-Ink Ratio} = \frac{\text{Ink used to display data}}{\text{Total ink used}}$.
> *   Goal: Maximize this. Remove background grids, 3D effects, heavy borders, and redundant labels.

---

## 5. Practical Edge Case: Color Blindness
*   **Problem**: Red/Green color maps are indistinguishable for 8% of men (Deuteranopia).
*   **Solution**: Use Perceptually Uniform colormaps like **Viridis**, **Magma**, or **Cividis**. Never use "Jet" or "Rainbow" (they introduce false artifacts).

