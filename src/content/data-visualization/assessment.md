This quiz covers the whole Data Visualization module: Matplotlib's object model, seaborn, Plotly, and choosing, drawing and honestly reading histograms, scatter plots, heatmaps, box plots and correlation plots. Aim for at least 13 out of 16 before moving on to SQL and Databases.

```quiz
[
  {
    "q": "In Matplotlib, what is an Axes?",
    "options": ["The x and y tick marks only", "A single plotting area with its own coordinate system, inside a Figure", "The whole image file", "A seaborn theme"],
    "answer": 1,
    "why": "A Figure is the canvas; each Axes is one plot within it, with its own axes, title and artists."
  },
  {
    "q": "What does this code create? `fig, (a, b) = plt.subplots(1, 2, sharey=True)`",
    "options": ["Two figures with one Axes each", "One figure with two Axes stacked vertically", "One figure with two side-by-side Axes sharing the same y-axis range", "One Axes with two lines"],
    "answer": 2,
    "why": "`(1, 2)` means one row and two columns, and `sharey=True` links their y-axes."
  },
  {
    "q": "Which change makes a bar chart of model accuracies honest?",
    "options": ["Starting the y-axis at zero", "Using a 3D bar style", "Setting the y-axis to the range of the data", "Adding a second y-axis"],
    "answer": 0,
    "why": "Bars encode value by length, so the axis must start at zero for lengths to be proportional."
  },
  {
    "q": "What is the main difference between `sns.histplot` and `sns.displot`?",
    "options": ["`displot` cannot draw histograms", "`histplot` only accepts NumPy arrays", "They are identical aliases", "`histplot` draws on one Axes; `displot` is figure-level and can facet into panels with `col` and `row`"],
    "answer": 3,
    "why": "Axes-level functions draw on an Axes you supply; figure-level functions create their own figure and grid."
  },
  {
    "q": "You have a wide DataFrame with columns `epoch`, `train_loss`, `val_loss`. What must you do before `sns.lineplot(..., hue=\"split\")`?",
    "options": ["Sort by epoch", "Melt it to long form so that `split` is a column", "Convert it to a NumPy array", "Drop the epoch column"],
    "answer": 1,
    "why": "Seaborn maps columns to visual properties, so the series name needs its own column, which `melt` creates."
  },
  {
    "q": "When is Plotly a better choice than Matplotlib?",
    "options": ["For a figure in a printed paper", "For the smallest possible file size", "When you need to hover over points to inspect individual examples, or build a dashboard", "When plotting a single small bar chart"],
    "answer": 2,
    "why": "Plotly's strength is interactivity in the browser: hover, zoom and toggling series."
  },
  {
    "q": "A histogram of a bimodal feature drawn with 4 bins shows a single peak. What happened?",
    "options": ["Too few bins merged the two peaks", "The data has outliers", "The axis is truncated", "`density=True` was missing"],
    "answer": 0,
    "why": "Wide bins over-smooth the distribution and can hide real structure such as a second mode."
  },
  {
    "q": "To compare training and production distributions of a feature fairly with overlaid histograms, you should:",
    "options": ["Use `bins=30` for each dataset", "Use shared bin edges and `density=True`", "Plot counts on a log y-axis", "Use different colours and nothing else"],
    "answer": 1,
    "why": "Shared edges put both on the same grid, and density removes the effect of different sample sizes."
  },
  {
    "q": "A scatter of 200,000 points is a solid blob. Which fix best reveals where the data is dense?",
    "options": ["Larger markers", "A legend", "A rainbow colour map", "A hexbin plot with a colour bar"],
    "answer": 3,
    "why": "Hexbin counts points per cell and encodes the count as colour, which shows density at any scale."
  },
  {
    "q": "In a t-SNE plot of embeddings, which conclusion is safe?",
    "options": ["Cluster B is twice as spread out as cluster A", "Cluster A is far from cluster C, so they are very dissimilar", "Points that sit together in a tight cluster tend to be similar to each other", "The axes have units of similarity"],
    "answer": 2,
    "why": "t-SNE preserves local neighbourhoods but not cluster sizes or between-cluster distances."
  },
  {
    "q": "In a row-normalised confusion matrix with true labels on rows, what is the diagonal?",
    "options": ["Per-class recall", "Per-class precision", "Overall accuracy", "Class frequency"],
    "answer": 0,
    "why": "Each row is all examples of one true class, so the diagonal is the fraction of that class predicted correctly."
  },
  {
    "q": "Why is `jet` a poor colour map for heatmaps?",
    "options": ["It only has two colours", "It is not perceptually uniform, so it creates false boundaries and fails for colour-blind readers and greyscale", "It cannot show negative numbers", "seaborn does not support it"],
    "answer": 1,
    "why": "Rainbow maps vary lightness unevenly, which invents bands and hides real differences."
  },
  {
    "q": "With Q1 = 10 and Q3 = 30, which value would a default box plot draw as an individual point?",
    "options": ["45", "55", "60", "65"],
    "answer": 3,
    "why": "IQR = 20, so the upper fence is 30 + 1.5 times 20 = 60. Only values above 60 are drawn individually."
  },
  {
    "q": "A box plot of latency for one model shows a tall box. A violin plot of the same data shows two separate bulges. Which statement is correct?",
    "options": ["The box plot is wrong", "The violin plot is wrong", "The data has two clusters, which a box plot cannot show", "The data is normally distributed"],
    "answer": 2,
    "why": "A box plot summarises with quartiles and cannot represent multiple modes; the violin reveals the shape."
  },
  {
    "q": "What does this code select? `corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))`",
    "options": ["Each feature pair exactly once, excluding the diagonal", "Only the diagonal", "The lower triangle including the diagonal", "All correlations above 0.5"],
    "answer": 0,
    "why": "`np.triu` with `k=1` marks the strict upper triangle, and `where` keeps only those cells."
  },
  {
    "q": "A feature has Pearson correlation 0.03 with the target. Which statement is justified?",
    "options": ["The feature is useless and should be dropped", "The feature causes no change in the target", "The feature must be leaking information", "There is no linear relationship; a scatter plot could still reveal a non-linear one"],
    "answer": 3,
    "why": "Pearson only measures linear association, so a curved or U-shaped dependence can give a value near zero."
  }
]
```
