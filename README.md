# dt-sensemaking-vba
## Research questions and analysis approach

### 1) Do participants converge more on Impact (Y) than Implementability (X)?
**Method:** For each seed statement placed by ≥2 participants, compute dispersion across participants on X and Y (SDx and SDy). Compare SD distributions across statements (mean/median SDy vs SDx; paired test on SDx–SDy). Lower SD = stronger convergence.

### 2) Are there “consensus zones” (grid regions) where placements concentrate?
**Method:** Aggregate all seed placements into the 7×7 grid. Compute count and percentage per cell (optionally normalize by number of participants). Visualize as a density heatmap and identify the top cells.

### 3) Which seed statements show the highest agreement in placement?
**Method:** For each seed statement (n≥k), compute agreement metrics: % placed in the exact same cell and/or % within a 1-cell radius. Also compute dispersion (SDx/SDy). Rank statements by highest agreement and lowest dispersion.

### 4) Which seed statements show the greatest disagreement in placement?
**Method:** For each seed statement (n≥k), compute a controversy score (mean Euclidean distance to the statement centroid) and dispersion (SDx/SDy). Rank statements by highest controversy / widest spread.

### 5) For which statements is disagreement driven more by X vs Y?
**Method:** For each seed statement, compare SDx vs SDy (or IQRx vs IQRy). Flag statements where SDx ≫ SDy (implementability debate) or SDy ≫ SDx (impact debate).

### 6) Which practices are seen as high impact but hard to implement?
**Method:** Compute each statement’s centroid (x̄, ȳ). Apply a “high-impact / hard-to-implement” rule (e.g., x≤3 and y≥5). List statements in this quadrant and report x̄, ȳ and dispersion.

### 7) Which practices are seen as easy to implement but low impact?
**Method:** Same approach as Q6 but for “easy / low impact” (e.g., x≥5 and y≤3). Rank by lowest impact and highest implementability; report dispersion.

### 8) Do different seed themes map to different grid regions?
**Method:** Assign each seed statement to a theme (manual tagging). Compute theme centroid and dispersion (mean x/y; SD). Compare themes using centroid differences and/or nonparametric tests.

### 9) Do participants exhibit different mapping styles (overall tendencies)?
**Method:** For each participant, compute mean x, mean y, variance, and a “high-high tendency” metric (optimism). Compare participants descriptively; optionally correlate with metadata.

### 10) How extreme/consistent are participants in their placements?
**Method:** Per participant, compute extremity rate (% placements where x∈{1,7} or y∈{1,7}) and within-person dispersion (SD of x and y across placed cards). Rank/cluster participants by these metrics.

### 11) Can participants be grouped into placement profiles based on shared seeds?
**Method:** Compute pairwise participant distances using only shared seed statements (mean Euclidean distance across shared items). Cluster participants (hierarchical or k-means) to form profiles.
Clustering seem not to be good because there are only 2 clusters --> Lets excluded q11

### 12) Which statements best separate participant clusters?
**Method:** After clustering, compute for each seed statement the between-cluster centroid gap (Δ), overall and by dimension (Δx/Δy). Rank statements by largest gaps.
Because excluded q11, excluded q12 also

### 13) Do placements differ by function (e.g., Marketing vs Product)?
**Method:** For each seed statement, compute group centroids (x̄,ȳ) by function using the function of the participants in participant.csv. Rank statements by between-group distance; test differences (Mann–Whitney / Kruskal–Wallis) where feasible.

### 14) Do placements differ by country/region?
**Method:** Same as Q13 but group by country. Rank statements by between-group distance and summarize whether differences are mainly X or Y.

### 15) Do placements differ by role level?
**Method:** Same as Q13 but group by role level. Report top differing statements, direction of effects, and dispersion.

### 16) Does tenure predict perceived implementability/impact?
**Method:** Correlate or regress tenure against participant-level tendencies (mean x/mean y, extremity) and/or per-statement deviations. Optionally model x/y vs tenure with mixed-effects if advanced.

### 17) Does hybrid % relate to placement patterns?
**Method:** Correlate hybrid% with participant-level metrics (mean x/y, extremity) and/or per-statement placements. Highlight statements most sensitive to hybrid%.

### 18) Do raw (self-generated) statements land in different regions than seed statements?
**Method:** Compare raw vs seed placement distributions (mean/median x,y; cell density; extremity). Use nonparametric tests + effect sizes (raw typically has fewer points).

### 19) What themes emerge in raw statements, and where do those themes land on the grid?
**Method:** Code raw statements into themes (manual or AI-assisted). For each theme, compute centroid and dispersion of placements. Compare across themes and participant groups.

### 20) Do raw statements reinforce or contradict seed priorities?
**Method:** Map each raw statement to the closest seed theme or nearest seed centroid (in grid space or theme space). Compute an alignment score (distance / quadrant match) and summarize patterns.

