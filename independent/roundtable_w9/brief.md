# CURRENT FINAL MODEL (incumbent) — facts measured by the operator
- File src/final_model.py (full code below; importable as `final_model` from BASELINE_SRC). Score = 0.5 x reference + 0.5 x additive,
  where reference = rank-average of an L2 logistic regression (C=74) and a random forest (500 trees, depth 4, max_features 0.4, min leaf 2,
  balanced), both on 18 features chosen inside each fold (MI -> |Spearman|>=0.95 de-dup -> RF importance -> fixed prune), and additive =
  L2 logistic on 21 fixed features with quartile-knot splines on 5 continuous features. Weight and C chosen by a fixed rule in nested CV
  (5 outer x 3 inner). Scores mapped to ranks via training-score ECDFs.
- Results: nested CV AUROC on the development rows 0.9835 (reference alone 0.9819); hard group (no 2013-14 subscription) 0.9115.
  Lockbox AUROC 0.99155 (95% CI 0.984-0.997). Kaggle public 0.97991 (best of all submissions so far).
- Uses NO boosting. Earlier experiments: adding an MLP to the blend +0.000; stronger regularisation neutral; explicit interactions -0.0004;
  separate hard-group model -0.010 hard group; removing account-ID cohort + uncertain donation features -0.010.
- 82% of remaining ranking errors involve the ~50 positives who did not subscribe in 2013-14.
- Student decisions: keep account-ID cohort features and the 2014 first-gift flag; prefer compact models.

# LOCKBOX RULE
A stratified 20% of training accounts (1,389 accounts, 66 positives) is hidden. Your script is run once with them removed from train.csv
and placed in test.csv; their AUROC is computed by the operator and NOT shown to you. Do all tuning/selection inside the training data.

# LECTURE 9: ENSEMBLE MODELS (AIPI 520, W9.pptx slide text)
[1] Ensemble Models
[2] Decision Trees for Classification
[3] Decision Trees ;; Ask a series of questions to narrow in on the label ;; Horns? ;; Moose ;; Yes ;; No ;; # Legs? ;; Bird ;; 2 ;; Color? ;; 4 ;; Lizard ;; Dog ;; Green ;; Brown
[4] How to choose the splits ;; Our goal is to build the most efficient tree – the one that successfully splits the data using the minimum number of splits / We define an objective function to optimize via the decision tree / Our objective function is to maximize the Information Gain (IG) at each split in the tree:
[5] Measures of impurity ;; Entropy ;; Gini Impurity
[6] Calculating IG using Gini ;; Two possible splitting scenarios: ;; Split A: ;; Split B: ;; (40,40) ;; (30,10) ;; (10,30) ;; (40,40) ;; (20,40) ;; (20,0)
[7] Getting predictions from the tree ;; We continue splitting the tree either until we cannot split any further, or we decide to stop / The bottom nodes are called “leaves” of the tree / If a leaf has >1 examples at it, we take the majority class and label all examples at the leaf with that class prediction ;; (4,4) ;; (3,1) ;; (1,3) ;; Prediction: ”Class A” ;; (A,B) ;; Prediction: ”Class B” ;; Leaf 1 ;; Leaf 2
[8] Tree depth ;; Depth of our tree (max # of splits of a branch) is a hyperparameter / Very shallow trees will underfit the data / Deep trees will overfit the data (every example ends up as own leaf)
[9] Visualizing a decision tree ;; We can use Scikit Learn’s plot_tree() function to visualize the tree model / This shows us the splits, the impurity values, contents of each branch, and the prediction at each node/leaf
[10] Regression trees ;; We can also use decision tress for regression / Rather than impurity to measure split quality we use MSE of children / Rather than taking the majority vote of samples in a leaf, we calculate the mean target value of the samples ;; Node ;; Leaf 1 ;; Leaf 2 ;; 4 samples / Target values: [5,9,6,8] / Prediction for leaf: (5+9+6+8)/4 = 7 ;; 3 samples / Target values: [4,2,3] / Prediction for leaf: (4+2+3)/3 = 3
[11] Benefits & challenges of trees ;; Highly interpretable / Train quickly / Handle non-linear relationships well / Do not require scaling or one-hot encoding variables ;; Benefits ;; Challenges ;; Highly sensitive to hyperparameters (tree depth) / Very prone to overfitting
[12] Decision tree Demo
[13] Ensembling
[14] Ensembling ;; Goal of ensembling is to combine multiple models together into a meta-model that has better generalization performance than each individual model alone / Averaging multiple models makes the aggregate model less likely to overfit and better at generalizing to new data / Bias is same -> bias of average predictions is same as bias of the predictions / Variance is lower -> if model outputs are independent (or close), the variance of the average prediction is lower than the variance of the individual model predictions by a factor N  (# of models in ensemble)
[15] Ensembling ;; Goal of ensembling is to combine multiple models together into a meta-model that has better generalization performance than each individual model alone ;; Data ;; D1 ;; D2 ;; DN ;; … ;; Model 1 ;; Model 2 ;; Model 3 ;; … ;; Aggregation function ;; Full dataset or slices of data ;; Same algorithm or different ;; Majority vote, simple average, weighted average etc. ;; Predictions
[16] 
[17] $1M Netflix Prize ;; ”We evaluated some of the new methods offline but the additional accuracy gains that we measured did not seem to justify the engineering effort needed to bring them into a production environment.” ;; Goal: to decrease RMSE of predicted movie ratings by >10% to under 0.8572 / The only 2 teams which achieved the goal both were mergers of teams that used ensemble models / Winning model consisted of 107 models ensembled together ;; https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429
[18] Challenges of ensemble models ;; Time and compute resources to train / Computational cost of running multiple models / Decrease in interpretability
[19] Random Forest
[20] Bootstrap aggregating (bagging) ;; In bagging, we use “bootstrapped” samples from the training dataset to train each model / Bootstrapping = sampling with replacement / We select the size of each bagging subset (usually as a % of total training) / Therefore, there are some duplicates and some points left out entirely / Because each model is trained on a different set of data, their output predictions can be considered close to independent / Thus, when we use the average of their predictions we reduce the variance
[21] From tree to random forest ;; Decision trees tend to overfit / We can grow several trees and take the majority vote / We use bagging to ensure each tree is trained on a different subset / To predict on new data, we take the majority vote / simple average ;; D1 ;; D2 ;; Tree 1 ;; Tree 2 ;; Average ;; Data
[22] RF hyperparameters ;; Random Forests require tuning of several hyperparameters: / Number of trees in the forest (default = 100) / Sampling strategy for bagging / Bagging sample size as % of total rows in training set / Max % of features represented in each bagging sample / Depth of trees / Max depth (max_depth) / Minimum samples per leaf (min_samples_leaf)
[23] Feature selection using RF
[24] Feature Importance ;; We can calculate the relative importance of each feature using the random forest’s model.feature_importances_ attribute / “Feature importance” is calculated by looking at how much the nodes that use each feature reduce gini impurity on average (across each tree and then all trees in the forest), weighted by number of samples at node
[25] Random Forest Demo
[26] Boosting
[27] What is boosting? ;; A form of ensemble modeling using multiple weak learners / Unlike random forests, in boosting the weak learners are related – for each successive model built, the model focuses on compensating for the mistakes of the previous models / Therefore, each successive model learns from the mistakes of the previous / The final model combines all previous weak learner models
[28] Decision Stumps ;; Single-node decision trees / Attempt to separate data based on values of a single feature / Moderately accurate at classifying the data ;; x1 ;; x2 ;; X1=0.6
[29] Decision Stumps ;; Let’s define a different decision stump to try to correctly classify the points that we previously got incorrect ;; x1 ;; x2 ;; X2=0.6
[30] Decision Stumps ;; Close, but not quite / Let’s try one more to correctly classify the points we got wrong last time ;; x1 ;; x2 ;; X1=0.9
[31] Decision Stumps ;; None of the individual decision stumps was able to correctly predict all points / But look what happens when we combine them together ;; x1 ;; x2
[32] AdaBoost ;; Source: Raschka, Sebastian. “Python Machine Learning” ;; We start by fitting a decision stump model on all training points / In round 2, we assign higher weight to the points misclassified in round 1 and lower weight to the points classified correctly / For round 3, we increase the weights of points misclassified in round 2, and decrease the weights of points correctly classified / We combine all 3 weak learners into an ensemble model by a weighted majority vote, weighted by classification performance
[33] AdaBoost algorithm
[34] AdaBoost in Scikit Learn ;; The default AdaBoost model uses a decision stump classifier / We can choose to use a different classifier if we wish / Our main hyperparameter is the number of weak learners (n_estimators)
[35] Gradient Boosting
[36] Thought exercise ;; We are given (x1,y1), (x2,y2)… (xn,yn) and we want to fit a model F(x) to minimize the squared loss (SSE) / We create a model F(x) but it has some error / We find F(x1) = 2.5 but y1 = 2.6, and F(x2) = 0.7 but y2 = 0.8 / How can we improve this model? / Rules: / You cannot change the model F(x) / But you can add more models
[37] Thought exercise ;; Suppose we create an additional model G(x) / But rather than training G(x) to estimate y, we train it to estimate the errors (residuals) of model F(x) / G(x1) = y1 - F(x1) / G(x2) = y2 - F(x2) / … / G(xn) = yn - F(xn) / So we train G(x) on the data (x1, y1-F(x1)), (x2, y2-F(x2)) … (xn, yn-F(xn)) / The role of model G(x) is to compensate for the shortcomings of F(x) / If our combined model is still not good, we can add another H(x)
[38] Thought exercise ;; Our new model is now F(x) + G(x) / If our combined model is still not satisfactory, we can add another H(x) / H(x) will attempt to model the residuals of F(x) + G(x) --> y - (F(x) + G(x) ) / This is actually very similar to gradient descent:
[39] Gradient Boosting
[40] Gradient boosting ;; Gradient boosting also works for classification by modeling the probabilities of each class and adding models to compensate for errors / Gradient boosting can quickly overfit / Methods to reduce overfitting: / Constrain the depth of each tree in the ensemble / Multiply the predictions of subsequent trees by a “learning rate” to reduce the influence of each tree and allow for more trees to be added / Use subsampling from rows and columns for each tree rather than the entire dataset (“stochastic gradient boosting”)
[41] Boosting Demo
[42] Boosting vs. Bagging
[43] Ensembles – Conclusion ;; Ensembled models can be very powerful, and are commonly used in the real world as well as in ML competitions / Bagging and boosting are two of the most common ensembling techniques / However, ensembling increases the computational complexity relative to individual models / We must decide if the increased computational costs are worth the improvement in performance

# DATA DOSSIER


## train.csv  (6,941 rows x 2 cols)

- `account.id` dtype=str null=0.0% nunique=6941
- `label` dtype=int64 null=0.0% nunique=2 | counts: {0: 6609, 1: 332}

Sample rows (CSV):
```
account.id,label
001i000000NuRdI,0
001i000000NuQoN,0
001i000000NuOHV,0
001i000000NuP2J,0
001i000000NuP5d,0
```

## test.csv  (2,975 rows x 1 cols)

- `ID` dtype=str null=0.0% nunique=2975

Sample rows (CSV):
```
ID
001i000000NuPnV
001i000000NuQwD
001i000000NuRnE
001i000000NuOgR
001i000000NuOvs
```

## sample_submission.csv  (2,975 rows x 2 cols)

- `ID` dtype=str null=0.0% nunique=2975
- `Predicted` dtype=float64 null=0.0% nunique=1 | counts: {0.5: 2975}

Sample rows (CSV):
```
ID,Predicted
001i000000NuPnV,0.5
001i000000NuQwD,0.5
001i000000NuRnE,0.5
001i000000NuOgR,0.5
001i000000NuOvs,0.5
```

## account.csv  (19,833 rows x 10 cols)

- `account.id` dtype=str null=0.0% nunique=19833
- `shipping.zip.code` dtype=str null=98.7% nunique=157
- `billing.zip.code` dtype=str null=14.9% nunique=2224
- `shipping.city` dtype=str null=98.6% nunique=109 | top: {'San Francisco': 79, 'Berkeley': 24, 'Oakland': 17, 'Walnut Creek': 6, 'Palo Alto': 6, 'Los Altos': 4, 'El Cerrito': 4, 'Menlo Park': 4, 'Lafayette': 4, 'Orinda': 4}
- `billing.city` dtype=str null=11.2% nunique=1272 | top: {'San Francisco': 3464, 'Berkeley': 1802, 'Oakland': 1175, 'Palo Alto': 741, 'Walnut Creek': 523, 'San Jose': 328, 'San Rafael': 262, 'Menlo Park': 259, 'Lafayette': 253, 'Mill Valley': 213}
- `relationship` dtype=str null=96.7% nunique=10 | counts: {nan: 19172, 'Education Participant': 422, 'Vendor - Artistic': 189, 'Vendor - General': 13, 'Vendor - Marketing': 12, 'Peer Organization': 10, 'Other': 9, 'Corporation': 3, 'Vendor - Artistic; Peer Organization': 1, 'Vendor - Marketing; Other': 1, 'Foundation': 1}
- `amount.donated.2013` dtype=float64 null=0.0% nunique=159 | mean=85.2, std=2225, min=0, 25%=0, 50%=0, 75%=0, 95%=0, 99%=600, max=2.114e+05
- `amount.donated.lifetime` dtype=float64 null=0.0% nunique=1733 | mean=1606, std=3.312e+04, min=0, 25%=0, 50%=0, 75%=25, 95%=1916, 99%=1.778e+04, max=2.288e+06
- `no.donations.lifetime` dtype=int64 null=0.0% nunique=57 | mean=1.631, std=4.974, min=0, 25%=0, 50%=0, 75%=1, 95%=10, 99%=26, max=95
- `first.donated` dtype=str null=72.1% nunique=1406

Sample rows (CSV):
```
account.id,shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated
001i000000NuNjz,,94513,,Brentwood,,0.0,0.0,0,
001i000000NuRIb,,92130,,San Diego,,0.0,0.0,0,
001i000000NuQ4Z,,94945,,Novato,,0.0,0.0,0,
001i000000NuOsK,,94109,,San Francisco,,0.0,154.0,1,2001-04-19 00:00:00
001i000000NuR1A,,,,San Rafael,,0.0,0.0,0,
```

## subscriptions.csv  (28,627 rows x 9 cols)

- `account.id` dtype=str null=0.0% nunique=6267
- `season` dtype=str null=0.0% nunique=21 | counts: {'1998-1999': 1943, '1996-1997': 1871, '1994-1995': 1842, '1997-1998': 1788, '1995-1996': 1774, '2000-2001': 1704, '1993-1994': 1624, '1999-2000': 1607, '2001-2002': 1457, '2003-2004': 1217, '2002-2003': 1199, '2006-2007': 1107, '2004-2005': 1094, '2005-2006': 1090, '2010-2011': 1075, '2007-2008': 1074, '2009-2010': 1065, '2012-2013': 1058, '2008-2009': 1042, '2011-2012': 1014, '2013-2014': 982}
- `package` dtype=str null=0.0% nunique=10 | counts: {'Full': 21004, 'Quartet': 3740, 'Quartet CC': 1385, 'Trio': 1288, 'Trio B': 431, 'Trio A': 364, 'Quartet B': 174, 'Quartet A': 157, 'CYO': 66, 'Full upgrade': 12, nan: 6}
- `no.seats` dtype=int64 null=0.0% nunique=15 | counts: {2: 17809, 1: 9532, 3: 613, 4: 597, 5: 41, 6: 17, 7: 4, 10: 3, 12: 3, 8: 2, 13: 2, 26: 1, 9: 1, 15: 1, 11: 1}
- `location` dtype=str null=0.0% nunique=8 | counts: {'San Francisco': 6680, 'Berkeley Sunday': 6045, 'Berkeley Saturday': 5670, 'Peninsula': 5488, 'Contra Costa': 3044, 'Santa Rosa': 1107, 'Orange County': 581, 'Los Angeles': 6, nan: 6}
- `section` dtype=str null=15.9% nunique=14 | counts: {'Orchestra': 13247, nan: 4543, 'Balcony Front': 2710, 'Premium Orchestra': 2487, 'Balcony': 1993, 'Dress Circle': 1225, 'Balcony Rear': 1004, 'Orchestra Front': 645, 'Box': 435, 'Orchestra Rear': 160, 'Santa Rosa': 106, 'Gallery': 47, 'Boxes House Right': 13, 'Boxes House Left': 11, 'Floor': 1}
- `price.level` dtype=float64 null=12.3% nunique=5 | counts: {2.0: 11722, 1.0: 4859, 3.0: 3827, nan: 3534, 4.0: 3477, 0.0: 1208}
- `subscription_tier` dtype=float64 null=0.0% nunique=20 | mean=1.518, std=0.7363, min=0.5, 25%=1, 50%=2, 75%=2, 95%=2, 99%=4, max=26
- `multiple.subs` dtype=str null=0.0% nunique=2 | counts: {'no': 28554, 'yes': 73}

Sample rows (CSV):
```
account.id,season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
001i000000Lhyd9,2013-2014,Full,2,San Francisco,,0.0,2.0,no
001i000000NuO8g,1995-1996,Full,2,Berkeley Sunday,Balcony Front,3.0,2.0,no
001i000000NuNWo,1993-1994,Full,2,Peninsula,Orchestra,2.0,2.0,no
001i000000LhyPS,2012-2013,Full,2,Berkeley Saturday,Balcony Front,2.0,2.0,no
001i000000LhyY7,1998-1999,Quartet,1,San Francisco,Balcony,4.0,0.5,no
```

## tickets_all.csv  (2,808 rows x 8 cols)

- `account.id` dtype=str null=0.0% nunique=2049
- `price.level` dtype=str null=7.4% nunique=9 | counts: {'4': 1042, '3': 563, '2': 489, '1': 435, nan: 208, '0': 35, 'Adult': 14, 'Youth': 14, 'GA': 6, '4.0': 2}
- `no.seats` dtype=int64 null=0.0% nunique=16 | mean=2.029, std=1.599, min=1, 25%=1, 50%=2, 75%=2, 95%=4, 99%=8, max=34
- `marketing.source` dtype=str null=79.2% nunique=17 | counts: {nan: 2224, 'Philharmonia website': 158, 'Other': 96, 'Postcard or brochure': 95, 'Word of mouth': 85, 'KDFC Radio': 39, 'City Box Office special offer email': 14, 'Other print': 13, 'San Francisco Chronicle': 13, 'News story or review': 12, 'Web search': 11, 'City Box Office newsletter': 9, 'Browsing City Box Office website': 9, 'Blog post': 9, 'San Francisco Classical Voice': 8, 'Other radio': 6, 'New York Times': 4, 'Facebook or Twitter': 3}
- `season` dtype=str null=0.0% nunique=4 | counts: {'2012-2013': 1001, '2011-2012': 782, '2013-2014': 735, '2010-2011': 290}
- `location` dtype=str null=1.6% nunique=6 | counts: {'San Francisco': 950, 'Berkeley Saturday': 771, 'Berkeley Sunday': 591, 'Peninsula': 388, 'Family concert': 63, nan: 44, 'Contra Costa': 1}
- `set` dtype=float64 null=0.8% nunique=8 | counts: {3.0: 473, 2.0: 469, 6.0: 456, 1.0: 422, 4.0: 413, 5.0: 336, 7.0: 194, 8.0: 23, nan: 22}
- `multiple.tickets` dtype=str null=0.0% nunique=2 | counts: {'no': 2692, 'yes': 116}

Sample rows (CSV):
```
account.id,price.level,no.seats,marketing.source,season,location,set,multiple.tickets
001i000000NuRCI,4,2,,2012-2013,San Francisco,2.0,no
001i000000NuRD9,2,2,,2012-2013,San Francisco,2.0,no
001i000000NuRTo,,2,,2012-2013,Berkeley Saturday,2.0,no
001i000000NuOQZ,4,1,,2012-2013,Berkeley Sunday,5.0,no
001i000000NuRQd,2,2,,2012-2013,Berkeley Saturday,1.0,no
```

## concerts.csv  (103 rows x 6 cols)

- `season` dtype=str null=0.0% nunique=4 | counts: {'2010-2011': 32, '2011-2012': 28, '2012-2013': 23, '2013-2014': 20}
- `concert.name` dtype=str null=0.0% nunique=27 | counts: {'Marion Verbruggen & the Italian Baroque Recorder': 6, 'Robert Levin Plays Mozart': 5, 'Richard Egarr: Masters of the English Baroque': 5, "Bach's Wedding Cantata": 4, "Vivaldi's Four Seasons": 4, "Handel's Messiah": 4, 'David Daniels': 4, 'Zheng Cao: The Life of a Singer': 4, "Haydn's Creation": 4, 'Sound the Trumpet': 4, 'Arias for Farinelli': 4, "Bach's Mass in B Minor": 4, "Schumann's Cello Concerto": 4, "Alexander's Feast": 4, "Beethoven's Fourths": 4, 'A Bach Christmas': 4, 'The Italian Violin': 4, 'Heroic Theseus': 4, 'Pergolesi in Naples': 4, 'Music from Imperial Saint Petersburg': 4, "Hummel's Concerto for Keyed Trumpet": 3, 'Quintessence of Classical Style': 3, 'Solomon in London': 3, 'CPE Bach and Haydn: Berlin and Vienna': 3, 'Music from the Heart of Europe': 3}
- `set` dtype=int64 null=0.0% nunique=8 | counts: {3: 17, 2: 16, 5: 16, 1: 14, 4: 14, 6: 14, 7: 8, 8: 4}
- `who` dtype=str null=0.0% nunique=27
- `what` dtype=str null=0.0% nunique=27
- `location` dtype=str null=2.9% nunique=6 | counts: {'San Francisco': 26, 'Berkeley Saturday': 26, 'Berkeley Sunday': 25, 'Peninsula': 20, nan: 3, 'Family concert': 2, 'Contra Costa': 1}

Sample rows (CSV):
```
season,concert.name,set,who,what,location
2010-2011,Zheng Cao: The Life of a Singer,7,"Nicholas McGegan, conductor | Zheng Cao, mezzo-soprano","REBEL: Les Caractres de la danse | HANDEL: ÒScherza infidaÓ from Ariodante | GLUCK: Selections from Orfeo ed Euridice, incl. Dance of the Blessed Spirits, Aria ÒChe faro senza Euridice,Ó and Ballet Ð Chaconne | STOOKEY: Into the Bright Lights (U.S. ",Peninsula
2011-2012,Alexander's Feast,7,"Nicholas McGegan, conductor | Dominique Labelle, soprano | Dann Coakwell, tenor | Philip Cutlip, baritone | Philharmonia Chorale, Bruce Lamott, direct",HANDEL: AlexanderÕs Feast,Berkeley Saturday
2010-2011,Robert Levin Plays Mozart,1,"Nicholas McGegan, conductor | Robert Levin, fortepiano","MOZART: Incidental Music from Thamos, King of Egypt, K. 345 | MOZART: Concerto for Fortepiano No. 20 in D minor, K. 466 | LEVIN: Improvisations on themes submitted by the audience | MOZART/LEVIN: Concerto Movement in G major from Nannerl's Music | Bo",San Francisco
2011-2012,Richard Egarr: Masters of the English Baroque,5,"Richard Egarr, harpsichord and conductor","HANDEL: Symphony from Saul | LOCKE: Music from The Tempest | PURCELL: Suite from The Fairy Queen | ARNE: Concerto for Harpsichord No. 5 in G minor | LAWES: String Fantasy  6 | HANDEL: Concerto Grosso Op. 3, No. 5 in D minor, HWV 316 | HANDEL: ÒArriv",Berkeley Sunday
2012-2013,A Bach Christmas,3,"Masaaki Suzuki, conductor | Sherezade Panthaki, soprano | Claire Kelm, soprano | Fabiana Gonzlez, alto | Dann Coakwell, tenor | Dashon Burton, bass-b","J.S. BACH | Orchestral Suite No. 3 in D major, BWV 1068 | Cantata No. 63 Christen, tzet diesen Tag, BWV 63 | Magnificat in E-flat major, BWV 243a",Peninsula
```

## concerts_2014-15.csv  (6 rows x 5 cols)

- `season` dtype=str null=0.0% nunique=1 | counts: {'2014-2015': 6}
- `concert.name` dtype=str null=0.0% nunique=6 | counts: {'Steven Isserlis, Boccherini, and Haydn': 1, 'Andreas Scholl, J.S. Bach, and Handel': 1, 'Vivaldi and Zelenka: A Joyous Christmas': 1, 'The Cousins Bach': 1, 'Rachel Podger and Vivaldi': 1, 'RossiniÕs The Marriage Contract': 1}
- `set` dtype=int64 null=0.0% nunique=6 | counts: {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
- `who` dtype=str null=0.0% nunique=6
- `what` dtype=str null=0.0% nunique=6

Sample rows (CSV):
```
season,concert.name,set,who,what
2014-2015,RossiniÕs The Marriage Contract,6,"Nicholas McGegan, conductor Soloists from San Francisco Opera CenterÕs Adler Fellowship Program Ted Huffman, director",ROSSINI La cambiale di matrimonio (The Marriage Contract)
2014-2015,Vivaldi and Zelenka: A Joyous Christmas,3,"Nicholas McGegan, conductor, Dominique Labelle, soprano, Christopher Ainslie, alto, Thomas Cooley, tenor, Dashon Burton, bass-baritone, The Philharmon","VIVALDI Dixit Dominus, RV 807 ZELENKA Missa Nativitatis Domini, ZWV 8 VEJVANOVSKY Sonata natalis Prince Paul ESTERHçZY Cur fles, Jesu HAYDN Ave Regina Coelorum"
2014-2015,"Andreas Scholl, J.S. Bach, and Handel",2,"Julian Wachner, conductor, Andreas Scholl, countertenor","HANDELArias including ÒVa tacitoÓ from Giulio Cesare J.S. BACHSinfonia to Cantata No. 42 Brandenburg Concerto No. 1 in F major, BWV 1046 Cantata No. 170 ÒVergnugte Ruh, beliebte SeelenlustÓ BWV 170 TELEMANNConcerto in D major for Violin, Oboe and Two"
2014-2015,The Cousins Bach,4,"Nicholas McGegan, conductor Sherezade Panthaki, soprano Clifton Massey, countertenor Brian Thorsett, tenor Jeffrey Fields, baritone The Philharmonia C","Johann Ludwig BACH, Trauermusik, Johann Christoph BACH Cantata ÒMeine Freundin, du bist schnÓ"
2014-2015,"Steven Isserlis, Boccherini, and Haydn",1,"Nicholas McGegan, conductor, Steven Isserlis, violoncello","BOCCHERINIConcerto for Violoncello No. 7 in G major, G. 480 C.P.E. BACH Concerto for Violoncello in A major, Wq 172 HAYDN Symphony No. 67 in F major Symphony No. 57 in D major"
```

## zipcodes.csv  (42,522 rows x 12 cols)

- `Zipcode` dtype=int64 null=0.0% nunique=42522 | mean=4.939e+04, std=2.822e+04, min=501, 25%=2.539e+04, 50%=4.861e+04, 75%=7.307e+04, 95%=9.569e+04, 99%=9.902e+04, max=9.995e+04
- `ZipCodeType` dtype=str null=0.0% nunique=4 | counts: {'STANDARD': 29971, 'PO BOX': 9465, 'UNIQUE': 2437, 'MILITARY': 649}
- `City` dtype=str null=0.0% nunique=18758 | top: {'APO': 351, 'WASHINGTON': 299, 'FPO': 202, 'HOUSTON': 199, 'NEW YORK': 165, 'EL PASO': 140, 'DALLAS': 130, 'ATLANTA': 116, 'SPRINGFIELD': 111, 'SACRAMENTO': 105}
- `State` dtype=str null=0.0% nunique=62
- `LocationType` dtype=str null=0.0% nunique=1 | counts: {'PRIMARY': 42522}
- `Lat` dtype=float64 null=1.5% nunique=2636 | mean=38.47, std=5.455, min=-44.25, 25%=34.95, 50%=39.09, 75%=41.85, 95%=46.26, 99%=48.63, max=71.28
- `Long` dtype=float64 null=1.5% nunique=5722 | mean=-90.88, std=16.42, min=-176.7, 25%=-97.31, 50%=-87.89, 75%=-80.05, 95%=-72.68, 99%=-69.26, max=171.2
- `Location` dtype=str null=0.0% nunique=30126
- `Decommisioned` dtype=bool null=0.0% nunique=2 | counts: {False: 41859, True: 663}
- `TaxReturnsFiled` dtype=float64 null=32.1% nunique=10921 | mean=4763, std=6231, min=250, 25%=631, 50%=1720, 75%=6782, 95%=1.811e+04, 99%=2.669e+04, max=7.048e+04
- `EstimatedPopulation` dtype=float64 null=32.1% nunique=13654 | mean=8320, std=1.081e+04, min=272, 25%=1108, 50%=3089, 75%=1.189e+04, 95%=3.134e+04, 99%=4.664e+04, max=8.85e+04
- `TotalWages` dtype=float64 null=32.2% nunique=28841 | mean=1.942e+08, std=2.88e+08, min=0, 25%=1.913e+07, 50%=5.63e+07, 75%=2.545e+08, 95%=8.172e+08, 99%=1.324e+09, max=2.133e+09

Sample rows (CSV):
```
Zipcode,ZipCodeType,City,State,LocationType,Lat,Long,Location,Decommisioned,TaxReturnsFiled,EstimatedPopulation,TotalWages
28681,STANDARD,TAYLORSVILLE,NC,PRIMARY,35.92,-81.17,NA-US-NC-TAYLORSVILLE,False,10570.0,19347.0,330514634.0
95160,PO BOX,SAN JOSE,CA,PRIMARY,37.21,-121.86,NA-US-CA-SAN JOSE,False,258.0,424.0,13634713.0
55371,STANDARD,PRINCETON,MN,PRIMARY,45.57,-93.59,NA-US-MN-PRINCETON,False,7498.0,13602.0,287586341.0
30303,STANDARD,ATLANTA,GA,PRIMARY,33.75,-84.39,NA-US-GA-ATLANTA,False,33430.0,33905.0,56802672.0
50305,PO BOX,DES MOINES,IA,PRIMARY,41.57,-93.61,NA-US-IA-DES MOINES,False,,,
```

## Example training accounts with full joined history
(30+ accounts sampled: 12 label=1, 12 label=0 with subscription history, 6 label=0 ticket-only, 5 label=0 with neither)


### 001i000000LhySm  label=1
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,93130,,Santa Barbara,,400.0,7296.66,16,2002-12-10 00:00:00
subscriptions (12 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
2002-2003,Full,1,San Francisco,,,1.0,no
2003-2004,Full,1,San Francisco,,,1.0,no
2004-2005,Full,1,San Francisco,,,1.0,no
2005-2006,Full,2,San Francisco,Orchestra Front,1.0,2.0,no
2006-2007,Full,2,San Francisco,Premium Orchestra,1.0,2.0,no
2007-2008,Full,2,San Francisco,Premium Orchestra,1.0,2.0,no
2008-2009,Full,2,San Francisco,Premium Orchestra,1.0,2.0,no
2009-2010,Full,2,San Francisco,Premium Orchestra,1.0,2.0,no
2010-2011,Full,2,San Francisco,Orchestra,0.0,2.0,no
2011-2012,Quartet,2,San Francisco,Orchestra,0.0,1.0,no
2012-2013,Trio B,2,San Francisco,Orchestra,0.0,1.0,no
2013-2014,Trio B,2,San Francisco,,0.0,1.0,no
```
tickets (0 rows): none

### 001i000000PzyQv  label=1
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94304,,Palo Alto,,0.0,0.0,0,
subscriptions (1 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
2013-2014,Full,1,Peninsula,,2.0,1.0,no
```
tickets (0 rows): none

### 001i000000LhyW8  label=1
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94114,,San Francisco,,100.0,1275.0,15,1997-12-30 00:00:00
subscriptions (17 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
1997-1998,Quartet,2,San Francisco,Orchestra,2.0,1.0,no
1998-1999,Quartet,2,San Francisco,Orchestra,2.0,1.0,no
1999-2000,Quartet,2,San Francisco,Orchestra,2.0,1.0,no
2000-2001,Quartet,2,San Francisco,Orchestra,2.0,1.0,no
2001-2002,Quartet,2,San Francisco,Orchestra,2.0,1.0,no
2002-2003,Quartet,1,San Francisco,,,0.5,no
2003-2004,Quartet,1,San Francisco,,,0.5,no
2004-2005,Quartet,1,San Francisco,,,0.5,no
2005-2006,Quartet,2,San Francisco,Orchestra Front,1.0,1.0,no
2006-2007,Quartet,2,San Francisco,Premium Orchestra,1.0,1.0,no
2007-2008,Quartet,2,San Francisco,Premium Orchestra,1.0,1.0,no
2008-2009,Quartet,2,San Francisco,Premium Orchestra,1.0,1.0,no
2009-2010,Quartet,2,San Francisco,Premium Orchestra,1.0,1.0,no
2010-2011,Quartet B,2,San Francisco,Orchestra,0.0,1.0,no
2011-2012,Trio,2,San Francisco,Orchestra,0.0,1.0,no
2012-2013,Trio B,2,San Francisco,Orchestra,0.0,1.0,no
2013-2014,Trio B,2,San Francisco,,0.0,1.0,no
```
tickets (0 rows): none

### 001i000000LhyYS  label=1
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94022,,Los Altos,,0.0,4563.0,23,1995-01-09 00:00:00
subscriptions (21 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
1993-1994,Full,2,Peninsula,Orchestra,2.0,2.0,no
1994-1995,Full,2,Peninsula,Orchestra,2.0,2.0,no
1995-1996,Full,2,Peninsula,Orchestra,2.0,2.0,no
1996-1997,Full,2,Peninsula,Orchestra,2.0,2.0,no
1997-1998,Full,2,Peninsula,Orchestra,2.0,2.0,no
1998-1999,Full,2,Peninsula,Orchestra,2.0,2.0,no
1999-2000,Full,2,Peninsula,Orchestra,2.0,2.0,no
2000-2001,Full,2,Peninsula,Orchestra,2.0,2.0,no
2001-2002,Full,2,Peninsula,Orchestra,2.0,2.0,no
2002-2003,Full,1,Peninsula,,,1.0,no
2003-2004,Full,1,Peninsula,,,1.0,no
2004-2005,Full,1,Peninsula,,,1.0,no
2005-2006,Full,2,Peninsula,Orchestra Rear,2.0,2.0,no
2006-2007,Full,2,Peninsula,Orchestra,2.0,2.0,no
2007-2008,Full,2,Peninsula,Orchestra,2.0,2.0,no
2008-2009,Full,2,Peninsula,Orchestra,2.0,2.0,no
2009-2010,Full,2,Peninsula,Orchestra,2.0,2.0,no
2010-2011,Full,2,Peninsula,Orchestra,2.0,2.0,no
2011-2012,Full,2,Peninsula,Orchestra,2.0,2.0,no
2012-2013,Full,2,Peninsula,Orchestra,2.0,2.0,no
2013-2014,Full,2,Peninsula,,2.0,2.0,no
```
tickets (0 rows): none

### 001i000000LhyWk  label=1
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94025,,Menlo Park,,0.0,120.0,3,2008-07-03 00:00:00
subscriptions (6 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
2007-2008,Quartet,2,Peninsula,Balcony Rear,4.0,1.0,no
2008-2009,Quartet,2,Peninsula,Balcony Rear,4.0,1.0,no
2009-2010,Trio,2,Peninsula,Balcony Rear,4.0,1.0,no
2010-2011,Quartet B,2,Peninsula,Orchestra,4.0,1.0,no
2012-2013,Trio B,1,Peninsula,Orchestra,4.0,0.5,no
2013-2014,Trio B,2,Peninsula,,4.0,1.0,yes
```
tickets (0 rows): none

### 001i000000LhyPH  label=1
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94124,,San Francisco,,0.0,0.0,0,
subscriptions (11 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
2003-2004,Full,1,San Francisco,,,1.0,no
2004-2005,Full,1,San Francisco,,,1.0,no
2005-2006,Full,2,San Francisco,Orchestra Front,1.0,2.0,no
2006-2007,Full,2,San Francisco,Premium Orchestra,1.0,2.0,no
2007-2008,Full,2,San Francisco,Premium Orchestra,1.0,2.0,no
2008-2009,Full,2,San Francisco,Premium Orchestra,1.0,2.0,no
2009-2010,Full,2,San Francisco,Premium Orchestra,1.0,2.0,no
2010-2011,Full,2,San Francisco,Orchestra,1.0,2.0,no
2011-2012,Full,2,San Francisco,Orchestra,1.0,2.0,no
2012-2013,Full,2,San Francisco,Orchestra,1.0,2.0,no
2013-2014,Full,2,San Francisco,,1.0,2.0,no
```
tickets (0 rows): none

### 001i000000mLusC  label=1
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94022,,Los Altos Hills,,0.0,100.0,1,2014-04-24 00:00:00
subscriptions (0 rows): none
tickets (0 rows): none

### 001i000000NuQws  label=1
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94577,,San Leandro,,0.0,10.0,1,2014-08-25 00:00:00
subscriptions (1 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
2012-2013,Trio A,2,Berkeley Sunday,Balcony Front,4.0,1.0,no
```
tickets (2 rows):
```
price.level,no.seats,marketing.source,season,location,set,multiple.tickets
Adult,2,,2012-2013,Family concert,5.0,no
3,2,Other,2013-2014,Berkeley Saturday,4.0,no
```

### 001i000000LhyZl  label=1
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94549,,Lafayette,,500.0,4758.0,18,2001-05-14 00:00:00
subscriptions (14 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
2000-2001,Trio,2,Berkeley Sunday,Balcony Front,3.0,1.0,no
2001-2002,Quartet CC,2,Contra Costa,Orchestra,2.0,1.0,no
2002-2003,Quartet CC,1,Contra Costa,,,0.5,no
2003-2004,Quartet CC,1,Contra Costa,,,0.5,no
2004-2005,Quartet CC,1,Contra Costa,,,0.5,no
2005-2006,Quartet CC,2,Contra Costa,Orchestra Front,1.0,1.0,no
2006-2007,Quartet CC,2,Contra Costa,Premium Orchestra,1.0,1.0,no
2007-2008,Quartet CC,2,Contra Costa,Premium Orchestra,1.0,1.0,no
2008-2009,Quartet CC,2,Contra Costa,Premium Orchestra,1.0,1.0,no
2009-2010,Quartet CC,2,Contra Costa,Premium Orchestra,1.0,1.0,no
2010-2011,Quartet CC,2,Contra Costa,Orchestra,0.0,1.0,no
2011-2012,Full,2,Berkeley Sunday,Orchestra,1.0,2.0,no
2012-2013,Full,2,Berkeley Sunday,Orchestra,1.0,2.0,no
2013-2014,Full,2,Berkeley Sunday,,1.0,2.0,no
```
tickets (0 rows): none

### 001i000000LhyZ8  label=1
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94707,,Kensington,,0.0,0.0,0,
subscriptions (4 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
2003-2004,Trio,1,Berkeley Sunday,,,0.5,no
2004-2005,Quartet,1,Berkeley Sunday,,,0.5,no
2006-2007,Trio,2,Berkeley Sunday,Balcony Front,3.0,1.0,no
2013-2014,Trio A,1,Berkeley Sunday,,4.0,0.5,no
```
tickets (1 rows):
```
price.level,no.seats,marketing.source,season,location,set,multiple.tickets
4,2,,2012-2013,Berkeley Sunday,1.0,no
```

### 001i000000NuOcN  label=1
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94306,,Palo Alto,,0.0,2060.0,11,1998-10-26 00:00:00
subscriptions (15 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
1995-1996,Full,2,Peninsula,Orchestra,2.0,2.0,no
1997-1998,Full,2,Peninsula,Orchestra,2.0,2.0,no
1998-1999,Full,2,Peninsula,Orchestra,2.0,2.0,no
1999-2000,Full,2,Peninsula,Orchestra,2.0,2.0,no
2000-2001,Full,2,Peninsula,Orchestra,2.0,2.0,no
2001-2002,Full,2,Peninsula,Orchestra,2.0,2.0,no
2002-2003,Full,1,Peninsula,,,1.0,no
2003-2004,Full,1,Peninsula,,,1.0,no
2004-2005,Full,1,Peninsula,,,1.0,no
2005-2006,Full,2,Peninsula,Orchestra Rear,2.0,2.0,no
2006-2007,Full,3,Peninsula,Orchestra,2.0,3.0,no
2007-2008,Full,3,Peninsula,Orchestra,2.0,3.0,no
2008-2009,Full,3,Peninsula,Orchestra,2.0,3.0,no
2009-2010,Full,3,Peninsula,Orchestra,2.0,3.0,no
2010-2011,Full,3,Peninsula,Orchestra,2.0,3.0,no
```
tickets (0 rows): none

### 001i000000LhyYj  label=1
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94595-2985,,Walnut Creek,,0.0,25.0,1,2001-01-29 00:00:00
subscriptions (4 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
2001-2002,Trio,2,Berkeley Sunday,Balcony Front,3.0,1.0,no
2011-2012,Trio,2,Berkeley Saturday,Balcony Front,2.0,1.0,no
2012-2013,Trio A,2,Berkeley Saturday,Balcony Front,2.0,1.0,no
2013-2014,Trio A,1,Berkeley Saturday,,2.0,0.5,no
```
tickets (0 rows): none

### 001i000000NuOE9  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94941,,Mill Valley,,0.0,50.0,1,1990-07-04 00:00:00
subscriptions (2 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
1993-1994,Full,2,Santa Rosa,Orchestra,2.0,2.0,no
1997-1998,Full,2,Santa Rosa,Orchestra,2.0,2.0,no
```
tickets (0 rows): none

### 001i000000NuNmX  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94303,,Palo Alto,,0.0,0.0,0,
subscriptions (2 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
1993-1994,Full,2,Peninsula,Orchestra,2.0,2.0,no
1994-1995,Full,2,Peninsula,Orchestra,2.0,2.0,no
```
tickets (0 rows): none

### 001i000000NuOfG  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94708,,Berkeley,,0.0,20.0,1,1996-11-18 00:00:00
subscriptions (3 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
1996-1997,Quartet,2,Peninsula,Orchestra,2.0,1.0,no
1997-1998,Quartet,2,Berkeley Sunday,Balcony Front,3.0,1.0,no
1998-1999,Full,2,Berkeley Sunday,Balcony Front,3.0,2.0,no
```
tickets (0 rows): none

### 001i000000NuPF0  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,92660,,Newport Beach,,0.0,0.0,0,
subscriptions (2 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
2000-2001,Full,2,Orange County,Orchestra,2.0,2.0,no
2001-2002,Full,2,Orange County,Orchestra,2.0,2.0,no
```
tickets (0 rows): none

### 001i000000NuOae  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94960,,San Anselmo,,0.0,0.0,0,
subscriptions (1 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
1994-1995,Full,3,Santa Rosa,Balcony,4.0,3.0,no
```
tickets (0 rows): none

### 001i000000NuOij  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94109,,San Francisco,,0.0,0.0,0,
subscriptions (1 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
1995-1996,Quartet,2,San Francisco,Box,1.0,1.0,no
```
tickets (0 rows): none

### 001i000000NuOjE  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94706,,Albany,,0.0,0.0,0,
subscriptions (1 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
1995-1996,Full,2,Berkeley Sunday,Balcony Front,3.0,2.0,no
```
tickets (0 rows): none

### 001i000000NuPZQ  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94708,,Berkeley,,0.0,0.0,0,
subscriptions (1 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
1998-1999,Quartet,2,Berkeley Saturday,Balcony Front,3.0,1.0,no
```
tickets (0 rows): none

### 001i000000NuO3j  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94563,,Orinda,,0.0,1250.0,5,1985-07-04 00:00:00
subscriptions (4 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
1997-1998,Full,2,Contra Costa,Orchestra,2.0,2.0,no
1998-1999,Full,2,Contra Costa,Orchestra,2.0,2.0,no
2000-2001,Quartet CC,2,Contra Costa,Orchestra,2.0,1.0,no
2003-2004,Quartet CC,1,Contra Costa,,,0.5,no
```
tickets (0 rows): none

### 001i000000NuOoH  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94107,,San Francisco,,0.0,50.0,1,1998-12-02 00:00:00
subscriptions (2 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
1996-1997,Full,2,San Francisco,Balcony,4.0,2.0,no
1998-1999,Full,2,San Francisco,Dress Circle,3.0,2.0,no
```
tickets (0 rows): none

### 001i000000NuPHZ  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94040,,Mountain View,,0.0,177.0,2,2000-08-01 00:00:00
subscriptions (2 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
2000-2001,Full,1,San Francisco,Orchestra,2.0,1.0,no
2001-2002,Full,1,San Francisco,Orchestra,2.0,1.0,no
```
tickets (0 rows): none

### 001i000000NuNpr  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94109,,San Francisco,,0.0,0.0,0,
subscriptions (3 rows):
```
season,package,no.seats,location,section,price.level,subscription_tier,multiple.subs
1993-1994,Quartet,2,San Francisco,Box,1.0,1.0,no
1994-1995,CYO,2,San Francisco,Dress Circle,3.0,1.0,no
1995-1996,Quartet,2,San Francisco,Dress Circle,3.0,1.0,no
```
tickets (0 rows): none

### 001i000000NuRUa  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94618,,Oakland,,0.0,0.0,0,
subscriptions (0 rows): none
tickets (3 rows):
```
price.level,no.seats,marketing.source,season,location,set,multiple.tickets
2,2,,2010-2011,Berkeley Saturday,4.0,no
,2,,2011-2012,Berkeley Sunday,7.0,no
,2,,2012-2013,San Francisco,4.0,no
```

### 001i000000NuRBo  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,95624,,Elk Grove,,0.0,0.0,0,
subscriptions (0 rows): none
tickets (1 rows):
```
price.level,no.seats,marketing.source,season,location,set,multiple.tickets
3,2,,2012-2013,Berkeley Sunday,3.0,no
```

### 001i000000NuRLP  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94002,,Belmont,,0.0,0.0,0,
subscriptions (0 rows): none
tickets (1 rows):
```
price.level,no.seats,marketing.source,season,location,set,multiple.tickets
1,1,,2011-2012,Peninsula,6.0,no
```

### 001i000000NuREX  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94708-1144,,Kensington,,0.0,0.0,0,
subscriptions (0 rows): none
tickets (1 rows):
```
price.level,no.seats,marketing.source,season,location,set,multiple.tickets
2,1,,2012-2013,Berkeley Sunday,3.0,no
```

### 001i000000NuRTz  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94612-4134,,Oakland,,0.0,0.0,0,
subscriptions (0 rows): none
tickets (2 rows):
```
price.level,no.seats,marketing.source,season,location,set,multiple.tickets
4,2,,2010-2011,Berkeley Saturday,8.0,no
,3,,2012-2013,Berkeley Sunday,2.0,no
```

### 001i000000NuNk5  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,,,,,0.0,0.0,0,
subscriptions (0 rows): none
tickets (1 rows):
```
price.level,no.seats,marketing.source,season,location,set,multiple.tickets
4,1,,2010-2011,San Francisco,6.0,no
```

### 001i000000NuRge  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,,,,,0.0,0.0,0,
subscriptions (0 rows): none
tickets (0 rows): none

### 001i000000NuPRX  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94070,,San Carlos,,0.0,0.0,0,
subscriptions (0 rows): none
tickets (0 rows): none

### 001i000000NuPvy  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94702,,Berkeley,,0.0,0.0,0,
subscriptions (0 rows): none
tickets (0 rows): none

### 001i000000NuQCB  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,94702,,Berkeley,,0.0,0.0,0,
subscriptions (0 rows): none
tickets (0 rows): none

### 001i000000NuRjU  label=0
account: shipping.zip.code,billing.zip.code,shipping.city,billing.city,relationship,amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated → ,,,,,0.0,0.0,0,
subscriptions (0 rows): none
tickets (0 rows): none


# INCUMBENT CODE (src/final_model.py)
```python
"""Final model: reference blend (pruned logistic + random forest) plus a nested-CV-selected additive spline logistic model.

Code generated during the blind modelling tournament (independent/tournament.py, "Model_Package_3") and kept unchanged except
for this header and default paths. It rebuilds features from the raw Kaggle files, runs 5-outer x 3-inner nested CV, refits on
all training rows and writes a submission.

Run (from the repo root, with the Kaggle files in for_students/):
    python3 src/final_model.py
Outputs in results/final_model/: submission.csv, oof_final.csv, results.json, diagnostics.json.
Environment overrides: DATA_DIR (raw files), OUT_DIR (outputs), BASELINE_SRC (module folder, default src/).
"""
import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
)
from threadpoolctl import threadpool_limits

START = time.monotonic()
SEED = 1729
CPUS = 8
CS = (0.1, 1.0, 10.0, 74.0)
WEIGHTS = (0.25, 0.5)
TOLERANCE = 0.0005
_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("DATA_DIR", str(_ROOT / "for_students"))
os.environ.setdefault("OUT_DIR", str(_ROOT / "results" / "final_model"))
os.environ.setdefault("BASELINE_SRC", str(_ROOT / "src"))
OUT = Path(os.environ["OUT_DIR"])
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ["DATA_DIR"])
sys.path.insert(0, os.environ["BASELINE_SRC"])

from clean import load_account, load_subscriptions, load_tickets, load_zipcodes
import clean as _clean
_clean.RAW = DATA          # read the raw files from DATA_DIR

np.random.seed(SEED)
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

ACTIVE = {
    "San Francisco", "Berkeley Saturday", "Berkeley Sunday", "Peninsula"
}
IDS = ["id_cohort_Lh", "id_newer_than_Nu"]
PRUNE = {"geo_km_sf", "geo_km_berkeley", "geo_km_peninsula"}
CANDIDATE_COLUMNS = [
    "sub_ever", "sub_seasons_since_last", "sub_last_season",
    "sub_n_last3", "sub_n_last5", "sub_n_seasons", "sub_streak",
    "cons_sub_active_share",
    "sub_last_package", "sub_last_seats", "sub_last_venue_active",
    "any_seasons_since_last", "any_n_seasons",
    "don_2013", "don_lifetime", "don_count", "don_years_since_first",
    "don_first_in_2014", "geo_km_nearest_venue",
    "id_cohort_Lh", "id_newer_than_Nu",
]
SPLINE_COLUMNS = {
    "sub_seasons_since_last", "any_seasons_since_last",
    "don_years_since_first", "don_2013", "don_lifetime",
}
LOG_COLUMNS = {
    "sub_n_last3", "sub_n_last5", "sub_n_seasons", "sub_streak",
    "sub_last_seats", "any_n_seasons", "don_count",
    "geo_km_nearest_venue",
}
BROAD_REMOVE = {
    "id_cohort_Lh", "id_newer_than_Nu", "don_lifetime", "don_count",
    "don_years_since_first", "don_first_in_2014",
}

def auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None

def sole_or_multiple(values):
    vals = pd.Series(values).dropna().astype(str).unique()
    if len(vals) == 0:
        return np.nan
    return vals[0] if len(vals) == 1 else "multiple"

def haversine(lat, lon, lat0, lon0):
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    lat0, lon0 = np.radians([float(lat0), float(lon0)])
    a = (np.sin((lat - lat0) / 2) ** 2
         + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def construct_features(all_ids):
    account = load_account()
    subs = load_subscriptions()
    tickets = load_tickets()
    zips = load_zipcodes()

    assert account["account.id"].is_unique
    assert zips["zip5"].is_unique
    audit = {
        "subscription_rows_after_cutoff_removed": int((subs["year"] > 2013).sum()),
        "ticket_rows_after_cutoff_removed": int((tickets["year"] > 2013).sum()),
    }
    subs = subs.loc[subs["year"] <= 2013].copy()
    tickets = tickets.loc[tickets["year"] <= 2013].copy()
    audit["multirow_account_seasons"] = int(
        (subs.groupby(["account.id", "year"]).size() > 1).sum()
    )
    index = pd.Index(all_ids, name="account.id")
    X = pd.DataFrame(index=index)

    records = []
    for account_id, g in subs.groupby("account.id", sort=False):
        years = sorted(set(int(v) for v in g["year"]))
        latest = g.loc[g["year"] == years[-1]]
        previous = g.loc[g["year"] == years[-2]] if len(years) > 1 else None
        ys = set(years)
        streak = 0
        while 2013 - streak in ys:
            streak += 1
        last_seats = float(latest["no.seats"].max())
        last_tier = float(latest["subscription_tier"].max())
        price = pd.to_numeric(latest["price.level"], errors="coerce")
        records.append({
            "account.id": account_id,
            "sub_n_seasons": len(years),
            "sub_n_last3": sum(v >= 2011 for v in years),
            "sub_n_last5": sum(v >= 2009 for v in years),
            "sub_n_locations": g["location"].nunique(),
            "sub_n_multiple": int(g["multiple.subs"].sum()),
            "sub_mean_seats": float(g["no.seats"].mean()),
            "sub_max_tier": float(g["subscription_tier"].max()),
            "sub_mean_price_level": pd.to_numeric(
                g["price.level"], errors="coerce"
            ).mean(),
            "sub_share_active_venue": float(g["location"].isin(ACTIVE).mean()),
            "sub_streak": streak,
            "sub_seasons_since_last": 2014 - years[-1],
            "sub_span": years[-1] - years[0] + 1,
            "sub_gap_seasons": years[-1] - years[0] + 1 - len(years),
            "sub_last_season": int(2013 in ys),
            "sub_prev_season": int(2012 in ys),
            "sub_returned_after_gap": int(
                2013 in ys and 2012 not in ys and len(ys) > 1
            ),
            "sub_last_package": sole_or_multiple(latest["package_group"]),
            "sub_last_seats": last_seats,
            "sub_last_tier": last_tier,
            "sub_last_price_level": price.median() if price.notna().any() else np.nan,
            "sub_last_location": sole_or_multiple(latest["location"]),
            "sub_last_venue_active": int(latest["location"].isin(ACTIVE).any()),
            "sub_last_multiple": int(latest["multiple.subs"].any()),
            "sub_seat_change": (
                last_seats - float(previous["no.seats"].max())
                if previous is not None else np.nan
            ),
            "sub_tier_change": (
                last_tier - float(previous["subscription_tier"].max())
                if previous is not None else np.nan
            ),
            "cons_sub_active_share": len(years) / (2014 - years[0]),
        })
    X = X.join(pd.DataFrame(records).set_index("account.id"))

    ticket_records = []
    for account_id, g in tickets.groupby("account.id", sort=False):
        ticket_records.append({
            "account.id": account_id,
            "tix_n_purchases": len(g),
            "tix_n_seasons": g["year"].nunique(),
            "tix_n_concerts": len(
                g[["year", "set"]].dropna().drop_duplicates()
            ),
            "tix_total_seats": g["no.seats"].sum(),
            "tix_mean_price_level": g["price.level.num"].mean(),
            "tix_any_multiple": int(g["multiple.tickets"].any()),
            "tix_has_marketing_source": int(g["marketing.source"].notna().any()),
            "tix_family_concert": int(g["location"].eq("Family concert").any()),
            "tix_seasons_since_last": 2014 - int(g["year"].max()),
            "tix_last_season": int(g["year"].eq(2013).any()),
        })
    X = X.join(pd.DataFrame(ticket_records).set_index("account.id"))

    a = account.set_index("account.id").reindex(index)
    fd = pd.to_datetime(a["first.donated"], errors="coerce")
    X["don_lifetime"] = a["amount.donated.lifetime"]
    X["don_count"] = a["no.donations.lifetime"]
    X["don_2013"] = a["amount.donated.2013"]
    X["don_any"] = X["don_lifetime"].gt(0).astype(float)
    X["don_2013_any"] = X["don_2013"].gt(0).astype(float)
    X["don_avg_gift"] = (
        X["don_lifetime"] / X["don_count"]
    ).where(X["don_count"] > 0, 0)
    X["don_years_since_first"] = (
        (pd.Timestamp("2014-01-01") - fd).dt.days / 365.25
    ).where(fd.dt.year < 2014)
    X["don_first_in_2014"] = fd.dt.year.eq(2014).astype(float)
    X["has_relationship"] = a["relationship"].notna().astype(float)
    X["has_billing_zip"] = a["billing.zip5"].notna().astype(float)
    X["has_shipping_zip"] = a["shipping.zip5"].notna().astype(float)

    z = zips.set_index("zip5")
    home = z.reindex(a["billing.zip5"].to_numpy()).copy()
    home.index = index
    X["geo_in_CA"] = home["State"].eq("CA").astype(float)
    for name, zipcode in (
        ("sf", "94102"), ("berkeley", "94704"), ("peninsula", "94301")
    ):
        X["geo_km_" + name] = haversine(
            home["Lat"], home["Long"], z.loc[zipcode, "Lat"], z.loc[zipcode, "Long"]
        )
    X["geo_km_nearest_venue"] = X[
        ["geo_km_sf", "geo_km_berkeley", "geo_km_peninsula"]
    ].min(axis=1)
    X["geo_zip_wages_per_return"] = home["wages_per_return"]
    X["geo_zip_population"] = home["EstimatedPopulation"]

    id_series = pd.Series(index.to_numpy(), index=index, dtype="string")
    prefix = "001i000000Lh"
    assert len(prefix) == 12
    # Compare values rather than Series.equals, which also compares dtypes.
    sliced = id_series.str[:len(prefix)].eq(prefix).to_numpy(
        dtype=bool, na_value=False
    )
    starts = id_series.str.startswith(prefix, na=False).to_numpy(
        dtype=bool, na_value=False
    )
    if not np.array_equal(sliced, starts):
        raise AssertionError("Prefix implementations disagree on actual ID values.")
    X["id_cohort_Lh"] = starts.astype(float)
    X["id_newer_than_Nu"] = (
        (id_series > "001i000000Nv").to_numpy(dtype=bool, na_value=False)
    ).astype(float)
    audit["prefix_value_equivalence_verified"] = True
    audit["prefix_length"] = len(prefix)

    zero_prefixes = (
        "sub_n", "sub_streak", "sub_last_season", "sub_prev_season",
        "sub_returned", "sub_share", "sub_last_venue", "sub_last_multiple",
        "tix_n", "tix_total", "tix_any", "tix_has", "tix_family",
        "tix_last_season", "cons_sub_active_share",
    )
    for col in X:
        if col.startswith(zero_prefixes):
            X[col] = X[col].fillna(0)
    for col in ("sub_seasons_since_last", "tix_seasons_since_last"):
        X[col] = X[col].fillna(99)

    history = pd.concat([
        subs[["account.id", "year"]], tickets[["account.id", "year"]]
    ], ignore_index=True).groupby("account.id")["year"]
    X["any_n_seasons"] = history.nunique().reindex(index).fillna(0)
    X["any_seasons_since_last"] = (
        2014 - history.max()
    ).reindex(index).fillna(99)
    X["sub_ever"] = X["sub_n_seasons"].gt(0).astype(float)
    X["tix_ever"] = X["tix_n_purchases"].gt(0).astype(float)

    for col in ("sub_last_package", "sub_last_location"):
        X[col] = X[col].astype(object)
        X.loc[X["sub_ever"].eq(0), col] = "none"
        X[col] = X[col].fillna("unknown").astype("category")

    X["ix_recent_x_don_count"] = X["sub_last_season"] * X["don_count"]
    X["ix_recent_x_tix_seats"] = X["sub_last_season"] * X["tix_total_seats"]

    for col in X:
        if col not in ("sub_last_package", "sub_last_location"):
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    assert X.index.is_unique
    assert len(X) == len(index)
    audit["feature_columns_before_selection"] = list(X.columns)
    audit["cohort_prevalence_all_scored_accounts"] = {
        c: float(X[c].mean()) for c in IDS
    }
    return X, audit

def reference_pool(X):
    prefixes = ("sub_", "tix_", "any_", "don_", "has_", "geo_", "id_", "ix_")
    return X[[c for c in X if c.startswith(prefixes)]]

def select_reference_features(X, y):
    Xn = X.copy()
    for c in Xn:
        if not pd.api.types.is_numeric_dtype(Xn[c]):
            Xn[c] = Xn[c].astype("category").cat.codes.astype(float)
    keep = [c for c in Xn if Xn[c].isna().mean() <= 0.50]
    Z = Xn[keep].copy()
    Z = Z.fillna(Z.median()).fillna(0)
    mi = pd.Series(
        mutual_info_classif(
            Z, y, random_state=SEED,
            discrete_features=[Z[c].nunique() <= 10 for c in Z]
        ),
        index=keep,
    )
    corr = Z.corr(method="spearman").abs()
    candidates = []
    for c in mi.sort_values(ascending=False, kind="stable").index:
        if not any(corr.loc[c, d] >= 0.95 for d in candidates):
            candidates.append(c)
        if len(candidates) == 35:
            break
    selector = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, n_jobs=CPUS, random_state=SEED
    )
    selector.fit(Z[candidates], y)
    ranked = pd.Series(
        selector.feature_importances_, index=candidates
    ).sort_values(ascending=False, kind="stable")
    chosen = [c for c in IDS if c in X]
    chosen += [c for c in ranked.index if c not in chosen][:25 - len(chosen)]
    return [
        c for c in chosen
        if not c.startswith(("tix_", "ix_")) and c not in PRUNE
    ]

def never_flags(X):
    out = X.copy()
    for c in list(out.columns):
        if c.endswith("seasons_since_last"):
            out[c + "_never"] = out[c].eq(99).astype(float)
            out[c] = out[c].where(out[c] != 99)
    return out

def make_reference_scaled(X):
    Z = never_flags(X)
    cats = [c for c in Z if not pd.api.types.is_numeric_dtype(Z[c])]
    nums = [c for c in Z if c not in cats]
    skew = [
        c for c in nums
        if (Z[c].dropna() >= 0).all() and Z[c].skew() > 2
    ]
    other = [c for c in nums if c not in skew]

    def numeric_steps(log):
        steps = []
        if log:
            steps.append(("log", FunctionTransformer(np.log1p)))
        steps += [
            ("impute", SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            )),
            ("scale", StandardScaler()),
        ]
        return Pipeline(steps)

    ct = ColumnTransformer([
        ("skew", numeric_steps(True), skew),
        ("num", numeric_steps(False), other),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
    ], sparse_threshold=0)
    return Pipeline([
        ("never", FunctionTransformer(never_flags)),
        ("columns", ct),
    ])

def make_reference_tree(X):
    cats = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
    nums = [c for c in X if c not in cats]
    return ColumnTransformer([
        ("num", SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        ), nums),
        ("cat", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ), cats),
    ], sparse_threshold=0)

def cdf_map(sorted_training_scores, scores):
    left = np.searchsorted(sorted_training_scores, scores, side="left")
    right = np.searchsorted(sorted_training_scores, scores, side="right")
    return (left + right) / (2.0 * len(sorted_training_scores))

class Reference:
    def fit(self, X, y):
        X = reference_pool(X)
        self.columns = select_reference_features(X, y)
        Z = X[self.columns]
        self.linear = Pipeline([
            ("prep", make_reference_scaled(Z)),
            ("model", LogisticRegression(
                C=74.0, l1_ratio=0.0, solver="saga",
                max_iter=20000, tol=1e-3, random_state=SEED
            )),
        ])
        self.forest = Pipeline([
            ("prep", make_reference_tree(Z)),
            ("model", RandomForestClassifier(
                n_estimators=500, max_depth=4, min_samples_leaf=2,
                max_features=0.4, class_weight="balanced",
                n_jobs=CPUS, random_state=SEED
            )),
        ])
        self.linear.fit(Z, y)
        self.forest.fit(Z, y)
        self.linear_cdf = np.sort(self.linear.predict_proba(Z)[:, 1])
        self.forest_cdf = np.sort(self.forest.predict_proba(Z)[:, 1])
        return self

    def predict(self, X):
        Z = X[self.columns]
        a = cdf_map(self.linear_cdf, self.linear.predict_proba(Z)[:, 1])
        b = cdf_map(self.forest_cdf, self.forest.predict_proba(Z)[:, 1])
        return 0.5 * (a + b)

class AdditivePreprocessor:
    def __init__(self, columns):
        self.columns = list(columns)

    def values(self, X, c):
        arr = pd.to_numeric(X[c], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        ).copy()
        arr[~np.isfinite(arr)] = np.nan
        if c.endswith("seasons_since_last"):
            arr[arr == 99] = np.nan
        if c in {"don_2013", "don_lifetime"}:
            if np.any(arr[np.isfinite(arr)] < 0):
                raise ValueError("Negative donation value violates frozen log transform.")
            arr = np.log1p(arr)
        return arr

    def fit(self, X):
        self.spec = {}
        for c in self.columns:
            if c == "sub_last_package":
                continue
            v = self.values(X, c)
            finite = v[np.isfinite(v)]
            if c in SPLINE_COLUMNS:
                if len(finite):
                    lower = (
                        0.0 if c in {"don_2013", "don_lifetime"}
                        else float(finite.min())
                    )
                    upper = float(finite.max())
                    pool = (
                        finite[finite > 0]
                        if c in {"don_2013", "don_lifetime"} else finite
                    )
                    knots = (
                        np.unique(np.quantile(pool, [0.25, 0.5, 0.75]))
                        if len(pool) else np.array([], dtype=float)
                    )
                    knots = knots[(knots > lower) & (knots < upper)]
                else:
                    lower = upper = 0.0
                    knots = np.array([], dtype=float)
                self.spec[c] = ("spline", lower, upper, knots)
            else:
                if c in LOG_COLUMNS:
                    finite = np.log1p(finite)
                median = float(np.median(finite)) if len(finite) else 0.0
                self.spec[c] = ("linear", median)
        self.scaler = StandardScaler().fit(self.numeric(X))
        self.encoder = None
        if "sub_last_package" in self.columns:
            self.encoder = OneHotEncoder(
                categories=[["Full", "Quartet", "Trio", "CYO",
                             "unknown", "multiple", "none"]],
                handle_unknown="ignore", sparse_output=False,
            ).fit(self.packages(X))
        self.n_columns = self.transform(X).shape[1]
        return self

    def packages(self, X):
        return X["sub_last_package"].astype(object).fillna(
            "unknown"
        ).astype(str).to_numpy().reshape(-1, 1)

    def numeric(self, X):
        blocks = []
        for c in self.columns:
            if c == "sub_last_package":
                continue
            v = self.values(X, c)
            missing = ~np.isfinite(v)
            spec = self.spec[c]
            if spec[0] == "spline":
                _, lower, upper, knots = spec
                v = np.clip(np.where(missing, lower, v), lower, upper)
                blocks.append(v)
                blocks.extend(np.maximum(v - knot, 0.0) for knot in knots)
            else:
                if c in LOG_COLUMNS:
                    v = np.log1p(v)
                blocks.append(np.where(missing, spec[1], v))
            blocks.append(missing.astype(float))
        mat = np.column_stack(blocks)
        if not np.isfinite(mat).all():
            raise ValueError("Nonfinite additive design matrix")
        return mat

    def transform(self, X):
        mat = self.scaler.transform(self.numeric(X))
        if self.encoder is not None:
            mat = np.column_stack([mat, self.encoder.transform(self.packages(X))])
        return mat

    def describe(self):
        return {
            "expanded_columns": int(self.n_columns),
            "spline_specs": {
                c: {
                    "lower": float(spec[1]),
                    "upper": float(spec[2]),
                    "knots": spec[3].tolist(),
                }
                for c, spec in self.spec.items() if spec[0] == "spline"
            },
        }

def fit_additive_scores(Xtr, ytr, Xval, cs=CS, columns=CANDIDATE_COLUMNS):
    prep = AdditivePreprocessor(columns).fit(Xtr)
    A, B = prep.transform(Xtr), prep.transform(Xval)
    output = {}
    info = prep.describe()
    info["iterations"] = {}
    for C in cs:
        model = LogisticRegression(
            C=C, l1_ratio=0.0, solver="lbfgs",
            class_weight=None, max_iter=10000, tol=1e-6,
            random_state=SEED,
        )
        model.fit(A, ytr)
        training = model.predict_proba(A)[:, 1]
        ref = np.sort(training)
        output[float(C)] = {
            "validation": cdf_map(ref, model.predict_proba(B)[:, 1]),
            "training": cdf_map(ref, training),
        }
        info["iterations"][str(C)] = int(model.n_iter_[0])
    return output, info

def make_splits(X, y, n_splits, seed):
    strata = np.asarray(y, dtype=int) * 2 + X["sub_last_season"].to_numpy(dtype=int)
    stratification = "label_x_previous_subscription"
    if pd.Series(strata).value_counts().min() < n_splits:
        strata = np.asarray(y, dtype=int)
        stratification = "label"
    if pd.Series(strata).value_counts().min() < n_splits:
        raise ValueError("Insufficient class counts for frozen validation protocol")
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    return list(splitter.split(np.zeros(len(y)), strata)), stratification

CONFIGS = [(0.0, None)] + [(w, C) for w in WEIGHTS for C in CS]

def config_name(w, C):
    return "reference" if w == 0 else f"additive_weight={w:g},C={C:g}"

def inner_select(X, y):
    splits, stratification = make_splits(X, y, 3, SEED + 1)
    scores = {config_name(w, C): [] for w, C in CONFIGS}
    details = []
    for fold, (tr, va) in enumerate(splits):
        ref = Reference().fit(X.iloc[tr], y[tr])
        rp = ref.predict(X.iloc[va])
        additive, add_info = fit_additive_scores(X.iloc[tr], y[tr], X.iloc[va])
        scores["reference"].append(auc(y[va], rp))
        for w, C in CONFIGS[1:]:
            pred = (1 - w) * rp + w * additive[C]["validation"]
            scores[config_name(w, C)].append(auc(y[va], pred))
        details.append({
            "fold": fold,
            "selected_reference_features": ref.columns,
            "additive_design": add_info,
            "training_positive_count": int(y[tr].sum()),
            "validation_positive_count": int(y[va].sum()),
        })
        del ref

    means = {name: float(np.mean(values)) for name, values in scores.items()}
    best_nonzero = max(means[config_name(w, C)] for w, C in CONFIGS[1:])
    chosen = (0.0, None)
    if best_nonzero - means["reference"] >= TOLERANCE:
        eligible = [
            (w, C) for w, C in CONFIGS[1:]
            if means[config_name(w, C)] >= best_nonzero - TOLERANCE
        ]
        chosen = min(eligible, key=lambda v: (v[0], v[1]))
    return chosen, {
        "stratification": stratification,
        "mean_inner_auroc": means,
        "fold_inner_auroc": scores,
        "choice": config_name(*chosen),
        "fold_details": details,
    }

def bootstrap_difference(y, p, q, seed=SEED, repetitions=500):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(np.asarray(y) == 1)
    neg = np.flatnonzero(np.asarray(y) == 0)
    differences = []
    for _ in range(repetitions):
        idx = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        differences.append(
            roc_auc_score(y[idx], p[idx]) - roc_auc_score(y[idx], q[idx])
        )
    lo, hi = np.quantile(differences, [0.025, 0.975])
    return {
        "difference": float(roc_auc_score(y, p) - roc_auc_score(y, q)),
        "approximate_95_percent_interval": [float(lo), float(hi)],
        "replicates": repetitions,
        "method": "paired account bootstrap, stratified by label",
        "limitation": (
            "Conditional on fitted OOF predictions; does not capture all training dependence."
        ),
    }

def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value

def write_json(name, value):
    with (OUT / name).open("w", encoding="utf-8") as handle:
        json.dump(json_ready(value), handle, indent=2, allow_nan=False)

def main():
    train = pd.read_csv(DATA / "train.csv", dtype={"account.id": str})
    test = pd.read_csv(DATA / "test.csv", dtype={"ID": str})
    assert train["account.id"].is_unique and test["ID"].is_unique
    assert train["account.id"].notna().all() and test["ID"].notna().all()
    assert set(train["account.id"]).isdisjoint(set(test["ID"]))
    assert set(train["label"].unique()) == {0, 1}
    y = train["label"].to_numpy(dtype=int)

    all_ids = train["account.id"].tolist() + test["ID"].tolist()
    features, audit = construct_features(all_ids)
    X = features.loc[train["account.id"]].copy()
    XT = features.loc[test["ID"]].copy()
    assert X.index.tolist() == train["account.id"].tolist()
    assert XT.index.tolist() == test["ID"].tolist()
    print(f"Features constructed: {len(X)} training, {len(XT)} test accounts.", flush=True)

    outer, stratification = make_splits(X, y, 5, SEED)
    oof = np.full(len(y), np.nan)
    reference_oof = np.full(len(y), np.nan)
    additive_c1_oof = np.full(len(y), np.nan)
    flag_ablation_oof = np.full(len(y), np.nan)
    broad_ablation_oof = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1, dtype=int)
    folds = []

    for fold, (tr, va) in enumerate(outer):
        chosen, inner_info = inner_select(X.iloc[tr], y[tr])
        w, C = chosen
        reference = Reference().fit(X.iloc[tr], y[tr])
        rp = reference.predict(X.iloc[va])
        rt = reference.predict(X.iloc[tr])
        reference_oof[va] = rp

        requested_cs = sorted(set([1.0] + ([C] if C is not None else [])))
        add, add_info = fit_additive_scores(
            X.iloc[tr], y[tr], X.iloc[va], cs=requested_cs
        )
        additive_c1_oof[va] = add[1.0]["validation"]
        if w == 0:
            prediction, training_prediction = rp, rt
        else:
            prediction = (1 - w) * rp + w * add[C]["validation"]
            training_prediction = (1 - w) * rt + w * add[C]["training"]
        oof[va] = prediction
        fold_ids[va] = fold

        no_flag = [c for c in CANDIDATE_COLUMNS if c != "don_first_in_2014"]
        diagnostic, _ = fit_additive_scores(
            X.iloc[tr], y[tr], X.iloc[va], cs=[1.0], columns=no_flag
        )
        flag_ablation_oof[va] = diagnostic[1.0]["validation"]
        broad_columns = [c for c in CANDIDATE_COLUMNS if c not in BROAD_REMOVE]
        diagnostic, _ = fit_additive_scores(
            X.iloc[tr], y[tr], X.iloc[va], cs=[1.0], columns=broad_columns
        )
        broad_ablation_oof[va] = diagnostic[1.0]["validation"]

        hard = X.iloc[va]["sub_last_season"].eq(0).to_numpy()
        folds.append({
            "fold": fold,
            "inner_selection": inner_info,
            "selected_reference_features": reference.columns,
            "additive_design": add_info,
            "final_validation_auroc": auc(y[va], prediction),
            "reference_validation_auroc": auc(y[va], rp),
            "final_training_auroc": auc(y[tr], training_prediction),
            "reference_training_auroc": auc(y[tr], rt),
            "validation_rows": len(va),
            "validation_positive_count": int(y[va].sum()),
            "hard_group_rows": int(hard.sum()),
            "hard_group_positive_count": int(y[va][hard].sum()),
            "hard_group_auroc": auc(y[va][hard], prediction[hard]),
        })
        print(
            f"Outer fold {fold + 1}/5 complete; "
            f"elapsed {(time.monotonic() - START) / 60:.1f} minutes",
            flush=True,
        )
        del reference

    assert np.isfinite(oof).all()
    assert (fold_ids >= 0).all()

    # The same frozen inner selection rule is applied to all supplied training rows.
    final_choice, final_inner = inner_select(X, y)
    w, C = final_choice
    reference = Reference().fit(X, y)
    test_reference = reference.predict(XT)
    if w == 0:
        test_prediction = test_reference
        final_add_info = None
    else:
        final_add, final_add_info = fit_additive_scores(X, y, XT, cs=[C])
        test_prediction = (
            (1 - w) * test_reference + w * final_add[C]["validation"]
        )
    assert len(test_prediction) == len(test)
    assert np.isfinite(test_prediction).all()
    assert np.all((test_prediction >= 0) & (test_prediction <= 1))

    pd.DataFrame({
        "account.id": train["account.id"].to_numpy(),
        "pred": oof,
    }).to_csv(OUT / "oof_final.csv", index=False)
    pd.DataFrame({
        "ID": test["ID"].to_numpy(),
        "Predicted": test_prediction,
    }).to_csv(OUT / "submission.csv", index=False)

    hard = X["sub_last_season"].eq(0).to_numpy()
    model_name = "nested_reference_additive_blend"
    write_json("results.json", {
        "models": {
            "corrected_reference": {
                "cv_auroc": auc(y, reference_oof),
                "notes": (
                    "Outer OOF evaluation of frozen corrected logistic/forest "
                    "reference with fold-local selection and training-score CDFs."
                ),
            },
            model_name: {
                "cv_auroc": auc(y, oof),
                "notes": (
                    "Outer OOF evaluation of the complete inner-selection rule. "
                    f"Full-training inner choice: {config_name(*final_choice)}. "
                    "Outer results and sensitivity diagnostics do not alter selection."
                ),
            },
        },
        "best": model_name,
        "validation": (
            "Five fixed outer folds and three inner folds; joint label/history "
            "stratification where feasible; no outer-score model selection."
        ),
    })

    diagnostics = {
        "seed": SEED,
        "elapsed_seconds": time.monotonic() - START,
        "training_rows": len(train),
        "test_rows": len(test),
        "training_positives": int(y.sum()),
        "outer_stratification": stratification,
        "audit": audit,
        "frozen_additive_source_features": CANDIDATE_COLUMNS,
        "inverse_penalty_grid": CS,
        "additive_weight_grid": [0.0, *WEIGHTS],
        "selection_tolerance": TOLERANCE,
        "outer_folds": folds,
        "outer_mean_final_auroc": float(np.mean([
            f["final_validation_auroc"] for f in folds
        ])),
        "outer_mean_reference_auroc": float(np.mean([
            f["reference_validation_auroc"] for f in folds
        ])),
        "paired_oof_difference": bootstrap_difference(y, oof, reference_oof),
        "hard_group": {
            "rows": int(hard.sum()),
            "positives": int(y[hard].sum()),
            "final_auroc": auc(y[hard], oof[hard]),
            "reference_auroc": auc(y[hard], reference_oof[hard]),
        },
        "diagnostic_only_ablations": {
            "fixed_C": 1.0,
            "full_additive_auroc": auc(y, additive_c1_oof),
            "without_first_gift_flag_auroc": auc(y, flag_ablation_oof),
            "without_cohorts_and_uncertain_donations_auroc": auc(
                y, broad_ablation_oof
            ),
            "flag_removal_difference": float(
                auc(y, flag_ablation_oof) - auc(y, additive_c1_oof)
            ),
            "broad_removal_difference": float(
                auc(y, broad_ablation_oof) - auc(y, additive_c1_oof)
            ),
            "used_for_selection": False,
        },
        "full_training_inner_selection": final_inner,
        "final_reference_features": reference.columns,
        "final_additive_design": final_add_info,
        "limitations": [
            "Training-score CDFs may differ from unseen score distributions.",
            "Outputs are bounded ranking scores, not calibrated probabilities.",
            "Bootstrap intervals omit some training and fold dependence.",
            "Donation snapshots and record cohorts have temporal transportability risks.",
            "Historical development choices may contain inherited selection optimism.",
            "Maximum within-season seats is a deterministic summary, not a verified total.",
            "No hidden labels or leaderboard outcomes are read.",
        ],
    }
    write_json("diagnostics.json", diagnostics)
    print("Required outputs written.", flush=True)

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

```
