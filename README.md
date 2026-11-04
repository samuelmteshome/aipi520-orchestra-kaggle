# Samuel Teshome Kaggle competition ReadMe and Writeup

## Overview
Below is an overview of the ensemble model used in my final Kaggle submission
It combines logistic regression, a random forest, and a spline logistic regression
Training this model on 6,941 labeled accounts to predict which of the 2,975 test accounts would subscribe during the 2014-15 season. 

## Repository structure
- 'src/clean.py' - loads and cleans raw files
- 'src/features.py' -- builds account-level features in experimental data pipeline
- 'src/select_features.py' - per-fold feature selection
- 'src/final_model.py - final ensemble model to be used for submission; builds it's own features and selection, imports clean.py.
- 'eda/' - exploratory analysis outcomes
- 'requirements.txt' - requirement library
  
## How to run
Put competition files in 'for_students/'
'pip install -r requirements.txt'
'python3 src/final_model.py' --> 'results/final_model/submission.csv' (around 1 minute)

## Model Short-Hand 
Acronym - Model 
  LR - Logistic regression
  RF - Random-Forest
  SP - Spline

## Data pipeline
The pipeline combines the raw csv files into one feature row per account. Data that was cleaned includes addresses encoding, inconsistent formats, and missing information. account.csv is read using Latin-1 encoding. ZIP codes are standardized to be five character strings. Postal codes containing letters with no relevant numbers were considered as missing. Ticket prices used numeric labels as well as strings such as Adult and Youth, so numeric levels and text labels are separated rather than attempting to ordinally encode them in one feature.

This exploratory data analysis (found in 'eda/folder') was used to select promising metrics/associations heuristically, such as subscribers, locations that may be closed based on subscriber informaiton, and streaks of subsscriptions and customer momentum. I saw previous subscription status was associated with an  82% chance of 2013-14 subscribers subscribing again again. Recency, frequency, and streak (how many years in a row someone subscribed) features were engineered to capture the momentum of subscribers going into the test season.
  
## Engineered Feature Dictionary 
'subscriptions.csv'
    Recent subscription
        [LR + RF]
            'sub_prev_season'- binary marker for if that individual subscriped during the 2012/13 season
        [SP]
            'sub_last_season'  - binary2013/14 marker which was the strongest signal
        All models
            'sub_seasons_since_last'- leasonns since last subscription. 1 = last season 5 = five seasons ago; 99 = never

    Subscription loyalty/frequency
        Used by all models
            'sub_streak' [All models] - 
            'sub_n_last3' and 'sub_n_last5' - how many of the last 3 or 5 seasons they subscribed
        Used in SPline model
            'sub_n_seasons' - total seasons ever subscribed from 1993 to 2014
            'sub_ever' - binary marker if they've ever subscribed
            'cons_sub_active_share'- % of seasons attended since their first subscription. 1.0 = never missed one.

    Previous purchase (source: subscriptions.csv)
        sub_last_package [All models]: Seat package type: Full (all concerts) / Quartet / Trio / CYO(choose your own) / none.
        sub_last_seats [SP]: number of seats purchased during  their last subscription.

    Venue information
        sub_last_venue_active [SP] - if venue series they subscribed to last was still open in most recent season (as determined by overall subscription data)
        sub_share_active_venue [LR and RF]: aggregate all their subscriptions that were at still-running venues.
 
'subscriptions.csv' + 'tickets_all.csv'
    All-models
        any_seasons_since_last: seasons since they bought anything at all. identifies ticket-only patrons. Bent with a spline in [Sp].
        any_n_seasons: number of seasons where they bought anything.

'account.csv'
    All models
         don_2013: dollars donated in  2013. Bent using spline in [Sp].
         don_count: # of lifetime donations.
         don_first_in_2014: Binary 1/0 if their first gift was in 2014. 
    SP
         don_lifetime: lifetime dollars donated. Bent using spline in [Sp].
         don_years_since_first: years since their first gift. Bent using spline[Sp].


'account.csv' + 'zipcodes.csv' 
    location was cross referenced so billing zipcdoe and shipping address from account.csv was matched to zipcode from zipcode.csv to ensure that only 5 digits were extracted
        All models
        geo_km_nearest_venue [R, S]: distance in km from the billing zipcode centre to the nearest open venue (specifically SF, Berkeley, Palo Alto).

        [LR + RF]
        geo_zip_population: population of the billing zip.
        geo_zip_wages_per_return: average wages per tax return in that zip, a neighbourhood income measure.
        has_shipping_zip: binary if the account has a separate shipping address ( 1.4% fall in this cat).

train.csv
    id_cohort_Lh: coded as 1 if the ID starts with 001i000000Lh, a batch of current subscribers.
    id_newer_than_Nu: 1 if the ID was created after the main batch 

## Feature selection

Feature Selection Process:
    - Removed features with over 50% missingness
    - Identified features that were promissing by looking at different in mean between labeled training groups
    - Removed features that were co-correlated and heuristically carried the same information (Spearman correlation > 0.95) using mutual_info_classif function in scikit-learn
    - Random forest used to narrow candidate features - ranks features based on how much they help the branches distinguish classes. Highest-ranked features from remaining (after the above features) 
    - Fixed pruning rule used after forest selection to remove
       - Direct ticket features like purchase counts, ticket seats, and recent ticket purchase
        explicit interactions multiplying previous-sesason subscription status by donation count or ticket seats
        seperate distances to SF, Berkeley, and Peninusla
       - The pruning rule comes from earlier permutation analysis and validation experiments on the same training dataset. This experiment suggested direct ticket features, explicit interactions, and separate venue distances did not add information to the model and thus they were excluded before the final model.
Temporary selection forest is trained before predictive models. Features are not removed afterward based on outer-validation scores. ultimatly 18 features survived to the inal full data fit. 

LR and RF - 18 features
    Subscription history:
        sub_seasons_since_last, sub_streak, sub_n_last3, sub_n_last5, sub_prev_season, sub_share_active_venue, sub_last_package, 
    Donations:
        don_2013, don_count, don_first_in_2014
    Combined Purchase History
        any_seasons_since_last, any_n_seasons 
    Geography
        geo_km_nearest_venue, geo_zip_population, geo_zip_wages_per_return
    Account indicators
        id_cohort_Lh, id_newer_than_Nu, has_shipping_zip

SP - 21 fixed inputs (applied to five continuous variables)
 Subscription history:
        sub_ever, sub_seasons_since_last, sub_last_season, sub_n_last3, sub_n_last5, sub_n_seasons, sub_streak, cons_sub_active_share, sub_last_package, sub_last_seats, sub_last_venue_active
    Combined Purchase History: 
        any_seasons_since_last, any_n_seasons
    Donations
        don_2013, don_lifetime, don_count, don_years_since_first, don_first_in_2014
    Geography
        geo_km_nearest_venue
    Account Indicators
        id_cohort_Lh, id_newer_than_Nu

SPline - Logistic regression that allows inputs to have changing slopes. One such example is subscription recency having a steep initial effect that goes away after many absent seasons.-  did not use the LR/RF selection process described above, and instead utilized list of 21 features (see above) that was specified before CV run and does not change fold. to fold. Model learns its coefficients and spline bend points from each fold's trianing data
    Spline was applied to
        subscription recency
        combined-purchase recency
        2013 donations
        lifetime donations
        giving tenure

## Modelling

Logistic Regression - Hyperparameter (C) =74
Random Forest - 500 trees, depth of 4, minmum leaf size 2, 40% of features per split
SPline model - c = 1
    Captures nonlinear effects of donations and recency

An earlier search randomly sampled 24 configurations for LR and RF and chose hyperparameters using AUROC across 5 CV folds (src/train.py); this is how the above hyperparameters were selected and they were held constant bfore final nested-CV run. Three fold inner CV selected spline's regularization and contribution to ensemble model using a predfined rule explaineind in the validation section

Adding MLP did not improve the pruned Logistic-forest blended model. Stronger regularization was generally neutral or worse and seperate hard group model reduced performance. This then lead to testing shallow, regularized grdadeint boosting as another ensemble member. All three comparison pipelines had zero boosting weight on the training data. Thus the LR-RF-spline ensemle was selected because it balanced complexity with accuracy. When each comparison pipeline reran it's selection procedure using labeled accounts with internal CV, it chose to give the boostin model no contribution to hte ultimate prediction. Spline improved the LR + RF reference from 0.9845 to 0.9851 AUROC. The MLP did not improve the earlierF blend, and boosting experiments showed no additional improvement over the tri-model ensemble. I thus kept the spline addition and excluded those alternatives.

Predicitions for each model are converted to ranked against each model's training score distribution 

MLP ensemble:
    LR and RF = 0.5 x rank(logistic) + 0.5 x rank(forest)
    Final = 0.5 x reference + 0.5 x rank(spline logistic)

Ensemble applies W9 bagging through the RF model, averaging within reference and weighted average


## Validation

Nested cross-validation was used with five outer folds to assess performance and three inner with each outer to choose spline settings. Accounts were stratified by subscription outcome and previous seasons subscription history. Seperatating selection from evaluation reduces the optimism that comes from the same validation techinque being used to choose a model and report it's performance

The inner CVV test looked at 4 spline settings and contributions from spline of 0, 25, and 50 percent. Smaller C values constrain the fitted relationship more strongly. Predefined rules let the spline addition if it's tested combination improves inner-validation AUROC above the reference by 0.00005. WIth this framework spline performed the best with the smallest possible C hyperparameter. When applied to all training data this rule selected 50 percent contribution and c=1. Outer folds then looked at the complete selection procedure (without actually choosing the procedure).

The selected package performed with a 0.99155 AUROC on the hidden test data.

Final procedure uses 5 outer x 3 inner folds, stratified by outcome and previous subscription status

## Results

| Model               | CV AUROC | Public AUROC |
-------------------------------------------------
| Logistic Regression | 0.9821   |    0.9717    |
| Random Forest.      | 0.9801   |    0.97679   |
| Ensemble            | 0.9797   |    0.97153   |
| Three model average | 0.9825   |    0.97338   |
| Pruned LR+RF        | 0.9835   |       -      |
| Corrected Reference | 0.9845   |       -      |
| Ensemble            | 0.9851   |    0.97991   |

the initial rows use five repetitions fo fvive-fold CV with hyperparameters selected using the first repetition. Because tuning and evaluation overlapped there may be overfitting here. the final reference and ensemble used outer folds so their CV values are not comparable to previous rows.



## Limitations and decisions
The creation of features that used account-ID numerical coding to estimate status may reflect an administrative process and not true subscription signal in a real world setting. 

Donation timing also extends to sepetember 2014 which 

## Use of AI
AI supported code generation, exploratory checks, model proposals, and comparisons