# DM
## Preprocessing
1. select out highlt co-related features, and draw a figure
## Feature Selection
1. CHI
2. Multi-Info
3. ANOVA

The selected features are seen 'Reduction_Result.csv'

## Models
1. knn: not shown in result
2. random forest
3. neural network, could be replaced by SVC, but I keep the code here

All the above three could be seen in '.ipynb'. BUT 

5. SVC
6. LightGBM  

there are some error with SVC and LightGBM while using jupyter(seems something wrong with the interpreter)

Just use SVC.py and LightGBM.py, their results are stored in 'lgb_result.csv' and 'lgb_result.csv'

## vote
there are only three methods' result in Dataframe(in Jupyter), and the other two set of results stored in 'lgb_result.csv' and 'lgb_result.csv' are emerged to result.csv manually :(

all outcomes of models are stored in 'result.csv', and our vote outcome is stored in 'vote.csv'

copy vote outcome to 'sample_submission.cvs', and submit it to kaggle.

## Some Changes
1. have deleted code that cannot run on Jupiter
2. Use reduction data set(??)
3. turn around to use original data set again.... But keep 2. in annotation
#_PS_
Since using reduced features,the score on kaggle is even worse

Maybe we need to use the ORIGINAL DATASET but still describe reduction thing on report

Maybe it's because my evaluation method, we could check again/rerun after merging & error analysis

![image](pic.jpg)


