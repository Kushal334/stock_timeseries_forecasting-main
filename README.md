# Stock Market Prediction Capstone

## Capstone Project Proposal Review Link

* https://review.udacity.com/#!/reviews/3019029

## Submission Files

> All files are in the main repo directory

1. `ts.ipynb` is the main submission file which contains all the code
2. `proposal.md` and `proposal.pdf` contains the project proposal 
3. `report.md` and `report.pdf` contain the final report for this submission 
4. `*.pickle` files are all the output data to avoid needing to make calls to the API.

## Technical Requirements

* Facebook Prophet Library  https://facebook.github.io/prophet/docs/installation.html
* Tensorflow 2.5 (Keras)
* Statsmodels 
* Pandas 
* Scikit-Learn
* AlphaVantage Python  https://github.com/RomelTorres/alpha_vantage 

**NOTE**

>  You do not need to use the AlphaVantage API or make the calls. All the data from each function (API call) has been saved and pickled and provided in the repo.

Just start from the **Read Pickle Data** section

```python

ss_price = pickle.load(open('strong.pickle', 'rb'))
sp_price = pickle.load(open('poor.pickle', 'rb'))
ss_fund = pickle.load(open('strong_fundamental.pickle', 'rb'))
sp_fund = pickle.load(open('poor_fundamental.pickle', 'rb'))
sector = pickle.load(open('strong_sp.pickle', 'rb'))
ss_ti = pickle.load(open('strong_ti.pickle', 'rb'))
sp_ti = pickle.load(open('poor_ti.pickle', 'rb'))
overview_s = pickle.load(open('fundamental_overview_s.pickle', 'rb'))
overview_p = pickle.load(open('fundamental_overview_p.pickle', 'rb'))
```

