## Introduction
There are two basic models:

#### LSTM for Evaluating Polynomial at one point
First model map polynomial coefficients and a point, to the value of the
   polynomial at that point. This is just an LSTM. To run

```bash
poly_eval.py --train_degrees "5,10,15" --valid_degrees "20,25"
```

This will train the model on degree 5, 10, 15, and test on degree 20, 25. It
will report the R-Squared

#### Recursive LSTM for mapping polynomial coefficients to its roots
Second model map the polynomial coefficients to its (3 smallest) roots. 
To run

```bash
python recursive_root.py --train_degrees "5,7,9,11,13,15" --valid_degrees "17,19"
```

This will train the model to find roots on degree 5,7,9,11,13,15 and test the performance on polynomial degree 17 and 19.

#### Check the presentation inside pdf/ for more details
