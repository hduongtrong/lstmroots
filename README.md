## Introduction
There are two basic models:
1. First model map polynomial coefficients and a point, to the value of the
   polynomial at that point. This is just an LSTM. To run
> python poly_eval.py --train_degrees "5,10,15" --valid_degrees "20,25"
This will train the model on degree 5, 10, 15, and test on degree 20, 25. It
will report the R-Squared
2. Second model map the polynomial coefficients to its (3 smallest) roots. 
To run
> python recursive_root.py --train_degrees "5,7,9,11,13,15" --valid_degrees "17,19"
