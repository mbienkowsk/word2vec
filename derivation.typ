

Derivative of sigmoid:
$ d/( d x) sigma(x) = $
$ d/( d x) ( 1/(1+e^(-x)) ) = $
$ = d / (d x) (1 + e^(-x))^(-1) = $
$ =-1(1+e^(-x))^(-2) dot d/(d x) (1 + e^(-x))= $
$ = -1(1+e^(-x))^(-2) dot d /(d x) e^(-x)= $
$ = -1(1+e^(-x))^(-2) dot -e^(-x) = (e^(-x))/(1+e^(-x))^2 = $
$ = e^(-x) / (1 + e^(-x)) dot 1 / (1 + e^(-x)) = (1-sigma(x)) dot sigma(x) $


Derivative of the log-loss positive term:

$ d/(d w) log(sigma(w dot w')) = $
$ 1/(sigma(w dot w')) dot d/(d w) sigma(w dot w') = $
$
  = ( 1/(sigma(w dot w')) ) dot (1 - sigma(w dot w')) dot sigma(w dot w') dot d/(d w) (w dot w') =
$
$ = (1 - sigma(w dot w')) dot w' = sigma(-w dot w') dot w' $

Derivative of the log-loss negative term:
$ d/(d w) log(sigma(-w dot w')) = $
$ 1/(sigma(-w dot w')) dot d/(d w) sigma(-w dot w') = $
$
  = ( 1/(sigma(-w dot w')) ) dot (1 - sigma(-w dot w')) dot sigma(-w dot w') dot d/(d w) (-w dot w') =
$
$
  = (1 - sigma(-w dot w')) dot -w'
$
$
  = -sigma(w dot w') dot w'
$

Sigmoid identity used to simplify both log losses:

$ sigma(-x) = 1 - sigma(x) arrow 1 - sigma(-x) = sigma(x) $
