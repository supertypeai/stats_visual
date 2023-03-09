# stats_visual

## Decsription
`stats_visual` is python package to visualize various statistical distributions. With this package, it's possible to generate distribution plots that include highlighted probability mass or density functions, as well as cumulative probabilities. The plots are produced using the Plotly library, which allows for an interactive user experience.

## Installation
`!pip install stats-visual`

## Requirements
`numpy`  
`plotly`  
`scipy`  

## Example Usage  
```
from stats_visual import distribution
binom = distribution.BinomialDist(n=20, p=0.45)
binom.calc_cum_p(k=12)
```
<img src="https://github.com/supertypeai/stats_visual/blob/master/images/binomial_cum_p.png" alt="drawing" width="800"/>

## Demo Notebook
https://colab.research.google.com/drive/1ulY7fWKu-n8gMntwvZZxZMoBTY_8pEKA?usp=sharing

## API Reference
https://github.com/supertypeai/stats_visual/wiki/API-Reference
