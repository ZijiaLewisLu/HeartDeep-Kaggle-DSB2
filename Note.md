# HeartDeepNet

## mxnet
- initialiser api?
- MAERegression

## paper
- deep network with internal selective attention
- entropy

## Top Solution
why (source-128)/128

## Second Solution
- how does net decide whether to use a model
- How to average over ensembles
  - KL divergence and cross entropy
- How to use two datasets

### Rebuild Process
- pathfinder.py  
- ​

## Kaggle Tutorial
- 研究tutorial accuracy.log and log.txt



## Caffe Note

- net -> blob : dict
  - keys: names of layers
  - values: blob object
    - attrs: channels, shapes, data, diff, count ….
    - -> data: Matice Like (N\*C\*H\* W)
    -  