# GPU-based BLEU Computation

* Based on Theano
* Super fast for computing bleus in batch

### Usage

```python
bleu = GPUBLEUEvaluator()
# A sentence can be a list of number or string
sents = [
    [1, 2, 3, 4, 5, 3, 4, 1], 
    [2, 2, 2, 5, 7], 
    [1, 2, 3, 3, 4, 5, 1, 5, 6, 7, 4, 3, 3, 6]
]
ref = [[1, 2, 3, 4, 5, 7]]
print (bleu.evaluate(sents, ref))
```
