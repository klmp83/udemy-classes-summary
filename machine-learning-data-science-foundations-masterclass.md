# Section 1: Data Structure for Linear Alebra

### References colab
https://colab.research.google.com/github/jonkrohn/ML-foundations/blob/master/notebooks/1-intro-to-linear-algebra.ipynb#scrollTo=SgUvioyUz8T2

### Scalars
```
import tensorflow as tf
x_tf = tf.Variable(25, dtype=tf.int16)
x_tf.shape
y_tf = tf.Variable(3, dtype=tf.int16)
x_tf + y_tf
tf_sum = tf.add(x_tf, y_tf)
tf_sum.numpy()
type(tf_sum.numpy())
tf_float = tf.Variable(25., dtype=tf.float16)
```

### Vectors and Vctor Transposition
- One dimentional array of numbers
- Denoted in lowercase, italics, bold, e.g.: x
- Represent a point in space: Vector of length two represent locations in 2D matrix
```
x = np.array([25, 2, 5]) # type argument is optional, e.g.: dtype=np.float16
len(x)
x.shape
type(x)
x[0] # zero-indexed
type(x[0])
```

```
# Transposing a regular 1-D array has no effect...
x_t = x.T

# ...but it does we use nested "matrix-style" brackets: 
y = np.array([[25, 2, 5]])
```





# References
- https://github.com/jonkrohn/ML-foundations
