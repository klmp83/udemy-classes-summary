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
+ Represent a point in space
- Vector of length two represent locations in 2D matrix
- 





# References
- https://github.com/jonkrohn/ML-foundations
