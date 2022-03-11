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

### Norms and Unit Vectors
#### Norms
- Norms are class of functions that allow us to quantify the magnitude, the length of a given vector
- L2 Norm is important: sqrt(sum(x^2))
- Measure simple (Euclidean) distance from origin
- expression: ||x|| or ||x||^2
```
x # array([25, 2, 5])
(25**2 + 2**2, 5**2)**(1/2)
np.linalg.norm(x)
```
#### Unit Vectors
- Special case of vector where its length is equal to one
- L1 Norm: sum(abs(x))
```
np.abs(25) + np.abs(2) + np.abs(5)
```
#### Max Norms
- max(abs(x))
#### L^p Norm
- sum(abs(x)^p)^(1/p)
- p must be grater than or equal to 1

### Basis, Orthogonal, and Orthonormal Vectors
#### Basis Vectors
- Can be scaled to represent any vector in a given vector space
- Typically use unit vectors along axes of vector space

#### Orthogonal Vectors
- x and y are orthogonal vectors if (x^t)y = 0
- are at 90 degree angle to each other
- n-dimentilanl space has max n mutually orthogonal vectors

### Matrix Tensors
- Two dementional array of numbers
- Denoted in uppercase, italics, bold, e.g: X
- Colon represents and entire row of colum
-- Left column of matrix X is X(subscript of :,1)
-- Middle row of matrix X is X(subscript of 2,:)

### Generic Tensor Notation
- Uppercase, bold, italics, sans serif, e.g.: X
- In a 4 tnesors X, element a position (i,j,k,l)O denoted as X(subscript of (i,j,k,l))


# Section 2: Tensor Operations

### Segment intro
- Data Structures for algebra
- Common Tensor Operations
- Matrix Peroperties

### Tensor Transposition
- Transpose of scala is itself
- Transpose of vector converts columns to row (and vice versa)
- X(superscript of T)(subscript of i,j) = X(subscript of j,i)

### Tensor Reduction
- X.sum(), X.sum(axis=0) # summjming over all columns, tf.reduce_sum(X_tf, 0)

### The Dot Product
- np.dot(x, y)





# Terms
- Superscript
- Subscript

# References
- https://github.com/jonkrohn/ML-foundations
