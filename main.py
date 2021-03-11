# Tool box

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # removes start information when run
import tensorflow as tf  # import library

# INITIALISATION OF TENSORS

x = tf.constant(4, shape=(1,1))     # shape is used to define matrix     1,1 matrix = scalar value


x = tf.constant([[1,2,3], [4, 5 ,6]])       # creates a 2 by 3 matrix, 2 dimensional tensor


x = tf.ones((3,3))  # creates a 3 by 3 matrix with the value of 1


x = tf.zeros((3,3))  # creates a 3 by 3 matrix with the value of 0


x = tf.random.normal((3,3), mean=3, stddev=1)   # creates standard normal distribution


x = tf.random.uniform((1, 3), minval=0, maxval=1)    # creates random list between min and max values
print(x)


x = tf.range(9)  # creates a vector list with 9 values
print(x)

x = tf.range(start=1, limit=10, delta=2)  # creates a vector list from start to limit, delta is the step- the list will go up 2 values each time
print(x)


# OPERATIONS

x = tf.constant([1, 2, 3])
y = tf.constant([9, 8, 7])

z = tf.add(x, y)        # adds the 2 lists
z = x + y               # can also be done like this
print(z)

u = tf.subtract(x, y)   # subtracts the 2 lists
u = x - y              # can also be done like this
print(u)

d = tf.divide(x, y)     # divides 2 lists
d = x / y               # can also be done like this
print(d)

m = tf.multiply(x, y)     # multiplies 2 lists
m = x * y               # can also be done like this
print(m)

t = x ** 5
print(t)

a = tf.random.normal((2, 3))
b = tf.random.normal((3, 4))

z = tf.matmul(a, b)     # matrix multiply
print(z)

# INDEXING

x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])
print(x[:])                          # ':' prints all elements
print(x[1:])                        # prints all elements apart form 1st
print(x[1:3])                       # prints values between 1, 3 (excludes 3rd value)
print(x[::2])                       # prints every second value in the list
print(x[::-1])                      # prints list in reverse order

indices = tf.constant([0,3])
x_ind = tf.gather(x , indices)
print(x_ind)                            # prints certain indices from list

x = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]])               # 2 by 3 dimensional matrix

print(x[0])                             # prints first row


# RESHAPING

x = tf.range(9)
x = tf.reshape(x, (3,3))        # reshapes vector to 3 by 3 dimensions
print(x)

x = tf.transpose(x, perm=[1,0])     # changes row values to column vice versa
print(x)
