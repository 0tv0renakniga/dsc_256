import numpy
import urllib
import scipy.optimize
import random
from math import exp
from math import log

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse255/data/amazon/book_images_5000.json"))
print "done"

def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
  return 1.0 / (1 + exp(-x))

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    loglikelihood -= log(1 + exp(-logit))
    if not y[i]:
      loglikelihood -= logit
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  print "ll =", loglikelihood
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
  dl = [0]*len(theta)
  for i in range(len(X)):
    # Fill in code for the derivative
  # Negate the return value since we're doing gradient *ascent*
  return numpy.array([-x for x in dl])

X = [b['image_feature'] for b in data]
y = ["Children's Books" in b['categories'] for b in data]

# Training data
X_train = X[:1000]
y_train = y[:1000]

# Test data
X_test = X[1000:]
y_test = y[1000:]

theta,l,info = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X[0]), fprime, args = (X_train, y_train, 1.0))
print "Final log likelihood =", -l
