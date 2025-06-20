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
data = list(parseData("http://jmcauley.ucsd.edu/cse255/data/amazon/book_descriptions_50000.json"))
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
  dl = [0.0]*len(theta)
  for i in range(len(X)):
    # Fill in code for the derivative
    pass
  # Negate the return value since we're doing gradient *ascent*
  return numpy.array([-x for x in dl])

D_child = [d for d in data if "Children's Books" in d['categories']]
D_notchild =\
  [d for d in data if not("Children's Books" in d['categories'])][:len(D_child)]

data = D_child + D_notchild
random.seed(0)
random.shuffle(data)

X = [[1, "child" in d['description'], "magic" in d['description'], "funny" in d['description']] for d in data]
y = ["Children's Books" in d['categories'] for d in data]

X_train = X[:len(X)/2]
X_valid = X[len(X)/2:3*len(X)/4]
X_test = X[3*len(X)/4:]

y_train = y[:len(y)/2]
y_valid = y[len(y)/2:3*len(y)/4]
y_test = y[3*len(y)/4:]

theta,l,info = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X[0]), fprime, args = (X_train, y_train, 1.0))
print "Final log likelihood =", -l

