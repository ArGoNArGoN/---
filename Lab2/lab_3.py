import numpy as np
import scipy.io
from sklearn import svm
from collections import OrderDict

from process_email import process_email
from process_email import email_features
from process_email import get_gictionary

with open('email.txt', 'r') as file:
    email = file.read().replace('\n', '');
