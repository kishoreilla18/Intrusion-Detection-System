from django.shortcuts import render,redirect
from django.contrib import messages
from mainapp.models import *
from django.core.mail import send_mail
from django.conf import settings 
from django.utils import timezone
import ssl 
import urllib.parse 
import urllib.request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from django.db.models import Count
from adminapp.models import *
from django.core.paginator import Paginator

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense