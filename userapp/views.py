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



def user_dashboard(req):
    return render(req, 'user-dashboard.html')

def userfeedback(req): 
    email = req.session["user_email"]  
    user = UserModel.objects.get(user_email=email) 
    if req.method == "POST": 
        stars = req.POST.get("stars") 
        review = req.POST.get("review") 
        
        if not stars or not review.strip(): 
            messages.warning(req, "Enter all the fields to continue!") 
            return render (req, "user-feedback.html") 
        rating = int(stars)
        sid = SentimentIntensityAnalyzer() 
        score = sid.polarity_scores(review) 
        sentiment = None 
        if score["compound"] > 0 and score["compound"] <= 0.5: 
            sentiment = "positive" 
        elif score["compound"] >= 0.5: 
            sentiment = "very positive" 
        elif score["compound"] < -0.5: 
            sentiment = "negative" 
        elif score["compound"] < 0 and score["compound"] >= -0.5: 
            sentiment = " very negative" 
        else: 
            sentiment = "neutral" 
        Feedback.objects.create( 
            Rating=rating, Review=review, Sentiment=sentiment, Reviewer=email 
        ) 
        messages.success(req, "Feedback recorded") 
        return redirect("userfeedback") 
    return render(req, "user-feedback.html", {"user": user})

def user_profile(req): 
    email = req.session.get("user_email") 
    if not email: 
        messages.error(req, "User not logged in.") 
        return redirect("login") 
 
    user = UserModel.objects.get(user_email=email)
    print(user.user_image)
 
    if req.method == "POST":
        user.user_name = req.POST.get("user_name")  
        user.user_contact = req.POST.get("user_contact") 
        user.user_email = req.POST.get("user_email") 
        user.user_password = req.POST.get("user_password")
        user.user_address = req.POST.get("address")
        user.age = req.POST.get("age")

        if req.FILES.get('userimage'):
            user.user_image = req.FILES['userimage']

        user.save()
        messages.success(req, "Profile updated successfully.") 
        return render(req, "user-profile.html", {"user": user})

    return render(req, "user-profile.html", {"user": user})



# detection

import sklearn 
import pickle 
import pandas as pd 
 
def detection(request): 
    if request.method == 'POST': 
        try: 
            Duration = int(request.POST.get('duration')) 
            Protocol = request.POST.get('protocol')
            Service = int(request.POST.get('service')) 
            Flag = request.POST.get('flag')
            Src_Bytes = int(request.POST.get('src_bytes')) 
            Dst_Bytes = int(request.POST.get('dst_bytes')) 
            Count = int(request.POST.get('count')) 
            Srv_Count = float(request.POST.get('srv_count')) 
            Serror_Rate = float(request.POST.get('serror_rate')) 
            Srv_Serror_Rate = float(request.POST.get('srv_serror_rate'))
            Dst_Host_Same_Srv_Rate = float(request.POST.get('dst_host_same_srv_rate'))
            Dst_Host_Serror_Rate = int(request.POST.get('dst_host_serror_rate')) 
 
        except (ValueError, TypeError): 
            messages.warning(request, "Please enter valid numbers.") 
            return render(request, "detection-page.html") 
 
        # Load the trained fraud detection model 
        file_path = r'IDS Dataset/rfc-cyber.pkl'   
        try: 
            with open(file_path, 'rb') as file: 
                loaded_model = pickle.load(file) 
 
            # Check if loaded model is a valid scikit-learn model 
            if not isinstance(loaded_model, sklearn.base.BaseEstimator): 
                messages.error(request, "Loaded model is not compatible.") 
                return redirect("detection_result") 
 
        except FileNotFoundError: 
            messages.error(request, "Model file not found.") 
            return redirect("detection_result") 
        except Exception as e: 
            messages.error(request, f"Error loading model: {str(e)}") 
            return redirect("detection_result")     
 
        # Prepare input features for prediction as a DataFrame 
        feature_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
        'dst_bytes', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'dst_host_same_srv_rate', 'dst_host_serror_rate']    
        features_df = pd.DataFrame([[ 
            Duration, Protocol, Service, Flag, Src_Bytes, Dst_Bytes, Count, Srv_Count, Serror_Rate, Srv_Serror_Rate, Dst_Host_Same_Srv_Rate, Dst_Host_Serror_Rate
        ]], columns=feature_names)  # Ensure correct feature names 
        # Make prediction 
        try: 
            prediction = loaded_model.predict(features_df) 
            prediction_result = int(prediction[0]) 
            print("RESULT---------" , prediction_result) 
            request.session['prediction_result'] = prediction_result 
        except Exception as e: 
            print("error...")
            print(e)
            messages.error(request, f"Detection error: {str(e)}") 
            return redirect("detection_result") 
 
        return redirect("detection_result") 
    return render(request,"detection-page.html") 
 
def detection_result(req): 
    prediction_result = req.session.get('prediction_result') 
    context = { 
        "prediction_result": prediction_result 
    } 
 
    return render(req,"detection-result.html", context) 