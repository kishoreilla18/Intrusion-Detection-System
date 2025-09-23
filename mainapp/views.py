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

# Create your views here.
def index(req):
    return render(req,'index.html')

def home(req):
    return render(req, 'home.html')

def about(req):
    return render(req, 'about.html')

def contact(req):
    return render(req, 'contact.html')

import random as rnd
def signup(req): 

    if req.method == "POST": 
        fullname = req.POST.get("username") 
        email = req.POST.get("useremail") 
        password = req.POST.get("userpassword")  
        address = req.POST.get("address") 
        phone = req.POST.get("phno")
        age = req.POST.get("age")
        userimage = req.FILES.get("userimage", None)

        number = rnd.randint(1000, 9999)

        print(type(userimage))
        if not fullname or not email or not password or not address or not phone or not userimage: 
            messages.warning(req, "Enter all the fields to continue") 
            return render(req, 'signup.html')
        try: 
            data = UserModel.objects.get(user_email=email) 
            messages.warning( 
                req, "Email was already registered, choose another email..!" 
            ) 
            return redirect("signup") 
        except:
            
            UserModel.objects.create( 
                user_name=fullname, 
                user_email=email, 
                user_contact=phone,
                age=age,
                user_address=address,
                user_password=password,
                user_image=userimage, 
                Otp_Num=number,
            ) 
            mail_message = ( 
                f"Registration Successfully\n Your 4 digit Pin is below\n {number}" 
            )  
            
            sendSMS(fullname,number,phone) 
            send_mail("Verify your OTP", mail_message , settings.EMAIL_HOST_USER, [email]) 
            user = UserModel.objects.get(user_email=email)
            req.session["user_email"] = email 
            messages.success(req, "Your account was created..") 
            return redirect("otp")
    print(req.method)  
    return render(req, "signup.html")   

def sendSMS(user, otp, mobile): 
    data = urllib.parse.urlencode( 
        { 
            "username": "Codebook", 
            "apikey": "6876b58478ee6ece5fad", 
            "mobile": mobile, 
            "message": f"Hello {user}, your OTP for account activation is {otp}. This message is generated from https://www.codebook.in server. Thank you", 
            "senderid": "CODEBK", 
        } 
    ) 
    data = data.encode("utf-8") 
    # Disable SSL certificate verification 
    context = ssl._create_unverified_context() 
    request = urllib.request.Request("https://smslogin.co/v3/api.php?") 
    f = urllib.request.urlopen(request, data, context=context) 
    return f.read()

def otp(req):
    user_email = req.session.get("user_email")
    if user_email:
        try:
            user_o = UserModel.objects.get(user_email=user_email)
        except UserModel.DoesNotExist:
            messages.error(req, "User not found.")
            return redirect("login")

        if req.method == "POST":
            otp1 = req.POST.get("otp1", "")
            otp2 = req.POST.get("otp2", "")
            otp3 = req.POST.get("otp3", "")
            otp4 = req.POST.get("otp4", "")

            if otp1 and otp2 and otp3 and otp4:
                user_otp = otp1 + otp2 + otp3 + otp4
                if user_otp.isdigit():
                    u_otp = int(user_otp)
                    if u_otp == user_o.Otp_Num:
                        user_o.Otp_Status = "verified"
                        user_o.save()
                        messages.success(
                            req, "OTP verification was successful. You can now login."
                        )
                        return redirect("login")
                    else:
                        messages.error(
                            req, "Invalid OTP. Please enter the correct OTP."
                        )
                else:
                    messages.error(
                        req, "Invalid OTP format. Please enter numbers only."
                    )
            else:
                messages.error(req, "Please enter all OTP digits.")

    else:
        messages.error(req, "Session expired. Please retry the OTP verification.")

    return render(req, "otp.html")



def login(req):
    if req.method == "POST":
        user_email = req.POST.get("email")
        user_password = req.POST.get("password")

        # Check empty fields
        if not user_email or not user_password:
            messages.warning(req, "Enter all the fields to continue")
            return render(req, "login.html")

        try:
            users_data = UserModel.objects.filter(user_email=user_email)
            if not users_data.exists():
                messages.error(req, "User does not exist")
                return redirect("login")

            for user_data in users_data:
                if user_data.user_password == user_password:
                    # Case 1: Verified OTP + Accepted User
                    if user_data.Otp_Status == "verified" and user_data.status == "accepted":
                        req.session["user_email"] = user_email 
                        messages.success(req, "You are logged in..")
                        user_data.last_login = timezone.now()
                        user_data.save()
                        return redirect("user_dashboard")

                    # Case 2: Verified OTP + Pending User
                    elif user_data.Otp_Status == "verified" and user_data.status == "pending":
                        messages.info(req, "Your Status is pending")
                        return redirect("login")

                    # Case 3: Verified OTP + Removed User
                    elif user_data.Otp_Status == "verified" and user_data.status == "rejected":
                        messages.info(req, "Your Account has been suspended...!")
                        return redirect("login")

                    # Case 4: OTP not verified
                    else:
                        messages.warning(req, "Please verify your OTP first...!")
                        req.session["user_email"] = user_data.user_email
                        return redirect("otp")
                else:
                    messages.error(req, "Incorrect credentials...!")
                    return redirect("login")

            # Fallback if no match found
            messages.error(req, "Incorrect credentials...!")
            return redirect("login")

        except Exception as e:
            print("Login error:", e)
            messages.error(req, "An error occurred. Please try again later.")
            return redirect("login")

    # GET request
    return render(req, "login.html")


# forgot password
def forgot_password(request):
    if request.method == "POST":
        gmail = request.POST.get("email")
        print(gmail)
        try:
            user = UserModel.objects.get(user_email=gmail)
        except UserModel.DoesNotExist:
            messages.error(request, "Mail not found.")
            return render(request, "forgot_password.html")

        # Generate a 4-digit OTP
        otp = str(rnd.randint(1000, 9999))

        # Save OTP in session
        request.session["reset_gmail"] = user.user_email
        request.session["otp"] = otp

        # Send OTP to Gmail
        sendSMS(user.user_name, otp, user.user_contact)
        send_mail(
            subject="Password Reset OTP",
            message=f"Hello {user.user_name},\n\nYour OTP is {otp}.",
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.user_email],
            fail_silently=False,
        )

        messages.success(request, "OTP sent to your Email and Mobile.")
        return redirect("otp_reset")

    return render(request, "forgot_password.html")

#Verify otp
def otp_reset(request):
    email = request.session.get("reset_gmail")
    if email:
        try: 
            user_o = UserModel.objects.get(user_email=email)
        except UserModel.DoesNotExist:
            messages.error(request, "User not found.") 
            return redirect("login") 
 
        if request.method == "POST":
            otp1 = request.POST.get("otp1", "") 
            otp2 = request.POST.get("otp2", "") 
            otp3 = request.POST.get("otp3", "") 
            otp4 = request.POST.get("otp4", "") 
            #print(otp1, otp2, otp3, otp4)
            if otp1 and otp2 and otp3 and otp4: 
                user_otp = otp1 + otp2 + otp3 + otp4 
                if user_otp.isdigit(): 
                    u_otp = int(user_otp) 
                    #print( u_otp, request.session.get("otp"), type(u_otp), type(request.session.get("otp")))
                    if u_otp == int(request.session.get("otp")): 
                        print("otp matched")
                        messages.success(request , "OTP verification was successful.")
                        return redirect("reset_password") 
                    else: 
                        messages.error( 
                            request, "Invalid OTP. Please enter the correct OTP." 
                        ) 
                else: 
                    messages.error( 
                        request, "Invalid OTP format. Please enter numbers only." 
                    ) 
            else: 
                messages.error(request, "Please enter all OTP digits.") 
 
    else: 
        messages.error(request, "Session expired. Please retry the OTP verification.")
    return render(request, "otp_reset.html")

def reset_password(req):
    user_email = req.session.get("user_email")
    print(user_email)

    if user_email:
        try:
            user = UserModel.objects.get(user_email=user_email)
            if req.method == "POST":
                new_password = req.POST.get("new_password")
                confirm_password = req.POST.get("confirm_password")

                if new_password != confirm_password:
                    messages.error(req, "Passwords do not match.")
                    return redirect(req.path)

                # ⚠️ For production, hash password (don’t store plain text!)
                user.user_password = new_password
                user.save()

                mail_message = f"Hi {user.user_name},\n\nYour password is reset successfully. Now you can login with your new password.\n\nIf it's not you please report."
                send_mail("Reset Your Password", mail_message, settings.EMAIL_HOST_USER, [user_email])
                messages.success(req, "Your password has been reset successfully. You can now login.")
                return redirect("login")

            return render(req, "reset_password.html")
        except UserModel.DoesNotExist:
            messages.error(req, "User not found. Please login again.")
            return redirect("login")
    else:
        messages.error(req, "Session expired. Please login again.")
        return redirect("login")


                                 #admin Views
#admin login
def admin_login(req): 
    admin_name = "admin" 
    admin_pwd = "admin" 
    if req.method == "POST": 
        admin_n = req.POST.get("username") 
        admin_p = req.POST.get("password") 
        if not admin_n or not admin_p: 
            messages.warning(req, "Enter all the fields to continue") 
            return render(req, 'admin-login.html') 
        if admin_n == admin_name and admin_p == admin_pwd: 
            messages.success(req, "You are logged in..") 
            return redirect("admin_dashboard") 
        else: 
            messages.error(req, "You are trying to login with wrong details..") 
            return redirect("admin_login") 
    return render(req,"admin-login.html")


def admin_dashboard(request):
    users = UserModel.objects.all().order_by('-user_id')[:5]
    rejected_count = UserModel.objects.filter(status="rejected").count()
    accepted_count = UserModel.objects.filter(status="accepted").count()
    pending_count = UserModel.objects.filter(status="pending").count()
    feedback_count = Feedback.objects.all().count()

    return render(request,'admin-dashboard.html', {"users":users, "rejected_count": rejected_count, "accepted_count": accepted_count, "pending_count": pending_count, "feedback_count": feedback_count})


def pendingusers(req):
    users = UserModel.objects.filter(status="pending")
    return render(req,"pendingusers.html", {"users":users}, )

def update_user_status(req,username,status):
    if status=="accept":
        UserModel.objects.filter(user_name=username).update(status="accepted")
    elif status=="reject":
        UserModel.objects.filter(user_name=username).update(status="rejected")
    return redirect("pendingusers")

def allusers(req):
    users = UserModel.objects.all().order_by("datetime")   # latest first

    # Pagination (5 per page)
    paginator = Paginator(users, 5)  
    page_number = req.GET.get("page")
    page_obj = paginator.get_page(page_number)

    return render(req, "allusers.html", {"page_obj": page_obj})

def uploaddata(request):
    if request.method == 'POST':
        file = request.FILES['file']
        file_size = file.size  # File size is already in bytes
        file_type = file.name.split('.')[-1].upper() # auto detect type

        # Ensure user is included (assuming user is logged in)
        UploadDatasetModel.objects.create(
            file_size=file_size,  # Store file size in bytes
            dataset=file,
            file_type=file_type,
        )

        messages.success(request, 'Your dataset was uploaded successfully.')
    
    return render(request, "uploaddataset.html")


import pandas as pd

def viewdata(req):
    dataset = UploadDatasetModel.objects.all()
    print(dataset)
    return render(req, "viewdataset.html",{"dataset":dataset})

def datasetresults(request, id):
    data = UploadDatasetModel.objects.get(s_no=id)  # Get the latest uploaded dataset
    if not data:
        messages.error(request, "No dataset found.")
        return redirect('viewdata')

    file_path = data.dataset.path  # Correctly get the file path
    
    try:
        df = pd.read_csv(file_path, nrows=50)  # Load only the first 50 rows
        table = df.to_html(classes="table table-bordered", table_id="data_table")  # Add styling for better UI
    except Exception as e:
        messages.error(request, f"Error reading file: {e}")
        return redirect('viewdata')

    return render(request, "datasetresults.html", {'t': table})

def deletedataset(req, id):
    UploadDatasetModel.objects.get(s_no=id).delete()
    messages.success(req, 'dataset was deleted successfully.')
    return redirect('viewdata')



def ann(req):
    return render(req,"ann.html")

def logistic(req):
    return render(req,"logistic.html")

def decision(req):
    return render(req,"decision.html")

def xgboost(req):
    return render(req,"xgboost.html")

def lightGBM(req):
    return render(req,"lightGBM.html")

def adaboost(req):
    return render(req,"adaboost.html")

def algorithmresults(req):
    return render(req, "algorithmresults.html")

from django.core.paginator import Paginator

def feedbackoverview(req):
    reviews = Feedback.objects.all().order_by("-datetime")   # latest first

    # Pagination (5 per page)
    paginator = Paginator(reviews, 5)  
    page_number = req.GET.get("page")
    page_obj = paginator.get_page(page_number)

    return render(req, "feedbackoverview.html", {"page_obj": page_obj})


def sentimentanalysis(req):
    reviews = Feedback.objects.all().order_by("-datetime")   # latest first
    
    # Pagination (5 per page)
    paginator = Paginator(reviews, 5)  
    page_number = req.GET.get("page")
    page_obj = paginator.get_page(page_number)
    return render(req, "sentimentanalysis.html", {"page_obj": page_obj})

def sentimentgraph(req):
    # Count how many feedbacks per sentiment
    sentiment_data = Feedback.objects.values("Sentiment").annotate(count=Count("Sentiment"))
    print(sentiment_data)
    # Prepare dictionary for all sentiments
    sentiment_counts = {
    "very_positive": Feedback.objects.filter(Sentiment="very positive").count(),
    "positive": Feedback.objects.filter(Sentiment="positive").count(),
    "neutral": Feedback.objects.filter(Sentiment="neutral").count(),
    "negative": Feedback.objects.filter(Sentiment="negative").count(),
    "very_negative": Feedback.objects.filter(Sentiment="very negative").count(),
}

    return render(req, "sentimentgraph.html", {"sentiment_counts": sentiment_counts})


def data_overview(req):
    df = pd.read_csv(r'IDS Dataset/cleaned_data.csv') 

    # Prepare each card
    cards = [
        {
            "title": "Dataset Columns Overview",
            "code": "df.columns",
            "output": f"{df.columns.tolist()}",
            "description": "This command lists all the column names present in the dataset. These columns represent different agricultural and environmental attributes such as crop ID, soil type, seedling stage, soil moisture index (MOI), temperature, humidity, the result (crop condition/irrigation need), and the recommended irrigation level."
        },
        {
            "title": "Dataset Shape Overview",
            "code": "df.shape",
            "output": f"Shape: {df.shape}",
            "description": "The shape function shows that the dataset contains 16,283 rows and 8 columns. Each row corresponds to a crop/environmental record, and each column represents a feature related to soil, crop stage, environmental conditions, or irrigation requirements."
        },

        {
            "title": "Descriptive Stats",
            "code": "df.describe()",
            "output": str(df.describe()),
            "description": "Generates a statistical summary of the dataset, including count, mean, standard deviation, minimum, quartiles, and maximum for each numeric column. This helps in understanding the distribution of agricultural and environmental variables such as crop ID, soil moisture index (MOI), temperature, humidity, and irrigation levels."
        },
        {
            "title": "Unique Values in Each Column",
            "code": "df.nunique()",
            "output": str(df.nunique()),
            "description": "This step reveals the number of unique entries in each column. It is useful to distinguish categorical features (like crop ID, soil type, seedling stage, and result) from continuous environmental features (like moisture index, temperature, and humidity). It also shows the range of values for irrigation levels, which helps in analyzing irrigation requirements."
        },
        {
            "title": "Null Value Check",
            "code": "df.isnull().sum()",
            "output": str(df.isnull().sum()),
            "description": "Counts the number of missing (NaN) values in each column of the dataset. In this case, no missing values were detected. However, zero values in environmental features (such as MOI, temperature, or humidity) may still indicate unrecorded or faulty sensor readings."
        },
        {
            "title": "Check for Duplicates",
            "code": "df.duplicated().sum()",
            "output": str(df.duplicated().sum()),
            "description": "Returns the total number of duplicate rows in the dataset. In this case, no duplicate records were found, ensuring that each entry corresponds to a unique crop or environmental record."
        }
    ]

    return render(req, 'dataoverview.html', {"cards": cards})



def train_split(req):
    df = pd.read_csv(r'IDS Dataset/cleaned_data.csv')  # Replace with your actual path

    # Prepare each card
    cards = [
        {
            "title": "Splitting of Data",
            "code": """from sklearn.model_selection import train_test_split

    # Define feature set and target label
    X = df.drop(columns=['result'])
    y = df['result']


    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,            # Features
        y,            # Target labels
        test_size=0.3, # Proportion of the dataset to include in the test split
        random_state=11 # Seed for the random number generator
    )

    # Optionally, print the shapes of the resulting splits
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)""",
            "output": "X_train shape:  (11398, 7)\ny_train shape:  (11398,)\nX_test shape:  (4885, 7)\ny_test shape:  (4885,)",
            "description": "Splits the irrigation dataset into training and testing sets using scikit-learn's train_test_split. The target variable result indicates the irrigation requirement or crop condition, while the remaining 7 columns represent agricultural and environmental features such as soil type, seedling stage, MOI, temperature, and humidity. 70% of the data (11,398 samples) is allocated for training, and 30% (4,885 samples) is for testing. A fixed random seed of 11 ensures reproducibility of results."
        }
    ]
    return render(req,'train-test-split.html', {"cards": cards})

def distribution_labels(req):
    return render(req, 'distribution_labels.html')

def protocol_distribution(req):
    return render(req, 'protocol_distribution.html')

def duration_distribution(req):
    return render(req, 'duration_distribution.html')

def attack_distribution(req):
    return render(req, 'attack_distribution.html')

def corelation_features(req):
    return render(req, 'corelation_features.html')

def protocol_type(req):
    return render(req, 'protocol_type.html')

def services(req):
    return render(req, 'services.html')

def detection_page(req):
    return render(req, 'detection-page.html')

def target_column(req):
    return render(req, 'target_column.html')



    
# algorithms
def load_and_prepare_data():
    try:
        df = pd.read_csv(r'IDS Dataset/cleaned_data.csv')
    except Exception as e:
        return None, None, None, None, None, None, None, f"Error loading data: {e}"

    if 'attack_category' not in df.columns:
        return None, None, None, None, None, None, None, "'attack_category' column missing"

    # One-hot encode features
    X = pd.get_dummies(df.drop(columns=['attack_category']), drop_first=True)

    # Convert target column to numeric (0/1)
    y = df['attack_category']
    if y.dtype == 'object':
        y = y.map({'No': 0, 'Yes': 1})  # adjust based on actual labels

    # Ensure correct dtypes for Keras (float32 for X, int for y)
    X = X.astype('float32')
    y = y.astype('int')

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=11
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train.values, y_test.values, X, y, list(X.columns), None



def random(request):
    context={
        "title":"Random Forest Algorithm",
        "description":"Random Forest is a machine learning algorithm that builds multiple decision trees and combines their results to improve accuracy. It uses random subsets of data and features for each tree, reducing overfitting and increasing robustness. It’s commonly used for both classification and regression tasks.",
        "link":"rf_result"
    }
    return render(request, 'algorithm-details.html',context)
    # return render(request, 'algorithm-details.html',context)



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def rf_result(request):
    # Step 0: Check if results already exist
    latest_result = RFM.objects.last()
    if latest_result:
        messages.success(request, "Fetched existing Random Forest results.")
        return render(request, 'algorithmresults.html', {
            'name': latest_result.Name,
            'accuracy': latest_result.Accuracy,
            'precision': latest_result.Precision,
            'recall': latest_result.Recall,
            'f1': latest_result.F1_Score,
            'report': "Previously computed. Report not stored."
        })

    # Step 1: Load data
    X_train, X_test, y_train, y_test, X, y, features, error = load_and_prepare_data()
    if error:
        messages.error(request, error)
        return redirect('uploaddata')

    # Step 2: Initialize and fit Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Step 3: Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Step 4: Accuracy
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    # Step 5: Evaluation Metrics
    precision = precision_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    report_text = classification_report(y_test, test_pred)

    # Step 6: Cross-validation
    try:
        cv_scores = cross_val_score(model, X, y, cv=5)
        cv_mean = cv_scores.mean()
    except Exception as e:
        print("Cross-validation failed:", e)
        cv_mean = 0.0
    # Step 7: Save to DB
    RFM.objects.create(
        Name="Random Forest Classifier",
        Accuracy=f"{train_accuracy * 100:.2f}",
        Precision=f"{precision:.2f}",
        Recall=f"{recall:.2f}",
        F1_Score=f"{f1:.2f}"
    )

    # Step 8: Render result
    context = {
        'name': "Random Forest Classifier",
        'accuracy': f"{train_accuracy * 100:.2f}",
        'precision': f"{precision:.2f}",
        'recall': f"{recall:.2f}",
        'f1': f"{f1:.2f}",
        'report': report_text.replace('\n', '<br>')
    }

    messages.success(request, 'Random Forest executed successfully.')
    return render(request, 'algorithmresults.html', context)

# logistic Regresion

def logistic(request):
    context = {
        "title": "Logistic Regression Algorithm",
        "description": """Logistic Regression is a statistical model used for binary classification problems. 
        It predicts the probability of an outcome (yes/no, 0/1) using a logistic (sigmoid) function. 
        Despite its name, it is mainly used for classification tasks rather than regression.""",
        "link": "logistic_result"
    }
    print(context)
    return render(request, 'algorithm-details.html', context)


def logistic_result(request):
    # Step 0: Check if results already exist
    print('hello')
    latest_result = LRM.objects.last()
    if latest_result:
        print('hii')
        messages.success(request, "Fetched existing Random Forest results.")
        return render(request, 'algorithmresults.html', {
            'name': latest_result.Name,
            'accuracy': latest_result.Accuracy,
            'precision': latest_result.Precision,
            'recall': latest_result.Recall,
            'f1': latest_result.F1_Score,
            'report': "Previously computed. Report not stored."
        })

    # Step 1: Load data
    X_train, X_test, y_train, y_test, X, y, features, error = load_and_prepare_data()
    if error:
        messages.error(request, error)
        return redirect('uploaddata')

    # Step 2: Initialize and fit Random Forest model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Step 3: Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Step 4: Accuracy
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    # Step 5: Evaluation Metrics
    precision = precision_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    report_text = classification_report(y_test, test_pred)

    # Step 6: Cross-validation
    try:
        cv_scores = cross_val_score(model, X, y, cv=5)
        cv_mean = cv_scores.mean()
    except Exception as e:
        print("Cross-validation failed:", e)
        cv_mean = 0.0
    # Step 7: Save to DB
    LRM.objects.create(
        Name="Logistic Regression",
        Accuracy=f"{train_accuracy * 100:.2f}",
        Precision=f"{precision:.2f}",
        Recall=f"{recall:.2f}",
        F1_Score=f"{f1:.2f}"
    )

    # Step 8: Render result
    context = {
        'name': "Logistic Regression",
        'accuracy': f"{train_accuracy * 100:.2f}",
        'precision': f"{precision:.2f}",
        'recall': f"{recall:.2f}",
        'f1': f"{f1:.2f}",
        'report': report_text.replace('\n', '<br>')
    }

    messages.success(request, 'Logistic Regression executed successfully.')
    return render(request, 'algorithmresults.html', context)

#Decision_Tree
def decision(req):
    card={
        "heading":"Decision Tree Classifier",
        "title":"Decision Tree Algorithm",
        "description":"Decision Tree is a supervised machine learning algorithm that splits the dataset into smaller subsets using rules based on feature values. Each internal node represents a decision based on a feature, each branch corresponds to the outcome of that decision, and each leaf node represents the final prediction. It is simple, easy to interpret, and can handle both classification and regression tasks. However, decision trees can easily overfit the data if not properly pruned or regularized.",
        "link":"dt_result",
    }
    return render(req,'algorithm-details.html',card)

def dt_result(request):
    # Step 0: Check if results already exist
    latest_result = DT.objects.last()
    if latest_result:
        messages.success(request, "Fetched existing Decision Tree results.")
        return render(request, 'algorithmresults.html', {
            'name': latest_result.Name,
            'accuracy': latest_result.Accuracy,
            'precision': latest_result.Precision,
            'recall': latest_result.Recall,
            'f1': latest_result.F1_Score,
            'report': "Previously computed. Report not stored."
        })

    # Step 1: Load data
    X_train, X_test, y_train, y_test, X, y, features, error = load_and_prepare_data()
    if error:
        messages.error(request, error)
        return redirect('admin_upload')

    # Step 2: Initialize and fit Random Forest model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Step 3: Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Step 4: Accuracy
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    # Step 5: Evaluation Metrics
    precision = precision_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    report_text = classification_report(y_test, test_pred)

    # Step 6: Cross-validation
    try:
        cv_scores = cross_val_score(model, X, y, cv=5)
        cv_mean = cv_scores.mean()
    except Exception as e:
        print("Cross-validation failed:", e)
        cv_mean = 0.0

    # Step 7: Save to DB
    DT.objects.create(
        Name="Decision Tree Classifier",
        Accuracy=f"{train_accuracy * 100:.2f}",
        Precision=f"{precision:.2f}",
        Recall=f"{recall:.2f}",
        F1_Score=f"{f1:.2f}"
    )

    # Step 8: Render result
    context = {
        'name': "Decision Tree Classifier",
        'accuracy': f"{train_accuracy * 100:.2f}",
        'precision': f"{precision:.2f}",
        'recall': f"{recall:.2f}",
        'f1': f"{f1:.2f}",
        'report': report_text.replace('\n', '<br>')
    }

    messages.success(request, 'Decision Tree executed successfully.')
    return render(request, 'algorithmresults.html', context)

def xgboost(req):
    card={
        "heading":"📊 eXtreme Gradient Boosting",
        "title":"XGBOOST Algorithm",
        "description":"XGBoost is an advanced implementation of the gradient boosting framework designed for efficiency and performance. It builds an ensemble of weak learners (typically decision trees) in a sequential mmodeler, where each new model corrects the errors of the previous ones. XGBoost includes regularization techniques to reduce overfitting, supports parallel processing, and provides high scalability. It is widely used in machine learning competitions and real-world applications for classification, regression, and ranking problems due to its accuracy and speed.",
        "link":"xgboost_result",
    }
    return render(req,'algorithm-details.html',card)

def xgboost_result(request):
    # Step 0: Check if results already exist
    latest_result = XGBOOST.objects.last()
    if latest_result:
        messages.success(request, "Fetched existing Random Forest results.")
        print(latest_result.Name,latest_result.Accuracy,latest_result.Precision,latest_result.Recall,latest_result.F1_Score)
        return render(request, 'algorithmresults.html', {
            'name': latest_result.Name,
            'accuracy': latest_result.Accuracy,
            'precision': latest_result.Precision,
            'recall': latest_result.Recall,
            'f1': latest_result.F1_Score,
            'report': "Previously computed. Report not stored."
        })

    # Step 1: Load data
    X_train, X_test, y_train, y_test, X, y, features, error = load_and_prepare_data()
    if error:
        messages.error(request, error)
        return redirect('uploaddata')

    # Step 2: Initialize and fit Random Forest model
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Step 3: Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Step 4: Accuracy
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    # Step 5: Evaluation Metrics
    precision = precision_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    report_text = classification_report(y_test, test_pred)

    # Step 6: Cross-validation
    try:
        cv_scores = cross_val_score(model, X, y, cv=5)
        cv_mean = cv_scores.mean()
    except Exception as e:
        print("Cross-validation failed:", e)
        cv_mean = 0.0

    # Step 7: Save to DB
    XGBOOST.objects.create(
        Name="eXtreme Gradient Boosting",
        Accuracy=f"{train_accuracy * 100:.2f}",
        Precision=f"{precision:.2f}",
        Recall=f"{recall:.2f}",
        F1_Score=f"{f1:.2f}"
    )

    # Step 8: Render result
    context = {
        'name': "eXtreme Gradient Boosting",
        'accuracy': f"{train_accuracy * 100:.2f}",
        'precision': f"{precision:.2f}",
        'recall': f"{recall:.2f}",
        'f1': f"{f1:.2f}",
        'report': report_text.replace('\n', '<br>')
    }

    messages.success(request, 'eXtreme Gradient Boosting executed successfully.')
    return render(request, 'algorithmresults.html', context)

def adaboost(req):
    card={
        "heading":"📊 Adaptive Boost Classifier",
        "title":"Adaptive Boost Classifier Algorithm",
        "description":"AdaBoost is an ensemble learning algorithm that combines multiple weak learners, usually shallow decision trees, to form a strong classifier. It works by assigning weights to data points and iteratively adjusting them so that misclassified samples get higher importance in the next iteration. The final prediction is a weighted vote of all the weak learners. AdaBoost is effective at improving accuracy but can be sensitive to noisy data and outliers.",
        "link":"adaboost_result",
    }
    return render(req,'algorithm-details.html',card)

def adaboost_result(request):
    # Step 0: Check if results already exist
    latest_result = ADABOOST.objects.last()
    if latest_result:
        messages.success(request, "Fetched existing Ada Boost Classifier results.")
        return render(request, 'algorithmresults.html', {
            'name': latest_result.Name,
            'accuracy': latest_result.Accuracy,
            'precision': latest_result.Precision,
            'recall': latest_result.Recall,
            'f1': latest_result.F1_Score,
            'report': "Previously computed. Report not stored."
        })

    # Step 1: Load data
    X_train, X_test, y_train, y_test, X, y, features, error = load_and_prepare_data()
    if error:
        messages.error(request, error)
        return redirect('uploaddata')

    # Step 2: Initialize and fit Random Forest model
    model = AdaBoostClassifier(n_estimators=50,random_state=11)
    model.fit(X_train, y_train)

    # Step 3: Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Step 4: Accuracy
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    # Step 5: Evaluation Metrics
    precision = precision_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    report_text = classification_report(y_test, test_pred)

    # Step 6: Cross-validation
    try:
        cv_scores = cross_val_score(model, X, y, cv=5)
        cv_mean = cv_scores.mean()
    except Exception as e:
        print("Cross-validation failed:", e)
        cv_mean = 0.0

    # Step 7: Save to DB
    ADABOOST.objects.create(
        Name="Ada Boost Classifier Classifier",
        Accuracy=f"{train_accuracy * 100:.2f}",
        Precision=f"{precision:.2f}",
        Recall=f"{recall:.2f}",
        F1_Score=f"{f1:.2f}"
    )

    # Step 8: Render result
    context = {
        'name': "Ada Boost Classifier",
        'accuracy': f"{train_accuracy * 100:.2f}",
        'precision': f"{precision:.2f}",
        'recall': f"{recall:.2f}",
        'f1': f"{f1:.2f}",
        'report': report_text.replace('\n', '<br>')
    }

    messages.success(request, 'Ada Boost Classifier executed successfully.')
    return render(request, 'algorithmresults.html', context)

def LightGBM(req):
    card={
        "heading":"📊 Light Gradient Boosting Machine",
        "title":"Light Gradient Boosting Machine Algorithm",
        "description":"LightGBM is a fast, efficient, and high-performance gradient boosting framework developed by Microsoft. It uses a leaf-wise growth strategy instead of the traditional level-wise approach, which allows it to reduce loss more quickly and handle large datasets with high-dimensional features. LightGBM supports parallel and GPU learning, making it highly scalable for big data tasks. It is widely used for both classification and regression problems and is preferred when speed and memory efficiency are crucial.",
        "link":"lgbm_result",
    }
    return render(req,'algorithm-details.html',card)


def lgbm_result(request):
    # Step 0: Check if results already exist
    latest_result = LGBM.objects.last()
    if latest_result:
        messages.success(request, "Fetched existing Light Gradient Boosting Machine results.")
        return render(request, 'algorithmresults.html', {
            'name': latest_result.Name,
            'accuracy': latest_result.Accuracy,
            'precision': latest_result.Precision,
            'recall': latest_result.Recall,
            'f1': latest_result.F1_Score,
            'report': "Previously computed. Report not stored."
        })

    # Step 1: Load data
    X_train, X_test, y_train, y_test, X, y, features, error = load_and_prepare_data()
    if error:
        messages.error(request, error)
        return redirect('uploaddata')

    # Step 2: Initialize and fit Random Forest model
    model = lgb.LGBMClassifier(
    n_estimators=100,        # Number of boosting rounds
    random_state=11          # Seed for reproducibility
)
    model.fit(X_train, y_train)

    # Step 3: Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Step 4: Accuracy
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    # Step 5: Evaluation Metrics
    precision = precision_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    report_text = classification_report(y_test, test_pred)

    # Step 6: Cross-validation
    try:
        cv_scores = cross_val_score(model, X, y, cv=5)
        cv_mean = cv_scores.mean()
    except Exception as e:
        print("Cross-validation failed:", e)
        cv_mean = 0.0

    # Step 7: Save to DB
    LGBM.objects.create(
        Name="Light Gradient Boosting Machine",
        Accuracy=f"{train_accuracy * 100:.2f}",
        Precision=f"{precision:.2f}",
        Recall=f"{recall:.2f}",
        F1_Score=f"{f1:.2f}"
    )

    # Step 8: Render result
    context = {
        'name': "Light Gradient Boosting Machine",
        'accuracy': f"{train_accuracy * 100:.2f}",
        'precision': f"{precision:.2f}",
        'recall': f"{recall:.2f}",
        'f1': f"{f1:.2f}",
        'report': report_text.replace('\n', '<br>')
    }

    messages.success(request, 'Light Gradient Boosting Machine executed successfully.')
    return render(request, 'algorithmresults.html', context)

def ann(req):
    card={
    "heading":"📊 Artificial Neural Network",
    "title":"Artificial Neural Network Algorithm",
    "description":"An Artificial Neural Network (ANN) is a computational model inspired by the structure and functioning of the human brain. It consists of interconnected layers of nodes (neurons) that process data. The input layer receives raw data, hidden layers transform it through weighted connections and activation functions, and the output layer produces the final prediction or classification. ANNs learn by adjusting weights through backpropagation and optimization techniques, making them powerful for handling complex, non-linear relationships. They are widely used in image recognition, natural language processing, healthcare, finance, and many other domains due to their ability to learn patterns and make accurate predictions.",
    "link":"ann_result",
    }

    return render(req,'algorithm-details.html',card)

#ANN
def ann_result(request):
    # Step 0: Check if results already exist
    latest_result = ANN.objects.last()
    if latest_result:
        messages.success(request, "Fetched existing Artificial Neural Network results.")
        return render(request, 'algorithmresults.html', {
            'name': latest_result.Name,
            'accuracy': latest_result.Accuracy,
            'precision': latest_result.Precision,
            'recall': latest_result.Recall,
            'f1': latest_result.F1_Score,
            'report': "Previously computed. Report not stored."
        })

    # Step 1: Load data
    X_train, X_test, y_train, y_test, X, y, features, error = load_and_prepare_data()
    if error:
        messages.error(request, error)
        return redirect('uploaddata')

    # Step 2: Initialize and fit Random Forest model
    model = Sequential()
    model.add(Dense(units=64, input_dim=X_train.shape[1], kernel_initializer='uniform', activation='relu'))  # input_dim should be the number of features
    model.add(Dense(units=32, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, y_train, epochs=15, batch_size=32)

    # Step 3: Predictions
    train_pred = (model.predict(X_train) > 0.5).astype("int32")
    test_pred = (model.predict(X_test) > 0.5).astype("int32")

    # Step 4: Accuracy
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    # Step 5: Evaluation Metrics
    precision = precision_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0) * 100
    report = classification_report(y_test, test_pred)


    # Step 6: Cross-validation
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)
        cv_mean = cv_scores.mean()
    except Exception as e:
        print("Cross-validation failed:", e)
        cv_mean = 0.0

    # Step 7: Save to DB
        ANN.objects.create(
        Name="Artificial Neural Network",
        Accuracy = f"{train_accuracy * 100:.2f}",
        Precision=f"{precision:.2f}",
        Recall=f"{recall:.2f}",
        F1_Score=f"{f1:.2f}"
    )

    # Step 8: Render result
    context = {
        'name': "Artificial Neural Network",
        'accuracy': f"{train_accuracy * 100:.2f}",
        'precision': f"{precision:.2f}",
        'recall': f"{recall:.2f}",
        'f1': f"{f1:.2f}"
    }

    messages.success(request, 'Artificial Neural Network executed successfully.')
    return render(request, 'algorithmresults.html', context)

def comparison_graph(req):
    rfm = RFM.objects.first()
    lrm = LRM.objects.first()
    dt = DT.objects.first()
    xgb = XGBOOST.objects.first()
    adab = ADABOOST.objects.first()
    lgb = LGBM.objects.first()
    an = ANN.objects.first()

    labels = ["ANN", "Logistic Regression", "AdaBoost", "LGBM", "Decision Tree", "Random Forest", "XGBoost"]
    values = [
        an.Accuracy if an else 0,
        lrm.Accuracy if lrm else 0,
        adab.Accuracy if adab else 0,
        lgb.Accuracy if lgb else 0,
        dt.Accuracy if dt else 0,
        rfm.Accuracy if rfm else 0,
        xgb.Accuracy if xgb else 0
    ]

    context = {
        "labels": labels,
        "values": values
    }
    return render(req, "comparison_graph.html", context)


