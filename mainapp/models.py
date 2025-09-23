from django.db import models

# Create your models here.
class UserModel(models.Model):
    user_id = models.AutoField(primary_key=True) 
    user_name = models.CharField(help_text="user_name", max_length=50)  
    user_email = models.EmailField(help_text="user_email", unique=True) 
    user_password = models.EmailField(help_text="user_password", max_length=50) 
    user_address = models.TextField(help_text="user_address", max_length=100)  
    user_contact = models.CharField(help_text="user_contact", max_length=15, null=True)
    age = models.CharField(help_text="age", max_length=3, null=True) 
    user_image = models.ImageField(upload_to="profile_images/", blank=True, null=True)
    datetime = models.DateTimeField(auto_now=True)  
    last_login = models.DateTimeField(null=True, blank=True) 
    STATUS_CHOISES = [
        ('pending', 'Pending'),
        ('accepted', 'Accepted'),
        ('rejected', 'Rejected'),
    ]
    status = models.CharField(max_length=10, choices=STATUS_CHOISES, default='pending')
    Otp_Num = models.IntegerField(null=True) 
    Otp_Status = models.TextField(default="pending", max_length=60, null=True)
    class Meta: 
        db_table = "user_details" 


class Feedback(models.Model): 
    Feed_id = models.AutoField(primary_key=True) 
    Rating = models.CharField(max_length=100, null=True) 
    Review = models.CharField(max_length=225, null=True) 
    Sentiment = models.CharField(max_length=100, null=True) 
    Reviewer = models.EmailField(help_text="user_email") 
    datetime = models.DateTimeField(auto_now=True) 
 
    class Meta: 
        db_table = "feedback_details" 




class UploadDatasetModel(models.Model):
    s_no = models.AutoField(primary_key = True)   # MongoDB's default primary key
    dataset = models.FileField(upload_to='')  
    file_size = models.PositiveIntegerField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_type = models.CharField(max_length=50, null=True, blank=True)

    class Meta:
        db_table = 'upload_dataset'


from django.db import models
from djongo import models
from django.contrib.auth.models import User


class BaseModelWithAutoIncrement(models.Model):
    S_NO = models.AutoField(primary_key=True)

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        if not self.S_NO:
            last_model = self.__class__.objects.all().order_by('-S_NO').first()
            self.S_NO = (last_model.S_NO + 1) if last_model and last_model.S_NO else 1
        super().save(*args, **kwargs)

        
        
class RFM(BaseModelWithAutoIncrement):
    Accuracy = models.FloatField()
    Precision = models.FloatField()
    F1_Score = models.FloatField()
    Recall = models.FloatField()
    Name = models.CharField(max_length=100)

    class Meta:
        db_table = 'Random_Forest'

class LRM(BaseModelWithAutoIncrement):
    Accuracy = models.FloatField()
    Precision = models.FloatField()
    F1_Score = models.FloatField()
    Recall = models.FloatField()
    Name = models.CharField(max_length=100)

    class Meta:
        db_table = 'Logistic_Regression'

class DT(BaseModelWithAutoIncrement):
    Accuracy = models.FloatField()
    Precision = models.FloatField()
    F1_Score = models.FloatField()
    Recall = models.FloatField()
    Name = models.CharField(max_length=100)

    class Meta:
        db_table = 'Decision_Tree'

class XGBOOST(BaseModelWithAutoIncrement):
    Accuracy = models.FloatField()
    Precision = models.FloatField()
    F1_Score = models.FloatField()
    Recall = models.FloatField()
    Name = models.CharField(max_length=100)

    class Meta:
        db_table = 'XGBoost'

class ADABOOST(BaseModelWithAutoIncrement):
    Accuracy = models.FloatField()
    Precision = models.FloatField()
    F1_Score = models.FloatField()
    Recall = models.FloatField()
    Name = models.CharField(max_length=100)

    class Meta:
        db_table = 'ADABoost'

class LGBM(BaseModelWithAutoIncrement):
    Accuracy = models.FloatField()
    Precision = models.FloatField()
    F1_Score = models.FloatField()
    Recall = models.FloatField()
    Name = models.CharField(max_length=100)

    class Meta:
        db_table = 'LGBM'

class ANN(BaseModelWithAutoIncrement):
    Accuracy = models.FloatField()
    Precision = models.FloatField()
    F1_Score = models.FloatField()
    Recall = models.FloatField()
    Name = models.CharField(max_length=100)

    class Meta:
        db_table = 'ANN'