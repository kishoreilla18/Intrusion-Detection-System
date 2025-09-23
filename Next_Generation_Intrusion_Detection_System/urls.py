"""Next_Generation_Intrusion_Detection_System URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from mainapp import views as mainviews
from userapp import views as userviews
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',mainviews.home,name='home'),
    path('home/',mainviews.home,name='home'),
    path('index/',mainviews.index,name='index'),
    path('about/',mainviews.about,name='about'),
    path('contact/',mainviews.contact,name='contact'),
    path('signup/',mainviews.signup,name='signup'),
    path('login/',mainviews.login,name='login'),
    path('admin-login/',mainviews.admin_login,name='admin_login'),
    path('otp/',mainviews.otp,name='otp'),
    path('user-dashboard/',userviews.user_dashboard,name='user_dashboard'),
    path('user-profile/',userviews.user_profile,name='user_profile'),
    path('user-feedback/',userviews.userfeedback,name='userfeedback'),
    path('admin-dashboard/',mainviews.admin_dashboard,name='admin_dashboard'),
    path('pending-users/',mainviews.pendingusers,name='pendingusers'),
    path('update_user_status/<str:username>/<str:status>/',mainviews.update_user_status,name="update_user_status"),
    path('all-users/',mainviews.allusers,name='allusers'),
    path('upload-data/',mainviews.uploaddata,name='uploaddata'),
    path('view-data/',mainviews.viewdata,name='viewdata'),
    path('dataset-results/<int:id>/',mainviews.datasetresults,name="datasetresults"),
    path('delete-dataset/<int:id>/',mainviews.deletedataset,name="deletedataset"),
    path('random/',mainviews.random,name='random'),
    path('ann/',mainviews.ann,name='ann'),
    path('logistic/',mainviews.logistic,name='logistic'),
    path('decision/',mainviews.decision,name='decision'),
    path('adaboost/',mainviews.adaboost,name='adaboost'),
    path('xgboost/',mainviews.xgboost,name='xgboost'),
    path('LightGBM/',mainviews.LightGBM,name='LightGBM'),
    path('algorithm-results/',mainviews.algorithmresults,name='algorithmresults'),
    path('feedback-overview/',mainviews.feedbackoverview,name='feedbackoverview'),
    path('sentiment-analysis/',mainviews.sentimentanalysis,name="sentimentanalysis"),
    path('sentiment-graph/',mainviews.sentimentgraph,name="sentimentgraph"),
    path('data-overview/',mainviews.data_overview,name='data_overview'),
    path('train_split/',mainviews.train_split,name='train_split'),
    path('distribution_labels/',mainviews.distribution_labels,name='distribution_labels'),
    path('protocol_distribution/',mainviews.protocol_distribution,name='protocol_distribution'),
    path('duration_distribution/',mainviews.duration_distribution,name='duration_distribution'),
    path('attack_distribution/',mainviews.attack_distribution,name='attack_distribution'),
    path('corelation_features/',mainviews.corelation_features,name='corelation_features'),
    path('protocol_type/',mainviews.protocol_type,name='protocol_type'),
    path('services/',mainviews.services,name='services'),
    path('detection-page/',mainviews.detection_page,name='detection_page'),
    path('target-column/',mainviews.target_column,name='target_column'),
    path("forgot-password/", mainviews.forgot_password, name="forgot_password"),
    path("reset-password/", mainviews.reset_password, name="reset_password"),
    path('otp-reset/',mainviews.otp_reset,name='otp_reset'),
    path('rf_result/',mainviews.rf_result,name='rf_result'),
    path('logistic_result/',mainviews.logistic_result,name='logistic_result'),
    path('dt_result/',mainviews.dt_result,name='dt_result'),
    path('xgboost_result/',mainviews.xgboost_result,name='xgboost_result'),
    path('adaboost_result/',mainviews.adaboost_result,name='adaboost_result'),
    path('lgbm_result/',mainviews.lgbm_result,name='lgbm_result'),
    path('ann_result/',mainviews.ann_result,name='ann_result'),
    path('comparison_graph/',mainviews.comparison_graph,name='comparison_graph'),
    path('detection/',userviews.detection,name='detection'),
    path('detection-result/',userviews.detection_result,name='detection_result'),
    

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
