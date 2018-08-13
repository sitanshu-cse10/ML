from django.shortcuts import render

# Create your views here.


from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
def index(request):
    return render(request,'index.html')
def marks(request):
    #receive hours request:
    hrs=int(request.GET["hours"])
    #stage 1 Collect the dataframe
    import pandas as pd
    
    df=pd.read_csv("D:\MarksPredict\predict\Grade_Set_1.csv")
    #Stage selection
    import numpy as np
    #Independent Variable
    X=df.Hours_Studied[:,np.newaxis]
    #Dependent Variable
    Y=df.Test_Grade.values 
    #linear Regression..
    import sklearn.linear_model as lm
    model=lm.LinearRegression()
    #Fit the model
    model.fit(X,Y)
    #predict the Response Variable
    marks=(model.predict(hrs))
    marks=int(marks[0])
    if(marks<=0):
        marks=0
    elif(marks>=100):
        marks=100
    else:
        marks=marks
    
    #return the response 
    return HttpResponse(marks)
def accuary(request):
    #stage 1 Collect the dataframe
    import pandas as pd
    df=pd.read_csv("D:\MarksPredict\predict\Grade_Set_1.csv")
    #Stage selection
    import numpy as np
    #Independent Variable
    X=df.Hours_Studied[:,np.newaxis]
    #Dependent Variable
    Y=df.Test_Grade.values 
    #generate the accuary..
    import sklearn.linear_model as lm
    model=lm.LinearRegression()
    model.fit(X,Y)
    from sklearn.metrics import r2_score
    #?r2_score
    accuracy=r2_score(Y,model.predict(X))
    format(accuracy, '.4f')
    #errors: 0-n
    from sklearn.metrics import mean_absolute_error
    mae=mean_absolute_error(Y,model.predict(X))
    format(mae, '.4f')
    response="accuracy"
    return HttpResponse(response)

	