from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np
import pandas as pd
import json
import ast
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
import pickle
from sklearn.metrics import mean_squared_error
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',filename=os.path.realpath("gs.log"), level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S');

 # Flask utils
from flask import Flask, request, render_template,send_from_directory
from werkzeug.utils import secure_filename



# Define a flask app
app = Flask(__name__)

from flask_cors import CORS
CORS(app)


# Model saved with Keras model.save()
MODEL_PATH = 'xreg_model_v3.pkl'
ENCODER_PATH='lbl_encoders_v2.pkl'

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')



def convert_to_date(x):
        a= x
        try:
            x = str(x)
            x = datetime.strptime(x, "%Y%m%d")
        except:
            pass


        return x

def getAMPM(hour):
    if hour>12:
        return "PM"
    else:
        return "AM"

def json_string_array_fix(data):
    data = ast.literal_eval(data)
    if(len(data)>0):
        return str(data[0])
    else:
        return ""

def Preprocess(filename,isPredict=True):

    logging.info('Preprocessing started')

    jsoncolumns = ["device", "geoNetwork", "totals", "trafficSource"]

    df = pd.read_csv(filename,
                     converters={column: json.loads for column in jsoncolumns},
                     dtype={"fullVisitorId": "str"})

    logging.info('5% Completed')

    for column in jsoncolumns:
        column_as_df = pd.json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]

    df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)


    logging.info('15% Completed')

    df["customDimensions"] = df["customDimensions"].str.replace("'",'"')
    df["hits"] = df["hits"].str.replace("'",'"')
    df['customDimensions'] = df.apply(lambda x: json_string_array_fix(x['customDimensions']),axis=1)
    df = json.loads(df.to_json(orient="records"))
    df = pd.json_normalize(df)
    df.drop("trafficSource.adContent", axis=1, inplace=True)
    df.drop("trafficSource.adwordsClickInfo.page", axis=1, inplace=True)
    df.drop("trafficSource.adwordsClickInfo.slot", axis=1, inplace=True)

    logging.info('25% Completed')

    df.drop("trafficSource.adwordsClickInfo.gclId", axis=1, inplace=True)
    df.drop("trafficSource.adwordsClickInfo.adNetworkType", axis=1, inplace=True)
    df.drop("customDimensions", axis=1, inplace=True)
    df['visitstarthour'] = (df['visitStartTime'].apply(lambda x: str(datetime.fromtimestamp(x).hour))).astype(int)
    df['visitstartAMPM'] = df.apply(lambda x: getAMPM(x['visitstarthour']),axis=1)
   # df.drop('date', axis=1, inplace=True)

    df.drop('trafficSource.referralPath', axis=1, inplace=True)
    df.drop('trafficSource.keyword', axis=1, inplace=True)


    logging.info('50% Completed')

    if "totals.hits" in df.columns:
        df["totals.hits"] = pd.to_numeric(df["totals.hits"])
    if "totals.pageviews" in df.columns:
        df["totals.pageviews"] = pd.to_numeric(df["totals.pageviews"])
    if "totals.sessionQualityDim" in df.columns:
        df["totals.sessionQualityDim"] = pd.to_numeric(df["totals.sessionQualityDim"])
    if "totals.timeOnSite" in df.columns:
        df["totals.timeOnSite"] = pd.to_numeric(df["totals.timeOnSite"])
    if "totals.transactions" in df.columns:
        df["totals.transactions"] = pd.to_numeric(df["totals.transactions"])
    if "totals.transactionRevenue" in df.columns:
        df["totals.transactionRevenue"] = pd.to_numeric(df["totals.transactionRevenue"])
    if "totals.transactions" in df.columns:
        df["totals.transactions"].fillna(0.0, inplace=True)
    if "totals.transactionRevenue" in df.columns:
        df["totals.transactionRevenue"].fillna(0.0, inplace=True)
    if "totals.totalTransactionRevenue" in df.columns:
        df["totals.totalTransactionRevenue"].fillna(0.0, inplace=True)

    if "visitId" in df.columns:
        df.drop('visitId', axis=1, inplace=True)
    if(isPredict):
        if "totals.transactionRevenue" in df.columns:
            df.drop('totals.transactionRevenue', axis=1, inplace=True)
    else:
        df['totals.transactionRevenue'] = np.log1p(df['totals.transactionRevenue'])


    df.drop('fullVisitorId', axis=1, inplace=True)


    logging.info('75% Completed')

    df["totals.pageviews"].fillna(1, inplace=True)
    df["totals.timeOnSite"].fillna(1, inplace=True)
    df.drop('hits', axis=1, inplace=True)
    df["totals.visits"] = pd.to_numeric(df["totals.visits"])
    df["totals.bounces"] = pd.to_numeric(df["totals.bounces"])
    df["totals.transactions"].fillna(0, inplace=True)
    df["totals.newVisits"] = pd.to_numeric(df["totals.newVisits"])
    df["totals.newVisits"].fillna(0, inplace=True)
    df["totals.sessionQualityDim"] = pd.to_numeric(df["totals.sessionQualityDim"])
    df["totals.sessionQualityDim"].fillna(0, inplace=True)
    df["totals.timeOnSite"]= pd.to_numeric(df["totals.timeOnSite"])
    df["totals.bounces"]= pd.to_numeric(df["totals.bounces"])
    df["totals.bounces"].fillna(0, inplace=True)

    df['date'] = df.apply(lambda x: convert_to_date(x['date']),axis=1)
    df.sort_values(by=['date'], inplace=True, ascending=True)

    logging.info('90% Completed')

    df["trafficSource.isTrueDirect"].fillna(False, inplace=True)

    logging.info('99% Completed')

    if "trafficSource.campaignCode" in df.columns:
        df["trafficSource.campaignCode"].fillna("notSet", inplace=True)

    logging.info('Preprocessing Completed!')

    return df


def model_predict_deprecated(filename,lbl_encoder_container,model):

    numerical_features = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',  'totals.newVisits']

    categorical_features = ['channelGrouping', 'socialEngagementType',
            'device.browser', 'device.browserVersion',
           'device.browserSize', 'device.operatingSystem',
            'device.isMobile',
           'device.mobileDeviceBranding', 'device.mobileDeviceModel',
           'device.mobileInputSelector', 'device.mobileDeviceInfo',
           'device.mobileDeviceMarketingName', 'device.flashVersion',
           'device.language', 'device.screenColors', 'device.screenResolution',
           'device.deviceCategory', 'geoNetwork.continent',
           'geoNetwork.subContinent', 'geoNetwork.country', 'geoNetwork.region',
           'geoNetwork.metro', 'geoNetwork.city', 'geoNetwork.cityId',
           'geoNetwork.networkDomain', 'geoNetwork.latitude',
           'geoNetwork.longitude', 'geoNetwork.networkLocation' , 'trafficSource.campaign',
           'trafficSource.source', 'trafficSource.medium',
           'trafficSource.adwordsClickInfo.criteriaParameters',
           'trafficSource.isTrueDirect',
            'visitstartAMPM']

    df = Preprocess(filename,True)

    logging.info("Encoding Started")

    for col in categorical_features:
        logging.info("Encoding : "+col)
        lbl_encoder = lbl_encoder_container[col]
        le_dict = dict(zip(lbl_encoder.classes_, lbl_encoder.transform(lbl_encoder.classes_)))
        df[col] = df[col].apply(lambda x: le_dict.get(x, 112233445566))
        #df[col]  = lbl_encoder.transform(list(df[col].values.astype('str'))) - gives error while seeing new data
        #Ref : https://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values
    df = df[categorical_features  + numerical_features]
    logging.info("Encoding Completed")


    logging.info("Prediction Started")
    prediction = model.predict(df)
    logging.info("Prediction Completed")


    logging.info("Preparing output")
    output = pd.read_csv(filename, converters={'fullVisitorId': str})
    predict_output = output[['fullVisitorId']].copy()
    predict_output.loc[:, 'PredictedLogRevenue'] = prediction
    predict_output["PredictedLogRevenue"] = predict_output["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
    predict_output["PredictedLogRevenue"] = predict_output["PredictedLogRevenue"].fillna(0.0)
    predict_output["PredictedLogRevenue"] = np.expm1(prediction)
    predict_output = predict_output.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
    predict_output.columns = ["fullVisitorId", "PredictedLogRevenue"]
    #predict_output["PredictedLogRevenue"] = np.log1p(predict_output["PredictedLogRevenue"])
    predict_output["PredictedLogRevenue"] = predict_output["PredictedLogRevenue"].fillna(0.0)
    #predict_output.to_csv("predict_output_5_test.csv", index=False)
    logging.info("Completed output")
    return round(sum(predict_output["PredictedLogRevenue"]),2)



def model_predict(filename,lbl_encoder_container,model,totalshits= 0,totalspageviews= 0,visitNumber= 0 ,visitStartTime=1535084481,totalsbounces=0,totalsnewVisits = 0,
channelGrouping='Display',devicebrowser='Safari (in-app)',  deviceisMobile=True,
devicedeviceCategory='mobile',geoNetworkcontinent='Americas',geoNetworksubContinent='Northern America',
geoNetworkcountry='United States', geoNetworkregion='not available in demo dataset',geoNetworkmetro='not available in demo dataset', geoNetworkcity='not available in demo dataset',
trafficSourcecampaign='(not set)',trafficSourcesource='dfa', trafficSourcemedium='cpm',trafficSourceisTrueDirect='False',date='01/01/2019'):

    numerical_features = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',  'totals.newVisits']

    categorical_features = ['channelGrouping','device.browser','device.isMobile',
                         'device.deviceCategory','geoNetwork.continent','geoNetwork.subContinent',
                         'geoNetwork.country', 'geoNetwork.region','geoNetwork.metro', 'geoNetwork.city',
                         'trafficSource.campaign','trafficSource.source', 'trafficSource.medium',
                         'trafficSource.isTrueDirect','visitstartAMPM']


    if not filename or filename.isspace():
        data = [{ 'channelGrouping' : channelGrouping,
                    'device.browser' : devicebrowser,
                    'device.isMobile' : deviceisMobile,
                    'device.deviceCategory' : devicedeviceCategory,
                    'geoNetwork.continent' : geoNetworkcontinent,
                    'geoNetwork.subContinent' : geoNetworksubContinent,
                    'geoNetwork.country' : geoNetworkcountry,
                    'geoNetwork.region' : geoNetworkregion,
                    'geoNetwork.metro' : geoNetworkmetro,
                    'geoNetwork.city' : geoNetworkcity,
                    'trafficSource.campaign' : trafficSourcecampaign,
                    'trafficSource.source' : trafficSourcesource,
                    'trafficSource.medium' : trafficSourcemedium,
                    'trafficSource.isTrueDirect' : trafficSourceisTrueDirect,
                    'totals.hits' : totalshits,
                    'totals.pageviews' : totalspageviews,
                    'visitNumber' : visitNumber,
                    'totals.bounces' :totalsbounces,
                    'totals.newVisits':totalsnewVisits,
                    'date' : date,
                    'visitStartTime' : visitStartTime}]

        # Creates DataFrame.
        df = pd.DataFrame(data)
        df['date'] = df.apply(lambda x: convert_to_date(x['date']),axis=1)
        df['visitstarthour'] = (df['visitStartTime'].apply(lambda x: str(datetime.fromtimestamp(x).hour))).astype(int)
        df['visitstartAMPM'] = df.apply(lambda x: getAMPM(x['visitstarthour']),axis=1)


    else:
        df = Preprocess(filename)

    logging.info("Encoding Started")

    for col in categorical_features:
        logging.info("Encoding : "+col)
        lbl_encoder = lbl_encoder_container[col]
        le_dict = dict(zip(lbl_encoder.classes_, lbl_encoder.transform(lbl_encoder.classes_)))
        df[col] = df[col].apply(lambda x: le_dict.get(x, 112233445566))
        #df[col]  = lbl_encoder.transform(list(df[col].values.astype('str'))) - gives error while seeing new data
        #Ref : https://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values
    df = df[categorical_features  + numerical_features]
    logging.info("Encoding Completed")


    logging.info("Prediction Started")
    prediction = model.predict(df)
    logging.info("Prediction Completed")



    pred = prediction[0]
    pred = 0.0 if pred < 0 else round(np.expm1(pred),2)

    logging.info("Predicted  - "+str(pred))

    return str(pred)


def train(filename):

    numerical_features = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',  'totals.newVisits']

    categorical_features = ['channelGrouping','device.browser', 'device.operatingSystemVersion','device.isMobile',
                         'device.deviceCategory','geoNetwork.continent','geoNetwork.subContinent',
                         'geoNetwork.country', 'geoNetwork.region','geoNetwork.metro', 'geoNetwork.city',
                         'trafficSource.campaign','trafficSource.source', 'trafficSource.medium',
                         'trafficSource.isTrueDirect','visitstartAMPM']

    df = Preprocess(filename,False)

    logging.info("Splitting the data to Test and Train")

    ev_df = df[df['date']<=pd.to_datetime('2017-5-31')]
    val_df = df[df['date']>pd.to_datetime('2017-5-31')]

    y_train = ev_df['totals.transactionRevenue']
    y_test  = val_df['totals.transactionRevenue']

    X_train = ev_df.drop(['totals.transactionRevenue'],axis=1)
    X_test =  val_df.drop(['totals.transactionRevenue'],axis=1)

    y_train  = np.log1p(y_train)
    y_test  = np.log1p(y_test)

    X_train = X_train[categorical_features  + numerical_features]
    X_test  = X_test [categorical_features  + numerical_features]


    logging.info("Encoding Started")

    lblencoders = {}
    for col in categorical_features:
        logging.info("Encoding : "+col)
        lbl = LabelEncoder()
        lbl.fit(list(X_train[col].values.astype('str')) + list(X_test[col].values.astype('str')))
        lblencoders[col] = lbl
        X_train[col] = lbl.transform(list(X_train[col].values.astype('str')))
        X_test[col]  = lbl.transform(list(X_test[col].values.astype('str')))
    logging.info("Encoding Completed")

    pickle.dump(lblencoders, open('lbl_encoders_test.pkl', 'wb'))

    logging.info("Model Training Started")
    x_reg=XGBRegressor(n_estimators=10,nthread=-1)
    x_reg.fit(X_train,y_train)

    predict_y_train = x_reg.predict(X_train)
    logging.info("The train loss is:",mean_squared_error(y_train,predict_y_train))

    predict_y_test = x_reg.predict(X_test)
    logging.info("The test loss is:",mean_squared_error(y_test,predict_y_test))


    pickle.dump(x_reg, open("googlestore_model_test.pkl", 'wb'))
    logging.info("Model Training Completed")


@app.route('/predict', methods=['GET', 'POST'])
def upload():

    logging.info('Request Parsed')

    lbl_encoder_container = pickle.load(open(ENCODER_PATH, 'rb'))

    model = pickle.load(open(MODEL_PATH, 'rb'))

    out = model_predict('',
                        lbl_encoder_container,
                        model,
                        totalshits=  int(request.args['totalshits']),
                        totalspageviews= int(request.args['totalspageviews']),
                        visitNumber=  int(request.args['visitNumber']) ,
                        visitStartTime=int(request.args['visitStartTime']),
                        totalsbounces=int(request.args['totalsbounces']),
                        totalsnewVisits = int(request.args['totalsnewVisits']),
                        channelGrouping=str(request.args['channelGrouping']),
                        devicebrowser=str(request.args['devicebrowser']),
                        deviceisMobile=str(request.args['deviceisMobile']),
                        devicedeviceCategory=str(request.args['devicedeviceCategory']),
                        geoNetworkcontinent=str(request.args['geoNetworkcontinent']),
                        geoNetworksubContinent=str(request.args['geoNetworksubContinent']),
                        geoNetworkcountry=str(request.args['geoNetworkcountry']),
                        geoNetworkregion=str(request.args['geoNetworkregion']),
                        geoNetworkmetro=str(request.args['geoNetworkmetro']),
                        geoNetworkcity=str(request.args['geoNetworkcity']),
                        trafficSourcecampaign=str(request.args['trafficSourcecampaign']),
                        trafficSourcesource=str(request.args['trafficSourcesource']),
                        trafficSourcemedium=str(request.args['trafficSourcemedium']),
                        trafficSourceisTrueDirect=str(request.args['trafficSourceisTrueDirect']),
                        date=str(request.args['date']))




    return str(out)


@app.route('/downloadsample/<path:filename>', methods=['GET', 'POST'])
def download_results(filename):
    return send_from_directory('uploads', filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
