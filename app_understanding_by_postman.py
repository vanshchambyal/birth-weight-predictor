from flask import Flask, request,jsonify
import pandas as pd
import pickle

app= Flask(__name__)

## define end point
@app.route("/predict", methods=["POST"])
def get_prediction():
    # get data from user
    baby_data= request.get_json()

    # converting into dataframe
    baby_df = pd.DataFrame(baby_data)


    # load machine learning trained model
    with open("saved_model/model.pkl", "rb") as obj:
        model= pickle.load(obj) # it will go to given directory read the particular file as binary mode and then load it as obj

    # make prediction on user data
    prediction= model.predict(baby_df)
    prediction = round(float(prediction[0]), 2) # we want to keep value upto 2 decimal
    
    # return responsein json format
    response ={"prediction": (prediction)}

    return jsonify(response)







if __name__ =="__main__":
    app.run(debug=True)