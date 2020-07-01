import os
import sys
import flask
from flask import request, jsonify
from web3 import Web3
import web3
import json
import http
import numpy as np
import pandas as pd
from flask_cors import CORS
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score




app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True

blockchain_url = 'https://kovan.infura.io/v3/' + \
    os.environ['WEB3_INFURA_PROJECT_ID']
abi = """[{"anonymous": false,"inputs": [{"indexed": false,"internalType": "address","name": "deviceID","type": "address"},{"indexed": false,"internalType": "string","name": "latestCID","type": "string"}],"name": "MappingUpdated","type": "event"},{"inputs": [{"internalType": "address","name": "deviceID","type": "address"},{"internalType": "string","name": "latestCID","type": "string"}],"name": "setLatestCID","outputs": [],"stateMutability": "nonpayable","type": "function"},{"inputs": [],"name": "getDeviceIDsLength","outputs": [{"internalType": "uint256","name": "","type": "uint256"}],"stateMutability": "view","type": "function"},{"inputs": [{"internalType": "uint256","name": "index","type": "uint256"}],"name": "getIDByIndex","outputs": [{"internalType": "address","name": "","type": "address"}],"stateMutability": "view","type": "function"},{"inputs": [{"internalType": "address","name": "deviceID","type": "address"}],"name": "getLatestCID","outputs": [{"internalType": "string","name": "latestCID","type": "string"}],"stateMutability": "view","type": "function"}]"""

conn = http.client.HTTPSConnection("kfs2.moibit.io")
moibit_url = 'https://kfs2.moibit.io/moibit/v0/'
moibit_header_obj = {
    'api_key': os.environ['MOIBIT_API_KEY'],
    'api_secret': os.environ['MOIBIT_API_SECRET'],
    'content-type': "application/json"
}

masterDataSet = []


@app.route('/', methods=['GET'])
def home():
    return "<h1>DICTAO - Decentralized Intelligent Contact Tracing of Animals and Objects</h1><p>This is a simple demonstration of applying blockchain, decentralized storage and AI to solve the COVID-19 crisis.</p>"


@app.errorhandler(404)
def page_not_found(e):
    return "The given ID could not be found", 404


@app.errorhandler(500)
def internal_server_error(e):
    return e, 500


@app.route('/api/v0/get_infections', methods=['GET'])
def get_infections():
    query_parameters = request.args
    if 'id' in query_parameters:
        id = query_parameters.get('id')
        print("Received ID from the user: "+id)
        if getLatestCID(id) == "":
            return page_not_found(404)
        else:
            # TODO: Find infections
            w3 = Web3(Web3.HTTPProvider(blockchain_url))
            contract = w3.eth.contract(
                os.environ['PROOF_SMART_CONTRACT_ADDRESS'], abi=abi)
            length = contract.functions.getDeviceIDsLength().call()
            print("Length of the deviceIDs: "+str(length))
            for i in range(length):
                tempId = contract.functions.getIDByIndex(i).call()
                # print(tempId)
                tempHash = contract.functions.getLatestCID(tempId).call()
                # print(tempHash)
                jsonData = getJsonDataFromMoiBit(tempHash)
                # print(jsonData)
                for location in jsonData:
                    masterDataSet.append(location)
            print("Generated live dataset of length: %d" % len(masterDataSet))
            with open('live_dataset.json', 'x') as outfile:
                json.dump(masterDataSet, outfile, indent=2)
            results = get_infected_ids(id)
            os.remove("live_dataset.json")
            return (jsonify(results))
    else:
        return "Error: Please specify an ID to identify potential infections."


def get_infected_ids(id):
    dtc = train('training_dataset.json')
    basePath = os.path.dirname(os.path.abspath(__file__))
    live_dataset_path = basePath + '/' + 'live_dataset.json'
    X = pd.read_json(live_dataset_path)
    X = X.sort_values(by=['timestamp'])
    X['timestamp'] = pd.to_datetime(X['timestamp'].sort_values())
    X['year'] = X['timestamp'].dt.year
    X['month'] = X['timestamp'].dt.month
    X['day'] = X['timestamp'].dt.day
    X['hour'] = X['timestamp'].dt.hour
    X['minute'] = X['timestamp'].dt.minute
    X['second'] = X['timestamp'].dt.second
    # sep = pd.DataFrame(columns=['infection'])
    # X
    # df_total = pd.merge(X,sep[['infection']],how = 'right',left_index = True, right_index = True)

    # sep.reset_index(level=0, inplace=True)
    # print(final_df)
    # final_df = X.merge(sep, on='index', indicator = True)
    ids = pd.DataFrame(X['id'])
    timestamps = pd.DataFrame(X['timestamp'])
    X.drop(['timestamp', 'id'], axis = 1, inplace=True)

    # X = X.drop(['id'],axis= 1)
    # y = pd.DataFrame(X['infection'])
    presub = pd.DataFrame(ids.join(timestamps))
    sub = pd.DataFrame(presub.join(X))
    y = dtc.predict(X)
    print(y)
    infection = pd.DataFrame(y)
    final = pd.DataFrame(sub.join(infection))
    final.rename(columns = {0:'infection'}, inplace=True)
    final.drop(['year', 'month', 'day', 'hour', 'minute', 'second'], axis = 1, inplace=True)
    print(final.head())
    print(final.columns)
    final.sort_values(by = 'timestamp')
    # final.drop(['timestamp', 'latitude', 'longitude'], axis = 1)


    # print(X.head())
    # X = pd.get_dummies(X, columns=['id'], drop_first=True)
    
    
    # X['infection'] = infection
    # X.join(infection)
    print(final.head(50))

    # live_dataset_file = open('live_dataset.json', 'r')
    # live_dataset_data = json.load(live_dataset_file)
    # print(live_dataset_data)

    
    potential_infected_ids = ["0xABC", "0xDEF", "0xGHI", "0xJKL", "0xMNO"]
    results = {
        "id": id,
        "potential_infected_ids": potential_infected_ids
    }
    return results


def train(dataset):
    basePath = os.path.dirname(os.path.abspath(__file__))
    training_dataset_path = basePath + '/' +dataset
    df = pd.read_json(training_dataset_path)
    df = df.sort_values(by=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].sort_values())
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    df.drop(['timestamp', 'id'], axis = 1, inplace=True)
    # df = pd.get_dummies(df, columns=['id'], drop_first=True)

    X = df.drop(['infection'],axis= 1)
    y = df['infection']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    dtc = DecisionTreeClassifier(criterion='gini', splitter='best')
    dtc.fit(X_train, y_train)
    pred = dtc.predict(X_test)
    print(classification_report(pred, y_test))
    print(confusion_matrix(pred, y_test))
    print('Accuracy score:\n',accuracy_score(y_test, pred))
    return dtc


def getJsonDataFromMoiBit(cid):
    pre_payload = {"hash": cid}
    payload = json.dumps(pre_payload)
    conn.request("POST", moibit_url+"readfilebyhash",
                 payload, moibit_header_obj)
    res = conn.getresponse()
    if res.status == 200:
        responseObject = json.loads(res.read())
        print(
            "updateLocationHistory(): Appending the captured data to historic data.")
        return responseObject


def getLatestCID(id):
    w3 = Web3(Web3.HTTPProvider(blockchain_url))
    contract = w3.eth.contract(
        os.environ['PROOF_SMART_CONTRACT_ADDRESS'], abi=abi)
    cid = ""
    try:
        cid = contract.functions.getLatestCID(id).call()
    except web3.exceptions.ValidationError:
        print("ID does not exist!")
        return ""
    except:
        print("Some other error occured!")
        return ""
    else:
        print(cid)
        return cid


app.run()
