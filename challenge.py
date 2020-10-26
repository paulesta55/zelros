import argparse
import jwt
import logging
import os
import tempfile
import time
import uuid

import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)


def getDataset(id):
    URL = "http://pms.zelros.com/ml/dataset/?id=" + str(id)
    r = requests.get(url=URL)
    fd, fname = tempfile.mkstemp()
    with open(fname, 'wb') as ff2:
        ff2.write(r.content)
    os.close(fd)
    df = pd.read_csv(fname)
    os.remove(fname)
    return df


def trainModel(dataframe):
    y = dataframe['t']
    X = dataframe[['a', 'b', 'c']]
    # Better than SVR or LinearRegression
    reg = RandomForestRegressor(max_depth=25, random_state=0).fit(X, y)
    return reg


def getInput(id):
    URL = "http://pms.zelros.com/ml/?id=" + str(id)
    r = requests.get(url=URL)
    input = r.json()
    input = [[input['a'], input['b'], input['c']]]
    return input


def getPeople(id):
    URL = "http://pms.zelros.com/people?id=" + str(id)
    r = requests.get(url=URL)
    logging.info(f"people: {r.content}")
    return r.content


def getFingerprint(id):
    URL = "http://pms.zelros.com/fingerprint?id=" + str(id)
    r = requests.get(url=URL)
    return r.content

def getScore(id, prediction):
    URL = "http://pms.zelros.com/?id=" + str(id)
    # Add +100 to score
    HEADERS = {'user-agent': 'zelros'}
    r = requests.post(url=URL, headers=HEADERS, data={"predict": prediction})
    info = r.json()
    return info["score"], info["win"]


def prepare(id):
    URLPrepare = "http://pms.zelros.com/prepare?id=" + str(id)
    # Add +50 to score
    requests.post(url=URLPrepare, data={"score": 200})


def solveWithMl(limit=999999, id=None):
    if id is None:
        id = uuid.uuid4()
    logging.info("id: {}".format(id))
    dataframe = getDataset(id)
    model = trainModel(dataframe)
    score = 0
    # Probably supposed to be sent somewhere. Can't find the good URL though
    people = getPeople(id)
    numBadPredictions = 0
    # solve bonus 0001
    for i in range(30):
        # Can send only one request/s
        time.sleep(1)
        prepare(id)
        time.sleep(1)
        score, win = getScore(id, None)
        logging.info(f"score: {score}")
        logging.info(f"win: {win}")
    while score < limit:
        time.sleep(1)
        inputs = getInput(id)
        start = time.time()
        prediction = model.predict(inputs)[0]
        delta = time.time() - start
        time.sleep(1 - delta)
        score, win = getScore(id, prediction)
        logging.info(f"score: {score}")
        logging.info(f"win: {win}")
        # Heuristic to detect dataset changes
        if win < 1000:
            numBadPredictions += 1
        else:
            numBadPredictions = 0
        if numBadPredictions > 10:
            dataframe = getDataset(id)
            model = trainModel(dataframe)
    # get JWT token
    time.sleep(1)
    fingerprint = getFingerprint(id)
    logging.info(f"id : {id}")
    logging.info(f"fingerprint : {fingerprint}")
    logging.info(jwt.decode(fingerprint, verify=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--id", default=None, required=False)
    parser.add_argument("-l", "--limit", default=None, required=False, type=int)
    args = parser.parse_args()
    if args.id is not None:
        if args.limit is not None:
            solveWithMl(id=args.id, limit=int(args.limit))
        else:
            solveWithMl(id=args.id)
    elif args.limit is not None:
        solveWithMl(limit=int(args.limit))
    else:
        solveWithMl()
