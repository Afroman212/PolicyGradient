import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import keras.backend as K
sns.set()

#two reward functions, one gives reward at end of episode, other gives reward after each state update
def reward_final(fluxes, times):
    multiply = times*0.2
    multiply[multiply > 1] = 1
    reward = multiply*fluxes/10
    f_reward = np.sum(reward)
    return f_reward


def reward_func(fluxes, action, times):
    times = times[action]
    if times <= 5:
        rew = fluxes[action]/5
    else:
        rew = -50
    return rew


def decay(length):
    decays = []
    for k in range(length):
        decays.append(0.99**k)
    return np.array(decays)


# setup the states
np.random.seed(1234)
df = pd.DataFrame()

fluxes = np.arange(600, 100, -100)
df['flux'] = fluxes
df['TimeObs'] = 0

numfeatures = len(df.columns)
numobj = len(df)
vectorlength = numobj*numfeatures

df['x'] = np.random.randint(1, 16, numobj)
df['y'] = np.random.randint(1, 16, numobj)

maxtime = 15
numberofepisodes = 100


def custom_loss(y, y_pred):
    a = y[:, -1]
    a = K.expand_dims(a, 1)
    y_true = y[:, 0:5]
    log_lik = K.log(1e-7+(y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred)))
    f = K.sum(log_lik * a)
    return f

model = Sequential()
model.add(Dense(100, input_shape=(vectorlength,), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(numobj, activation='softmax'))
model.compile(loss=custom_loss,
              optimizer=SGD(lr=1e-3))


alltrainX = np.empty(0).reshape(0, vectorlength)
alltrainY = actions = np.empty(0).reshape(0, numobj)
alladvantages = np.empty(0)

for i in range(numberofepisodes):
    fluxes = np.arange(600, 100, -100)
    df['flux'] = fluxes
    df['TimeObs'] = 0

    trainX = np.empty(shape=(0, vectorlength), dtype=float)
    trainY = np.empty(shape=(0, numobj), dtype=float)
    rewards = np.empty(shape=(0,), dtype=float)
    totreward = 0

    for j in range(maxtime):
        state = np.empty(shape=(1, vectorlength))
        state[0] = df[['flux', 'TimeObs']].values.flatten()
        x = np.random.random()
        modelaction = model.predict(state)[0]
        action = np.argmax(modelaction)

        if x < 0 and i < 0.75*numberofepisodes:
            action = np.random.choice(numobj)
        else:
            action = np.argmax(modelaction)

        # predict = modelp.predict([state])[0]
        # action = np.random.choice(range(numobj), p=predict)
        action_one_hot = np.zeros(numobj)
        action_one_hot[action] = 1
        action_prob_one_hot = modelaction * 0.9*action_one_hot
        trainX = np.append(trainX, [state[0]], axis=0)
        trainY = np.append(trainY, [action_one_hot], axis=0)

        times = df['TimeObs'].values
        fluxes = df['flux'].values
        times[action] += 1
        if j == maxtime-1:
            reward = reward_final(fluxes, times)
        else:
            reward = 0
        rewards = np.append(rewards, reward)
        df['TimeObs'] = times
        # print(times)
        totreward += reward
    adv = []

    for x in range(len(rewards)):
        length = maxtime-x
        d = decay(length)
        sums = np.sum(rewards[x:]*d)
        adv.append(sums)

    advantage = np.array(adv, dtype=float)
    alladvantages = np.append(alladvantages, advantage)
    alladvantages -= alladvantages.mean()
    alladvantages /= alladvantages.std()

    alltrainX = np.vstack([alltrainX, trainX])
    alltrainY = np.vstack([alltrainY, trainY])

    # print(trainX, trainY, advantage)
    tempalltrainY = np.insert(alltrainY, 5, alladvantages, axis=1)
    model.train_on_batch(alltrainX, tempalltrainY)
    # print(totreward)
