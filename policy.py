import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
import keras.layers as layers
from keras.models import Model
from keras.optimizers import SGD
import keras.backend as K
sns.set()


# def sigmoid(centers, stretch, x):
#     res = 1 / (1 + np.exp(-stretch*(x-centers)))
#     return res
#
#
# def sqrt(origin, x):
#     res = np.sqrt(x-origin)
#     return res
#
#
# def reward_1(df):
#     f = 1/(df['flux'].values/10) * 100
#     time = df['TimeObs'].values
#     rewards = sigmoid(f, 1/2, time)
#     rewards *= f
#     rewards[f == np.inf] = -5*time[f == np.inf]
#     return np.sum(rewards), rewards
#
#
# def reward_2(df):
#     f = 1 / (df['flux'].values / 10) * 100
#     time = df['TimeObs'].values
#     rewards = sqrt(f, time)
#     rewards *= f
#     rewards[np.isnan(rewards)] = 0
#     return np.sum(rewards), rewards

# def train_model(model, states, actionvectors, rewards):
#     X_train = states
#     Y_train = actionvectors
#     discount_rewards = rewards - np.mean(rewards)
#     model.fit(X_train, Y_train, sample_weight=discount_rewards, epochs=2, verbose=0)


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
numberofepisodes = 3

inp = layers.Input(shape=(vectorlength, ), name="input_x")
adv = layers.Input(shape=(1,), name="advantages")
x1 = layers.Dense(500,
                 activation="relu",
                 name="dense_1")(inp)
x2 = layers.Dense(1000,
                 activation="relu",
                 name="dense_2")(x1)
x3 = layers.Dense(1000,
                 activation="relu",
                 name="dense_3")(x2)
x4 = layers.Dense(1000,
                 activation="relu",
                 name="dense_4")(x3)
x5 = layers.Dense(1000,
                 activation="relu",
                 name="dense_5")(x4)
x6 = layers.Dense(500,
                 activation="relu",
                 name="dense_6")(x5)
out = layers.Dense(numobj,
                   activation="softmax",
                   name="out")(x6)


def custom_loss(y_true, y_pred):
    log_lik = K.log(1e-7+(y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred)))
    return K.sum(log_lik * adv)

model = Model(inputs=[inp, adv], outputs=out)
model.compile(loss=custom_loss, optimizer=SGD(lr=1e-4))
modelp = Model(inputs=[inp], outputs=out)


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
        modelaction = modelp.predict(state)[0]
        print(modelaction)
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
    alltrainX = np.vstack([alltrainX, trainX])
    alltrainY = np.vstack([alltrainY, trainY])
    alladvantages = np.append(alladvantages, advantage)
    alladvantages -= alladvantages.mean()
    alladvantages /= alladvantages.std()
    # print(trainX, trainY, advantage)
    model.train_on_batch([trainX, advantage], trainY)
    # print(totreward)
