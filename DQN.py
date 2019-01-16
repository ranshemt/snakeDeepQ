#//Ran Shem Tov      -   206059586
#//Natali Mahmali    -   311266399
#
#
#optimizer, required for keras model compilation
from keras.optimizers import Adam
#basic model
from keras.models import Sequential
#how we will stack the layers
from keras.layers.core import Dense, Dropout
#python casual libraries
import random
import numpy as np
#data structurs & data anylysis tools
import pandas as pd
#operators
from operator import add
#my constants
import consts as C
#
#
#
class DQNAgent(object):
    #constructor
    def __init__(self):
        self.reward = C.reward
        #higher gamma = smaller discount = cares more about long term reward
            #(example of the rat and cheese)
        self.gamma = C.gamma
        #DataDrame can contain any type of data
            #easier to work with it than list/dictionary
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        #prediction, the goal of the neural netwrok is to minimize the loss
            #reducing the difference between agent target and predicted target
        self.agent_target = C.agentTarget
        self.agent_predict = C.agentPredict
        #how quickly the network abandons former value for new one
            #easier to understand with q table (simple q learning)
        self.learning_rate = C.learningRate
        #new network (untrained)
        self.model = self.network()
        #
        #
        #old network
        self.model = self.network("weights.hdf5")
        #
        #
        # low epsilon = exploitation = rely on data
        self.epsilon = C.epsilon
        #
        
        #long term memory
        self.memory = []
    #
    #get state of the player (snake) in the game
    def get_state(self, game, player, food):
        #update the states (11 possible)
        state = [
            #
            # danger straight
            (player.x_change == 20
                and player.y_change == 0
                and ((list(map(add, player.position[-1], [20, 0]))
                        in player.position)
                    or player.position[-1][0] + 20 >= (game.game_width - 20))
            )
            or(player.x_change == -20
                and player.y_change == 0
                and ((list(map(add, player.position[-1], [-20, 0]))
                        in player.position)
                    or player.position[-1][0] - 20 < 20)
            )
            or (player.x_change == 0
                and player.y_change == -20
                and ((list(map(add, player.position[-1], [0, -20]))
                        in player.position)
                    or player.position[-1][-1] - 20 < 20)
            )
            or (player.x_change == 0
                and player.y_change == 20
                and ((list(map(add, player.position[-1], [0, 20]))
                        in player.position)
                    or player.position[-1][-1] + 20 >= (game.game_height - 20))
            ),
            #
            # danger right
            (player.x_change == 0
                and player.y_change == -20
                and ((list(map(add,player.position[-1],[20, 0]))
                        in player.position)
                    or player.position[-1][0] + 20 > (game.game_width - 20))
            )
            or (player.x_change == 0
                and player.y_change == 20
                and ((list(map(add,player.position[-1], [-20, 0]))
                        in player.position)
                    or player.position[-1][0] - 20 < 20)
            )
            or (player.x_change == -20
                and player.y_change == 0
                and ((list(map(add,player.position[-1],[0, -20]))
                        in player.position)
                    or player.position[-1][-1] - 20 < 20)
            )
            or (player.x_change == 20
                and player.y_change == 0
                and ((list(map(add,player.position[-1],[0, 20]))
                        in player.position)
                    or player.position[-1][-1] + 20 >= (game.game_height - 20))
            ),
            #
            #danger left
            (player.x_change == 0 and
                player.y_change == 20
                and ((list(map(add,player.position[-1],[20, 0]))
                        in player.position)
                    or player.position[-1][0] + 20 > (game.game_width - 20))
            )
            or (player.x_change == 0
                and player.y_change == -20
                and ((list(map(add, player.position[-1],[-20, 0]))
                        in player.position)
                    or player.position[-1][0] - 20 < 20)
            )
            or (player.x_change == 20
                and player.y_change == 0
                and ((list(map(add,player.position[-1],[0, -20]))
                        in player.position)
                    or player.position[-1][-1] - 20 < 20)
            )
            or (player.x_change == -20
                and player.y_change == 0
                and ((list(map(add,player.position[-1],[0, 20]))
                        in player.position)
                    or player.position[-1][-1] + 20 >= (game.game_height - 20))
            ),
            #
            #
            player.x_change == -20, # move left
            player.x_change == 20,  # move right
            player.y_change == -20, # move up
            player.y_change == 20,  # move down
            food.x_food < player.x, # food left
            food.x_food > player.x, # food right
            food.y_food < player.y, # food up
            food.y_food > player.y  # food down
        ]
        #return the current state
        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0

        return np.asarray(state)
    #
    #rewards
    def set_reward(self, player, crash):
        #@#@try to give other rewards??
        self.reward = 0
        if crash:
            self.reward = C.crashReward
            return self.reward
        if player.eaten:
            self.reward = C.eatenReward
        return self.reward
    #
    #
      # # # # # # # # # # # # # # #
     # definning model for agent #
    # # # # # # # # # # # # # # #
    def network(self, weights=None):
        #model is a data structure of Kears
            #Sequiential is the basic one (linear stack of layers)
                #linear = no branching, every layer has one input and output
                #the output of one layer is the input of the following layer
        #default model
        model = Sequential()
        #first layer indicates which shape the model needs to receive
            #expected input data shape is 11 dimensional array
        model.add(Dense(output_dim = C.outputDim, activation=C.activ, input_dim=C.inputDim))
        #Dropout is used to prevent overfitting
        model.add(Dropout(C.dropout))
        model.add(Dense(output_dim = C.outputDim, activation=C.activ))
        model.add(Dropout(C.dropout))
        model.add(Dense(output_dim = C.outputDim, activation=C.activ))
        model.add(Dropout(C.dropout))
        model.add(Dense(output_dim=C.outputDimLast, activation=C.activLast))
        #
        #Adam = method for stockastic (random) optimization
            # with learning rate as we defined
            # can also get (bet1_1, beta_2, epsilon, decay, amsgrad)
        opt = Adam(self.learning_rate)
        #mse = mean squared error [regression problem]
            # the average squared difference
            # between the estimated values and what is estimated
                # (predict/target)
        #
        # Configuring the learning process with model.compile()
        model.compile(loss=C.compileLoss, optimizer=opt)
        #
        #load 'old' data, already trained agent
        if weights:
            model.load_weights(weights)
        #
        return model
    #
    #long term memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    #
    #in the new learning loop train the agent randomly with the long term memory
    def replay_new(self, memory):
        if len(memory) > C.memoryMax:
            minibatch = random.sample(memory, C.memoryMax)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
    #
    #
    #custom function for trainning using model.fit
        #could be written directly using keras API
    def train_short_memory(self, state, action, reward, next_state, done):
        #creating numpy array from reward-gamma-prediction
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 11)))[0])
        target_f = self.model.predict(state.reshape((1, 11)))
        target_f[0][np.argmax(action)] = target
        #actual trainning
            #2 numpy arrays
            #epochs = nuber of iterations on data provided
            #verbose = verbosity (Gibuv) mode, 0 = silent
        self.model.fit(state.reshape((1, 11)), target_f, epochs=1, verbose=0)
    #end of class DQNAgent