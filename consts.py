#//Ran Shem Tov      -   206059586
#//Natali Mahmali    -   311266399
import enum as Enum
#UI (snakeClass.py)
showGame = True
speed = 1
trains = 10
#agent model
outputDim = 120
outputDimLast = 3
inputDim = 11
dropout = 0.15
activ = 'relu'
activLast = 'softmax'
compileLoss = 'mse'
memoryMax = 1000
epochs = 1
verbose = 0
#agent init settings
reward = 0
gamma = 0.9
agentTarget = 1
agentPredict = 0
learningRate = 0.0005
epsilon = 0
#rewards
crashReward = -10
eatenReward = 10
#Game class
score = 0
#snakeClass.py
wait = 1
epslonInit = 80
#
bestRunI = 0
   # # # # # # # # # # # # # #
  # BACKUP default settings #
 # # # # # # # # # # # # # #
# import enum as Enum
# #UI (snakeClass.py)
# showGame = True
# speed = 10
# trains = 150
# #agent model
# outputDim = 120
# outputDimLast = 3
# inputDim = 11
# dropout = 0.15
# activ = 'relu'
# activLast = 'softmax'
# compileLoss = 'mse'
# memoryMax = 1000
# epochs = 1
# verbose = 0
# #agent init settings
# reward = 0
# gamma = 0.9
# agentTarget = 1
# agentPredict = 0
# learningRate = 0.0005
# epsilon = 0
# #rewards
# crashReward = -10
# eatenReward = 10
# #Game class
# score = 0
# #snakeClass.py
# wait = 150
# epslonInit = 80