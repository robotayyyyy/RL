import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#Locate current script dir, then add to sys path
import sys
current_dir = 'D:/GitHub/RL/'#manual set
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import fileRecord

class Agent:

    def __init__(self, state_size, action_size, batch_size):
        self.state_size    = state_size
        self.action_size   = action_size
        self.batch_size    = batch_size
        self.memory        = deque(maxlen=2000)
        self.gamma         = 0.95   # discount rate
        self.epsilon       = 1.0    # exploration rate
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.995
        self.dqn           = self.build_model()
    
    def rewardFunction2(self, nextState, reward, isAbsorb, info):#best
        
        if isAbsorb:
            reward = -1
        else:
            Angle = nextState[2]
            AngleV = nextState[3]
            
            if Angle <0 :
                if AngleV <0:
                    reward = 0
                else:
                    reward = 1
            else :
                if AngleV <0:
                    reward = 1
                else:
                    reward = 0
        return(reward)
    
    def build_model(self):
        dqn = Sequential()
        dqn.add(Dense(24, input_dim=self.state_size, activation='tanh')) # input dimension = #states
        dqn.add(Dense(self.action_size, activation='linear'))            # output nodes = #action
        dqn.compile(loss='mse', optimizer=Adam(lr=0.01))                      
        print(dqn.summary())
        return dqn

    def act(self, state, explore):
        if explore and np.random.rand() <= self.epsilon: # explore/exploit tradeoff
            return random.randrange(self.action_size)
        act_values = self.dqn.predict(state)
        return np.argmax(act_values[0]) 

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        if len(self.memory)<self.batch_size:
            return

        X, dqnY = [], []
        minibatch = random.sample(self.memory, self.batch_size) 
        for state, action, reward, next_state, done in minibatch:
            X.append( state[0] )
            target = reward if done else reward + self.gamma * np.max(self.dqn.predict(next_state)[0])
            target_dqn = self.dqn.predict(state)
            target_dqn[0][action] = target
            dqnY.append( target_dqn[0] )

        self.dqn.train_on_batch( np.array(X), np.array(dqnY) )

        if self.epsilon > self.epsilon_min:    # gradually change from explore to exploit
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env._max_episode_steps = 50000
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print("{} actions, {}-dim state".format(action_size, state_size))

    agent = Agent(state_size, action_size, 32)
    
    emax = 1500 #emax is max of episode
    

    consecutiveSuccess = []
    averageReward = []
        
    for e in range(emax):
        state = env.reset()
        state = state.reshape((1, state_size))
        for time in range(6000):#6000 is max of time step
            #env.render()
            action = agent.act(state, True)
            next_state, reward, done, _ = env.step(action)
            
            #reward = agent.rewardFunction2(next_state, reward, done, '') #-------------
            
            next_state = next_state.reshape( (1,state_size) )
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        print("episode: {}/{}, score: {}, e: {:.2}".format(e, emax, time, agent.epsilon))
        #train here-----------------------    
        agent.train()
        
        #test here------------------------
        sumReward =0
        sumTimeStep = 0
        for t in range(100):#test for 100 episode
            success = False
            state = env.reset()
            state = state.reshape((1, state_size))
            for time in range(6000):
                #env.render()
                action = agent.act(state, False)
                next_state, reward, done, _ = env.step(action)
                sumReward += agent.rewardFunction2(next_state, reward, done, '')
                sumTimeStep+=1
                next_state = next_state.reshape( (1,state_size) )
                state = next_state
                
                if time>=5999:
                    success = True
                    print('S time:'+str(time))
                    break
                if done:
                    success = False
                    print('F time:'+str(time))
                    break
            if not success:
                break
            #record--------------------------------

        consecutiveSuccess.append(t)
        averageReward.append(sumReward/sumTimeStep)

        
        if success:
            print("Cartpole-v0 SOLVED!!! {}".format(e))
            fileRecord.ExpSaveLoad().saveExp('DQN_Sparse_Reward1',[1,1],['consecutiveSuccess','average reward'],consecutiveSuccess=consecutiveSuccess, averageReward=averageReward)
            break
            
