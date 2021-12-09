##from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product



class TicTacToe():

    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix

        self.reset()


    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        ## converting the input to 3*3 shape
        state = np.reshape(curr_state, (3, 3))
        
        ## we check for the total of 15 in vertical axis, if nan is found, the step is not completed and hence keeping it as nan itself
        vertical_axis = (15 in (np.sum(state, axis=0)))
        
        ## We check for the total of 15 in horizontal axis, if nan is found, the step is not completed and hence keeping it as nan itself
        horizontal_axis = (15 in (np.sum(state, axis=1)))
        
        ## We check for the diagonal count for both left to right diagonal and right to left diagonal
        lr_diagonal = ((np.trace(state)) == 15)
        
        rl_diagonal = (np.trace(np.fliplr(state)) == 15)
        
        ##along_axis = (15.0 in np.concatenate((np.nansum(state, axis=1), np.nansum(state, axis=0))))

        ##return ((np.trace(state) == 15.0) or (np.trace(np.fliplr(state)) == 15.0) or (along_axis))
        return (vertical_axis or horizontal_axis or lr_diagonal or rl_diagonal)
         

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        curr_state[curr_action[0]] = curr_action[1]
        return curr_state


    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        
        last_state = False
        newnext_state = self.state_transition(curr_state, curr_action)
        last_state, currgame_status = self.is_terminal(newnext_state)
        if last_state == True:
            if currgame_status == 'Win':
                reward=10  ## +10 if the agent wins (makes 15 points first)
            else:
                reward=0  ## 0 if the game ends in a draw (no one is able to make 15 and the board is filled up)
        else:
            reward=-1 ## -1 for each move agent takes
            pos = random.choice(self.allowed_positions(newnext_state))
            val = random.choice(self.allowed_values(newnext_state)[1])
            newnext_state[pos]= val
            last_state, currgame_status = self.is_terminal(newnext_state)
            if last_state == True:
                if currgame_status == 'Win':
                    reward=-10  ## -10 if the environment wins
                else:
                    reward=0  ## 0 if the game ends in a draw (no one is able to make 15 and the board is filled up)

        return newnext_state, reward, last_state



    def reset(self):
        return self.state
