import torch
import torch.nn.functional as F
import string
import numpy as np
import agent
from cooperative_craft_world import CooperativeCraftWorldState
from dqn import DQN, DQN_Config
from dialog import Dialog

import os
import re

def find_folders(directory, required_objects):
    pattern = r'([a-zA-Z]+_\d+\.?\d*)'
    trained_model_paths = []
    trained_model_goal_dics = []
    
    # Iterate through the directory
    for root, dirs, files in os.walk(directory):
        for folder_name in dirs:
            matches = [x.split('_') for x in re.findall(pattern, folder_name)]
            obj_list = [x[0] for x in matches]
            weight_list = [x[1] for x in matches]
            if set(obj_list) == set(required_objects):
                trained_model_paths.append(folder_name)
                trained_model_goal_dic = {}
                for obj in required_objects:
                    trained_model_goal_dic[obj] = weight_list[obj_list.index(obj)]
                trained_model_goal_dics.append(trained_model_goal_dic)
    
    return trained_model_paths, trained_model_goal_dics

class GoalRecogniser(object):

    def __init__(self, goal_list=[], model_temperature=0.01, hypothesis_momentum=0.9999, kl_tolerance=0.0, saved_model_dir=None, dqn_config:DQN_Config=None, show_graph=False, log_dir=None):

        self.saved_model_dir = saved_model_dir
        self.model_temperature = model_temperature
        self.hypothesis_momentum = hypothesis_momentum
        self.kl_tolerance = kl_tolerance
        self.show_graph = show_graph
        self.log_dir = log_dir
        self.first_log_write = True
        self.step_number = 0
        self.dqn_config = dqn_config

        self.device = torch.device("cuda" if dqn_config.gpu >= 0 else "cpu")
        self.probability_plot = Dialog()

        self.goal_list = goal_list

        self.trained_model_paths, self.trained_model_goal_dics = find_folders(self.saved_model_dir, self.goal_list)

        self.trained_models = []
        for i in range(len(self.trained_model_paths)):
            model_file = f'{self.saved_model_dir}/{self.trained_model_paths[i]}/model.chk'
            checkpoint = torch.load(model_file, map_location=self.device)
            self.trained_models.append(DQN(self.dqn_config))
            self.trained_models[i].load_state_dict(checkpoint['model_state_dict'])
        print(f"GR model loaded: {self.trained_model_paths}")

    def set_external_agent(self, other_agent:agent.Agent):

        self.other_agent = other_agent
        # print(list(other_agent.externally_visible_goal_sets.keys()) + [self.other_agent.goal])
        # for goal in list(other_agent.externally_visible_goal_sets.keys()) + [self.other_agent.goal]:
            
        #     if goal not in self.models:
        #         model_file = self.saved_model_dir + goal + '.chk'
        #         checkpoint = torch.load(model_file, map_location=self.device)
        #         self.models[goal] = DQN(self.dqn_config)
        #         self.models[goal].load_state_dict(checkpoint['model_state_dict'])

        self.probability_plot.reset()
        self.total_kl = np.zeros((len(self.other_agent.externally_visible_goal_sets)), dtype=np.float32)
        self.total_kl_moving_avg = np.zeros((len(self.other_agent.externally_visible_goal_sets)), dtype=np.float32)
        self.total_kl_moving_avg_debiased = np.zeros((len(self.other_agent.externally_visible_goal_sets)), dtype=np.float32)
        self.moving_avg_updates = 0
        self.current_hypothesis = None
        self.step_number = 0


    def calculate_kl_divergence(self, model_no, state:CooperativeCraftWorldState, observed_action:int, max_value=100.0):

        state = torch.from_numpy(state.getRepresentation()).float().to(self.device).unsqueeze(0)
        q = self.trained_models[model_no].forward(state).cpu().detach().squeeze()
        probs = F.softmax(q.div(self.model_temperature), dim=0)
        print(f'KL_Div Softmax probabilities,{probs}')
        return min(probs[observed_action].pow(-1).log().item(), max_value)

 
    def softmax(self, x, temperature):

        x = np.divide(x, temperature)
        e_x = np.exp(x - np.max(x)).astype(np.float64) # Necessary to ensure that the sum of the values is close enough to 1.
        result = e_x / e_x.sum(axis=0)
        result = result / result.sum(axis=0,keepdims=1)
        return result


    def perceive(self, state:CooperativeCraftWorldState, action:int):

        for i in range(len(self.trained_models)):
            kl_div = self.calculate_kl_divergence(i, state, action)
            print(f"KL_DIV Model {self.trained_model_goal_dics[i]}, {kl_div}")
        # if self.show_graph and self.moving_avg_updates == 0:
        #     self.probability_plot.add_data_point("moving_kl_div", 0, np.zeros_like(self.total_kl_moving_avg) * (1.0 / len(self.other_agent.externally_visible_goal_sets)), False, True)

        # self.moving_avg_updates += 1
        # for i in range(len(self.other_agent.externally_visible_goal_sets)):
        #     kl_div = self.calculate_kl_divergence(state, self.other_agent.externally_visible_goal_sets[i], action)
        #     self.total_kl[i] += kl_div
        #     self.total_kl_moving_avg[i] = self.hypothesis_momentum * self.total_kl_moving_avg[i] + (1.0 - self.hypothesis_momentum) * kl_div
        #     self.total_kl_moving_avg_debiased[i] = self.total_kl_moving_avg[i] / (1.0 - self.hypothesis_momentum ** self.moving_avg_updates)

        # if self.show_graph:
        #     self.probability_plot.add_data_point("moving_kl_div", self.moving_avg_updates, self.total_kl_moving_avg_debiased, False, True)
            
        #     labels = []
        #     for id in self.other_agent.externally_visible_goal_sets:
        #         labels.append(id)

        #     self.probability_plot.update_image("moving_kl_div", labels)

        # if self.log_dir is not None and self.step_number == 0:

        #     if self.first_log_write:
        #         file_mode = 'w'
        #         self.first_log_write = False
        #     else:
        #         file_mode = 'a'  

        #     header = 'step'
        #     for id in self.other_agent.externally_visible_goal_sets:
        #         header = header + ", " + id

        #     with open(self.log_dir + 'moving_kl_div.csv', file_mode) as fd:
        #         fd.write(header + "\n")

        #     with open(self.log_dir + 'state_log.txt', file_mode) as fd:
        #         fd.write("STATE LOG:\n")

        # self.step_number += 1

        # if self.log_dir:
        #     data_row = str(self.step_number)
        #     for kl in self.total_kl_moving_avg_debiased:
        #         data_row = data_row + ", " + str(kl)

        #     with open(self.log_dir + 'moving_kl_div.csv', 'a') as fd:
        #         fd.write(data_row + "\n")

        #     with open(self.log_dir + 'state_log.txt', 'a') as fd:
        #         fd.write("\n\nSTEP: " + str(self.step_number) + "\n")

        #     state.render(log_dir=self.log_dir)


    def update_hypothesis(self):

        arg_min = np.argmin(self.total_kl_moving_avg_debiased)
        min_kl = self.total_kl_moving_avg_debiased[arg_min]

        selected_goals = []
        for i in range(0, len(self.total_kl_moving_avg_debiased)):
            if self.total_kl_moving_avg_debiased[i] <= min_kl + self.kl_tolerance:
                selected_goals.append(self.other_agent.externally_visible_goal_sets[i])

        self.current_hypothesis = selected_goals[0]
        for i in range(1, len(selected_goals)):
            self.current_hypothesis = self.current_hypothesis + "_and_" + selected_goals[i]
