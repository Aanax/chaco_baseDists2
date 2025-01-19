from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy

def unload_batch_to_cpu(batch_dict, values_too=True):
    detached_dict = lambda key: [v.detach().cpu() for v in batch_dict[key]]
    copy_dict = {
        "states": detached_dict("states"),
        "rewards": batch_dict["rewards"],
        "ss1": detached_dict("ss1"),
        "actions": batch_dict["actions"],
        "Q_11s": batch_dict["Q_11s"],
        "values": detached_dict("values") if values_too else batch_dict["values"],
        "restoreds": detached_dict("restoreds"),
        "restore_labels": detached_dict("restore_labels"),
        "gs1": [],
        "memories": detached_dict("memories"),
        "prev_g1": batch_dict["prev_g1"].cpu(),
        "prev_action1": batch_dict["prev_action1"],
    }
    return copy_dict

class ReplayBuffer:
    def __init__(self, max_batches=3000, device='cuda'):
        self.buffer = []
        self.max_batches = max_batches
        self.device = device
        
    def append(self, batch):
        if len(self.buffer) >= self.max_batches:
            self.buffer.pop(0)
        self.buffer.append(unload_batch_to_cpu(batch))
           
    def sample(self, num=1):
        samples = copy.copy(np.random.choice(self.buffer, num, replace=False))
        for sample in samples:
            for key in sample.keys():
                if key not in ["prev_g1", "prev_action1", "rewards"]:
                    sample[key] = [v.to(self.device) for v in sample[key]]
            sample["prev_g1"] = sample["prev_g1"].to(self.device)
        return samples
                
def init_lstm_states(agent):
    print("Initializing LSTM states")
    if agent.gpu_id >= 0:
        device = torch.device(f'cuda:{agent.gpu_id}')
        agent.hx1 = Variable(torch.zeros((1, 32, 20, 20), device=device), requires_grad=False)
        agent.cx1 = Variable(torch.zeros((1, 32, 20, 20), device=device), requires_grad=False)
        agent.hx2 = Variable(torch.zeros((1, 64, 5, 5), device=device), requires_grad=False)
        agent.cx2 = Variable(torch.zeros((1, 64, 5, 5), device=device), requires_grad=False)

class Agent:
    def __init__(self, model1, model_target, env, args, state, gpu_id):
        self.model1 = model1
        self.model_target = model_target
        self.env = env
        self.state = state
        self.hx1 = self.cx1 = self.hx2 = self.cx2 = None
        self.prev_action_1 = torch.zeros((1, 6)).to(gpu_id)
        self.prev_g1 = torch.zeros((1, 32, 20, 20)).to(gpu_id)
        self.memory_1 = torch.zeros((1, 32, 20, 20)).to(gpu_id)
        self.prev_s1 = torch.zeros((1, 32, 20, 20)).to(gpu_id)
        self.gpu_id = gpu_id
        self.args = args
        self.eps_len = 0
        self.done = False
        self.clear_state()
        self.gamma1 = args["Training"]["initial_gamma1"]
        self.buffer = ReplayBuffer()
        self.first_batch_action = torch.zeros((1, 6)).to(gpu_id)
        self.get_probs = self.get_probs_method(args["Player"]["action_sample"])
        init_lstm_states(self)
    
    def get_probs_method(self, method_name):
        methods = {
            "max": self.get_probs_max,
            "multinomial": self.get_probs_multinomial
        }
        return methods.get(method_name, self.invalid_action_sample_method)

    @staticmethod
    def invalid_action_sample_method():
        raise ValueError("Invalid action_sample method")
    @staticmethod
    def get_probs_multinomial(logit):
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        action = prob.multinomial(1)
        log_prob = log_prob.gather(1, Variable(action))
        return prob, log_prob, action, entropy
    
    @staticmethod
    def get_probs_max(logit):
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        action = torch.argmax(prob).unsqueeze(0).unsqueeze(0)
        log_prob = log_prob.gather(1, Variable(action))
        return prob, log_prob, action, entropy

    def action_train(self):
        self.eps_len+=1
        if not self.rewards1:
            self.first_batch_action = self.prev_action_1
        self.memory_1s.append(self.memory_1)
        
        with torch.autograd.set_detect_anomaly(True):
            res1 = self.model1(Variable(self.state.unsqueeze(0)), self.prev_action_1, self.prev_s1)
            resT1 = self.model_target(Variable(self.state.unsqueeze(0)), self.prev_action_1, self.prev_s1)
            x_restored1, v1_ext, v1_int, Q11_ext, Q11_int, s1, g1 = res1
            x_restored1_T, v1_ext_T, v1_int_T, Q11_ext_T, Q11_int_T, s1_T, g1_T = resT1
            self.prev_s1 = s1.detach()
            
            
            action_probs = F.softmax(Q11_ext)
            self.action_probss.append(action_probs)
            action1 = action_probs.multinomial(1).data
            self.actions.append(action1)
            self.prev_action_1 = torch.zeros((1, 6)).to(Q11_ext.device)
            self.prev_action_1[0][action1.item()] = 1

        self.train_episodes_run += 1
        self.restoreds1.append(x_restored1)
        self.restore_labels1.append(self.state.unsqueeze(0).detach())
        self.states.append(self.state.unsqueeze(0).detach())
        self.Q_11s_ext_T.append(Q11_ext_T)
        self.Q_11s_ext.append(Q11_ext)
        self.ss1.append(s1)

        state, self.reward, self.done, self.info = self.env.step(action1.cpu().numpy())
        
        self.state = torch.from_numpy(state).float().to(self.gpu_id)
        self.reward = max(min(self.reward, 1), -1)
        self.rewards1.append(self.reward)
        
        return self

    def reset_lstm_states(self):
        if self.gpu_id >= 0:
            device = torch.device(f'cuda:{self.gpu_id}')
            self.hx1, self.cx1 = [Variable(torch.zeros((1, 32, 20, 20), device=device), requires_grad=False) for _ in range(2)]
            self.hx2, self.cx2 = [Variable(torch.zeros((1, 64, 5, 5), device=device), requires_grad=False) for _ in range(2)]

    def detach_lstm_states(self, levels=[1]):
        if self.gpu_id >= 0:
            if 1 in levels:
                self.hx1, self.cx1 = [v.detach() for v in [self.hx1, self.cx1]]
                self.train_episodes_run = 0
            if 2 in levels:
                self.hx2, self.cx2 = [v.detach() for v in [self.hx2, self.cx2]]
                self.train_episodes_run_2 = 0
                
    def action_test(self, ZERO_ABASE=False):
        self.eps_len+=1
        with torch.no_grad():
            if self.done:
                self.reset_lstm_states()
            else:
                for hx, cx in [(self.hx1, self.cx1), (self.hx2, self.cx2)]:
                    hx.data, cx.data
            x_restored1, v1_ext, v1_int, Q11_ext, Q11_int, s1, g1 = self.model1(Variable(self.state.unsqueeze(0)), self.prev_action_1, self.prev_s1)
            self.prev_s1 = s1.detach()
            
            action_probs = F.softmax(Q11_ext)
            self.action_probss.append(action_probs)
            action1 = action_probs.multinomial(1).data
            self.actions.append(action1)
            self.last_a = action_probs
            
            self.prev_action_1 = torch.zeros((1, 6)).to(Q11_ext.device)
            self.prev_action_1[0][action1.item()] = 1
            self.memory_1 = self.memory_1 * self.gamma1 + s1.detach()
            self.prev_g1 = g1
        state, self.reward, self.done, self.info = self.env.step(action1.cpu().numpy())
        self.state = torch.from_numpy(state).float().to(self.gpu_id)
        return self

    def clear_state(self):
        self.values1 = []
        self.log_probs1 = []
        self.log_probs_base = []
        self.probs1 = []
        self.logits1 = []
        self.logits_base = []
        self.logits_play = []
        self.probs_base = []
        self.probs_play = []
        self.actions = []
        self.memory_1s = []
        self.action_probss = []
        self.Q_11s_ext = []
        self.Q_11s_int = []
        self.Q_11s_ext_T = []
        self.alphas1 = []
        self.rewards1 = []
        self.entropies1 = []
        self.klds1 = []
        self.log_probs1_throughbase = []
        self.probs_throughbase = []
        self.states = []
        
        self.values1_int = []
        
        self.V_ints = []
        self.V_exts = []
        
        self.values1_runningmean = []
        self.gs1_runningmean = []
        
        self.ss1 = []
        self.gs1 = []
        self.Q_11s = []
        self.states1 = []
        self.Vs_wave = []
        self.restoreds1 = []
        self.restore_labels1 = []
