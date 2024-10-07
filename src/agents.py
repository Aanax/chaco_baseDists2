from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

## STOP rewrite agetnt with respect to hidden_states. 
## add Star features or switches or separate class


def init_lstm_states(player):
    print("lstm states inited!!!!!")
    requires_grad = False
    if player.gpu_id >= 0:
        with torch.cuda.device(player.gpu_id):
            
            num_lstm_layers=1
            #TODO switch to none or zero if no such layer.
            player.hx1 = Variable(torch.zeros((1,32,20,20)), requires_grad=requires_grad).cuda()
#            
            player.cx1 = Variable(torch.zeros((1,32,20,20)), requires_grad=requires_grad).cuda()
                
            
            player.hx2 = Variable(torch.zeros((1,64,5,5)), requires_grad=requires_grad).cuda()
            player.cx2 = Variable(torch.zeros((1,64,5,5)), requires_grad=requires_grad).cuda()
        
    return player

class Agent(object):
    def __init__(self, model1, model2, env, args, state, gpu_id):
        self.model1 = model1
        self.model2 = model2
        
        self.env = env
        self.state = state
#         self.hx = None
#         self.cx = None
        self.hx1 = None #hidden_states1 = [None]
#         self.hx2 = None #cell_states1 = [None]
        self.cx1 = None #hidden_states2 = [None]
#         self.cx2 = None #cell_states2 = [None]
        
#         self.states = []
#         self.g1_prev = torch.zeros((1,args["Model"]["g1_dim"])).to("cuda:"+str(gpu_id))
#         self.g2_prev = torch.zeros((1,args["Model"]["g2_dim"])).to("cuda:"+str(gpu_id))
        
        self.prev_action_1 = torch.zeros((1,6)).to("cuda:"+str(gpu_id))
#         self.prev_action_2 = torch.zeros((1,6)).to("cuda:"+str(gpu_id))
        
        self.prev_g1 = torch.zeros((1,32,20,20)).to("cuda:"+str(gpu_id))
#         self.prev_g2 = torch.zeros((1,64,5,5)).to("cuda:"+str(gpu_id))
        
        self.memory_1 = torch.zeros((1,32,20,20)).to("cuda:"+str(gpu_id))
#         self.memory_2 = torch.zeros((1,64,5,5)).to("cuda:"+str(gpu_id))

        self.train_episodes_run_2 =0
        self.train_episodes_run =0
        self.eps_len = 0
        self.args = args
        self.rank = 0
        self.train_episodes_run = 0
        self.clear_actions()
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = gpu_id
        self.batch_size = self.args["Training"]["num_steps"]
        self.w_restoration = self.args["Training"]["w_restoration"]
        self.w_restoration_future = self.args["Training"]["w_restoration_future"]
        
        self.g1_runningmean = 0
#         self.V1_runningmean = 0
#         self.D1 = 0
        self.VD_runningmean = 0
        
#         self.restorelosses1 = []
#         self.restorelosses2 = []
        
        self.restoreds1 = []
#         self.restoreds2 = []
        self.restore_labels1 = []
#         self.restore_labels2 = []
        
        self.gamma1 = self.args["Training"]["initial_gamma1"]
#         self.gamma2 = self.args["Training"]["initial_gamma2"]
        
        self.states = []
        
        if self.args["Player"]["action_sample"]=="max":
            self.get_probs = self.get_probs_max
        elif self.args["Player"]["action_sample"]=="multinomial":
            self.get_probs = self.get_probs_multinomial
        else:
            print("NO such action_sample method ", self.args["Player"]["action_sample"])
            #TODO raise error instead
        
        model_params_dict = args["Model"]
        init_lstm_states(self)
    
    @staticmethod
    def get_probs_multinomial(logit):
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))
        return prob, log_prob, action, entropy
    
    @staticmethod
    def get_probs_max(logit):
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        action = torch.argmax(prob).data.unsqueeze(0).unsqueeze(0)
        log_prob = log_prob.gather(1, Variable(action))
        return prob, log_prob, action, entropy
    
    def action_train(self):
        
        with torch.autograd.set_detect_anomaly(True):
            #decoded,v,Q11, s, g
            x_restored1, v1, Q_11, s1, g1 = self.model1(Variable(
                self.state.unsqueeze(0)), self.prev_action_1, self.prev_g1, self.memory_1)
            
#             #kl, v, a_21, a_22, Q_22, hx,cx,s,S
#             x_restored2, v2, a_22, Q_22, s2,g2, V_wave = self.model2(self.prev_g1.detach(), self.prev_action_2, self.prev_g2, self.memory_2)
            
#             self.Q_21_prev = Q_21
#             self.a_22_prev = a_22.to(Q_22.device)
            
#             self.Vs_wave.append(V_wave)
            action_probs = F.softmax(Q_11) #+Q_22)
            action1 = action_probs.multinomial(1).data
            self.actions.append(action1)
            
            #TODO
            self.prev_action_1 = torch.zeros((1,6)).to(Q_11.device)
            self.prev_action_1[0][action1.item()] = 1
            self.prev_action_1 = self.prev_action_1.to(Q_11.device)
            
#             self.prev_action_2 = (Q_22.detach()>=v2.detach()).type(torch.float).to(Q_11.device) #Q_11.detach()
            
#             self.prev_action_2 = Q_22.detach()
            
            self.memory_1 = self.memory_1*self.gamma1 + s1.detach()
#             self.memory_2 = self.memory_2*self.gamma2 + s2.detach()
            
            
            self.prev_g1 = g1.detach()
#             self.prev_g2 = g2.detach()
           
            
            self.train_episodes_run+=1
#             self.train_episodes_run_2+=1
            self.restoreds1.append(x_restored1)
#             self.restoreds2.append(x_restored2)
            self.restore_labels1.append(self.state.unsqueeze(0).detach())
#             self.restore_labels2.append(g1.detach())
        
        state, self.reward, self.done, self.info = self.env.step(
            action1.cpu().numpy())
        
        self.Q_11s.append(Q_11)
#         self.Q_21s.append(Q_21)
#         self.Q_22s.append(Q_22)
#         self.a_22s.append(a_22)
        
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.reward = max(min(self.reward, 1), -1)
        
#         self.alphas1.append(alpha1)
#         self.alphas2.append(alpha2)
        
        
        
        self.values1.append(v1)
        
        
        self.rewards1.append(self.reward)
        self.ss1.append(s1)
        self.gs1.append(g1)
        
#         self.gs1_runningmean.append(torch.clone(self.g1_runningmean).detach())
#         self.values1_runningmean.append(torch.clone(self.V1_runningmean).detach())
        
#         self.entropies2.append(entropy2)
#         self.values2.append(v2)
#         self.log_probs2.append(log_prob2)
#         self.klds_actor2.append(kl_actor2)
#         self.a_klds2.append(a_kld2)
#         self.ss2.append(s2)
#         self.gs2.append(g2)
        
        
        self.states1.append(self.state)
#         self.states2.append(g1)

        return self

    def reset_lstm_states(self):
        requires_grad = False
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):

                num_lstm_layers=1
                #TODO switch to none or zero if no such layer.
                self.hx1 = Variable(torch.zeros((1,32,20,20)), requires_grad=requires_grad).cuda()
    #            
                self.cx1 = Variable(torch.zeros((1,32,20,20)), requires_grad=requires_grad).cuda()


                self.hx2 = Variable(torch.zeros((1,64,5,5)), requires_grad=requires_grad).cuda()
#                 self.cx2 = Variable(torch.zeros((1,64,5,5)), requires_grad=requires_grad).cuda()

    def detach_lstm_states(self, levels=[1]):#2
        requires_grad = False
        
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):

                num_lstm_layers=1
                if 1 in levels:
                    #TODO switch to none or zero if no such layer.
                    self.hx1 = self.hx1.detach() #Variable(self.hx1.data, requires_grad=requires_grad).cuda()
        #            
                    self.cx1 = self.cx1.detach()#Variable(torch.zeros(num_lstm_layers, (32,40,40)), requires_grad=requires_grad).cuda()
                    self.train_episodes_run = 0
                if 2 in levels:
                    self.hx2 = self.hx2.detach()#Variable(torch.zeros(num_lstm_layers, (64,6,6)), requires_grad=requires_grad).cuda()
                    self.cx2 = self.cx2.detach()#Variable(torch.zeros(num_lstm_layers, (64,6,6)), requires_grad=requires_grad).cuda()
                    self.train_episodes_run_2 = 0
                #TODO
    def action_test(self, ZERO_ABASE=False):
        with torch.no_grad():
            if self.done:
                self.reset_lstm_states()
            else:
#                 for i in range(len(self.hidden_states1)):
#                     self.hidden_states1[i] = self.hidden_states1[i].data
#                 for i in range(len(self.cell_states1)):
#                     self.cell_states1[i] = self.cell_states1[i].data
                    
#                 for i in range(len(self.hidden_states2)):
#                     self.hidden_states2[i] = self.hidden_states2[i].data
#                 for i in range(len(self.cell_states2)):
#                     self.cell_states2[i] = self.cell_states2[i].data
                self.hx1 = self.hx1.data
                self.cx1 = self.cx1.data
#                 self.hx2 = self.hx2.data
#                 self.cx2 = self.cx2.data
                
             #decoded,v,Q11, s, g
            x_restored1, v1, Q_11, s1, g1 = self.model1(Variable(
                self.state.unsqueeze(0)), self.prev_action_1, self.prev_g1, self.memory_1)
            
            #kl, v, a_21, a_22, Q_22, hx,cx,s,S
#             x_restored2, v2, a_22, Q_22, s2,g2, V_wave = self.model2(self.prev_g1.detach(), self.prev_action_2, self.prev_g2, self.memory_2)
            
            self.prev_Q11 = Q_11
#             self.prev_Q22 = Q_22
#             self.a_22_prev = a_22
            self.prev_state = s1

    
            action_probs = F.softmax(Q_11)#+Q_22)
            action1 = action_probs.multinomial(1).data #?
            self.last_a = action_probs
            
            #TODO
            self.prev_action_1 = torch.zeros((1,6)).to(Q_11.device)
            self.prev_action_1[0][action1.item()] = 1
            self.prev_action_1 = self.prev_action_1.to(Q_11.device)
            
#             self.prev_action_2 = (Q_22.detach()>=v2.detach()).type(torch.float).to(Q_11.device) #Q_11.detach()
            
            
            self.memory_1 = self.memory_1*self.gamma1 + s1.detach()
#             self.memory_2 = self.memory_2*self.gamma2 + s2.detach()
            
            
            self.prev_g1 = g1.detach()
#             self.prev_g2 = g2.detach()
        
        state, self.reward, self.done, self.info = self.env.step(action1.cpu().numpy())
        self.state = torch.from_numpy(state).float()
        self.original_state = state
#         self.original_state2 = g1
        self.restored_state = x_restored1
#         self.restored_state2 = x_restored2
        
#         self.restored_after_lstm = self.model.Decoder2(S)
#         self.g1_prev = g1.detach()
#         self.g2_prev = g2.detach()
#         self.a1_prev = a1.detach()
#         self.a2_prev = a2.detach()
        
#         self.last_g2 = g2
        self.last_g1 = g1
        self.last_v = v1
#         self.last_v2 = v2
        self.last_s = s1
#         self.last_s2 = s2
        
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        return self

    def clear_actions(self):
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
#         self.alphas2 = []
        self.alphas1 = []
        self.rewards1 = []
        self.entropies1 = []
        self.klds1 = []
        self.log_probs1_throughbase = []
        self.probs_throughbase = []
        
        self.values1_runningmean = []
        self.gs1_runningmean = []
        
        self.ss1 = []
        self.gs1 = []
        self.Q_11s = []
#         self.Q_21s = []
#         self.Q_22s = []
#         self.a_22s = []
        self.states1 = []
        
        self.Vs_wave = []
        
#         self.values2 = []
#         self.log_probs2 = []
#         self.rewards2 = []
#         self.entropies2 = []
#         self.klds2 = []
#         self.klds_actor2 = []
#         self.a_klds2 = []
#         self.ss2 = []
#         self.gs2 = []
        
        self.restoreds1 = []
#         self.restoreds2 = []
        self.restore_labels1 = []
#         self.restore_labels2 = []
        
        return self