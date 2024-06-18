from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable


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
        self.hx2 = None #cell_states1 = [None]
        self.cx1 = None #hidden_states2 = [None]
        self.cx2 = None #cell_states2 = [None]
        
#         self.states = []
        self.S1_prev = torch.zeros((1,args["Model"]["S1_dim"])).to("cuda:"+str(gpu_id))
        self.S2_prev = torch.zeros((1,args["Model"]["S2_dim"])).to("cuda:"+str(gpu_id))
        
        self.a1_prev = torch.zeros((1,env.action_space.n)).to("cuda:"+str(gpu_id))
        self.a2_prev = torch.zeros((1,4)).to("cuda:"+str(gpu_id))
        
        self.prev_action_logits = torch.zeros((1,6)).to("cuda:"+str(gpu_id))


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
        
        self.S1_runningmean = 0
#         self.V1_runningmean = 0
#         self.D1 = 0
        self.VD_runningmean = 0
        
#         self.restorelosses1 = []
#         self.restorelosses2 = []
        
        self.restoreds1 = []
        self.restoreds2 = []
        self.restore_labels1 = []
        self.restore_labels2 = []
        
        self.gamma1 = self.args["Training"]["initial_gamma1"]
        self.gamma2 = self.args["Training"]["initial_gamma2"]
        
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
            kld1, x_restored1, v1, a1, self.hx1, self.cx1, s1, S1 = self.model1((Variable(
                self.state.unsqueeze(0)),self.hx1, self.cx1, self.prev_action_logits.detach()))
            
#             self.S1_runningmean = self.S1_runningmean*self.gamma1 + S1.detach()*(1-self.gamma1)
            
            #r_i + g*r_i+1 + g^2*r_i+2 + ... + g^(T-i-1)*r_(T-1) + g^(T-i)*V1_T
            
            
            
#             self.V1_runningmean = self.V1_runningmean*self.gamma1 + (1-self.gamma1)*v1.detach()

            kld2, x_restored2, v2, a2, a_base, self.hx2, self.cx2, s2, S2, entropy2, log_prob2, kl_actor2 = self.model2((S1.detach(), self.hx2, self.cx2))
            
            a = 0*a1 + a_base.detach()

#             alpha1 = (abs(v1)/(abs(v1)+abs(v2)+0.001)).detach()
#             alpha2 = (abs(v2)/(abs(v1)+abs(v2)+0.001)).detach()
            
#             a = alpha1*a1 + alpha2*a_base.detach()
    
            a_throughbase = 0*a1.detach() + a_base

#             a_throughbase = alpha1*a1.detach() + alpha2*a_base
            
            self.train_episodes_run+=1
            self.train_episodes_run_2+=1
            
            self.prev_action_logits = a_base

#             restoration_loss1 = self.w_restoration * (x_restored1 - self.state.unsqueeze(0).detach()).pow(2).sum()/ self.batch_size
            
            self.restoreds1.append(x_restored1)
            self.restore_labels1.append(self.state.unsqueeze(0).detach())
        
            self.restoreds2.append(x_restored2)
            self.restore_labels2.append(S1.detach())
        
        prob1, log_prob1, action1, entropy1 = self.get_probs(a)
        self.actions.append(action1)
        
        prob_base, log_prob_base, action_base, entropy_base = self.get_probs(a_base)
        
        prob_throughbase, log_prob_throughbase, action_throughbase, entropy_throughbase = self.get_probs(a_throughbase)
        
        prob_play, log_prob_play, action_play, entropy_play = self.get_probs(a1)
        
        self.log_probs1_throughbase.append(log_prob_throughbase)
#         prob2, log_prob2, action2, entropy2 = self.get_probs(a2)
#         action2 = a2
        
        state, self.reward, self.done, self.info = self.env.step(
            action1.cpu().numpy())
        
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.reward = max(min(self.reward, 1), -1)
        
#         self.alphas1.append(alpha1)
#         self.alphas2.append(alpha2)
        self.S1_prev = torch.clone(S1.detach())
        self.S2_prev = torch.clone(S2.detach())
        self.a1_prev = torch.clone(a1.detach())
        self.a2_prev = torch.clone(a2.detach())
        
        
        
        self.entropies1.append(entropy1)
        self.values1.append(v1)
        self.log_probs1.append(log_prob1)
        self.log_probs_base.append(log_prob_base)
        self.probs1.append(prob1)
        self.probs_base.append(prob_base)
        self.probs_play.append(prob_play)
        
        self.probs_throughbase.append(prob_throughbase)
        
        self.logits1.append(a)
        self.logits_base.append(a_base)
        self.logits_play.append(a1)
        
        self.rewards1.append(self.reward)
        self.klds1.append(kld1)
        self.ss1.append(s1)
        self.Ss1.append(S1)
        
#         self.Ss1_runningmean.append(torch.clone(self.S1_runningmean).detach())
#         self.values1_runningmean.append(torch.clone(self.V1_runningmean).detach())
        
        self.entropies2.append(entropy2)
        self.values2.append(v2)
        self.log_probs2.append(log_prob2)
        self.klds2.append(kld2)
        self.klds_actor2.append(kl_actor2)
#         self.a_klds2.append(a_kld2)
        self.ss2.append(s2)
        self.Ss2.append(S2)
        
        self.states1.append(self.state)
#         self.states2.append(S1)

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
                self.cx2 = Variable(torch.zeros((1,64,5,5)), requires_grad=requires_grad).cuda()

    def detach_lstm_states(self, levels=[1,2]):
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
                self.hx2 = self.hx2.data
                self.cx2 = self.cx2.data
            
            kld1, x_restored1, v1, a1, self.hx1, self.cx1, s1, S1 = self.model1((Variable(
                self.state.unsqueeze(0)),self.hx1, self.cx1, self.prev_action_logits.detach()))
            
#             self.S1_runningmean = self.S1_runningmean*self.gamma1 + S1.detach()*(1-self.gamma1)
#             self.S1_runningmean = self.S1_runningmean*self.gamma1 + S1.detach()*(1-self.gamma1)

            kld2, x_restored2, v2, a2, a_base, self.hx2, self.cx2, s2, S2, entropy2, log_prob2, kl_actor2 = self.model2((S1, self.hx2, self.cx2))
            
            a = 0*a1 + a_base*int(not ZERO_ABASE)
            
    
#             alpha1 = abs(v1)/(abs(v1)+abs(v2)+0.001)
#             alpha2 = abs(v2)/(abs(v1)+abs(v2)+0.001)
            
#             a = alpha1*a1 + alpha2*a_base
            
            self.prev_action_logits = a_base
                
        prob = F.softmax(a, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()
        state, self.reward, self.done, self.info = self.env.step(action[0])
        self.state = torch.from_numpy(state).float()
        self.original_state = state
        self.original_state2 = S1
        self.restored_state = x_restored1
        self.restored_state2 = x_restored2
        
#         self.restored_after_lstm = self.model.Decoder2(S)
        self.S1_prev = S1.detach()
        self.S2_prev = S2.detach()
        self.a1_prev = a1.detach()
        self.a2_prev = a2.detach()
        
        self.last_S = S1
        self.last_S2 = S2
        self.last_v = v1
        self.last_v2 = v2
        self.last_s = s1
        self.last_s2 = s2
        
        self.last_a1 = a1
        self.last_a2 = a2
        self.last_abase = a_base
        self.last_a = a
        
        
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
        self.alphas2 = []
        self.alphas1 = []
        self.rewards1 = []
        self.entropies1 = []
        self.klds1 = []
        self.log_probs1_throughbase = []
        self.probs_throughbase = []
        
        self.values1_runningmean = []
        self.Ss1_runningmean = []
        
        self.ss1 = []
        self.Ss1 = []
        self.states1 = []
        
        self.values2 = []
        self.log_probs2 = []
        self.rewards2 = []
        self.entropies2 = []
        self.klds2 = []
        self.klds_actor2 = []
#         self.a_klds2 = []
        self.ss2 = []
        self.Ss2 = []
        
        self.restoreds1 = []
        self.restoreds2 = []
        self.restore_labels1 = []
        self.restore_labels2 = []
        
        return self