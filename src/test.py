from setproctitle import setproctitle as ptitle
import torch
# from agents import Agent
from environment import atari_env
from utils import setup_logger
from torch.autograd import Variable
import os
import logging
import time
from agents import Agent
import imageio
import copy


from models import Level1
# from models import Level2

def test(args, shared_model, env_conf, counter, num):
    ptitle('Test Agent')
    gpu_id = args["Training"]["gpu_ids"][-1]
    log = {}
    architecture_type = args["Training"]["log_dir"].replace("/","_")
    #"_idk_"
    
    logname = args["Training"]["log_dir"]+args["Training"]["env"]+"_"+args["Training"]["log_name"]+".log"
    if os.path.exists(logname):
        time_str = time.strftime("%Y-%m-%d_%H:%M:%S")
        logname = logname[:-4]+"_"+time_str+"_"+str(num)+".log"
    
    setup_logger('{}_log'.format(args["Training"]["env"]),logname)
#     setup_logger('{}_log'.format(args.env), r'{0}{1}{2}{3}_log'.format(
#         args.log_dir, args.env, architecture_type, time.strftime("%Y-%m-%d_%H:%M")))
    
    log['{}_log'.format(args["Training"]["env"])] = logging.getLogger('{}_log'.format(
        args["Training"]["env"]))
#     d_args = vars(args)
    for k in args.keys():
        log['{}_log'.format(args["Training"]["env"])].info('{0}: {1}'.format(k, args[k]))

    torch.manual_seed(args["Training"]["seed"])
    if gpu_id >= 0:
        torch.cuda.manual_seed(args["Training"]["seed"])
    env = atari_env(args["Training"]["env"], env_conf, args)
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    
    #     player.gpu_id = gpu_id
    model_params_dict = args["Model"]
#     AgentClass = eval(args["Training"]["agent)
    _model1 = Level1(args, env.observation_space.shape[0],
                           env.action_space, device = "cuda:"+str(gpu_id))
#     _model2 = Level2(args, env.observation_space.shape[0],
#                            env.action_space, device = "cuda:"+str(gpu_id))
    
    player = Agent(_model1, None, env, args, None, gpu_id) #Agent(None, env, args, None)
    
    
    

    #AgentClass(player.env.observation_space.shape[0],
                    #       player.env.action_space, int(args.t_hidden))

    player.state = player.env.reset()
    player.eps_len += 2
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model1 = player.model1.cuda()
#             player.model2 = player.model2.cuda()
            player.state = player.state.cuda()
    flag = True
    max_score = -20
    rgb_frames = []
    prev_video_at = 0
    while True:
        if flag:
#             print("R out weight MEAN", torch.mean(player.model.R_out.weight.data))
#             print("s conv4 weight MEAN", torch.mean(player.model.conv4.weight.data))
#             print("R out weight MAX", torch.max(player.model.R_out.weight.data))
#             print("s conv4 weight MAX", torch.max(player.model.conv4.weight.data))
# #             print("A lstm bias weight MAX", torch.max(player.model.A_lstm.bias_ih))
#             print("T lstm bias weight MAX", torch.max(player.model.TA_lstm.bias_ih))
            
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model1.load_state_dict(shared_model[0].state_dict())
#                     player.model2.load_state_dict(shared_model[1].state_dict())
            else:
                player.model1.load_state_dict(shared_model[0].state_dict())
#                 player.model2.load_state_dict(shared_model[1].state_dict())
            player.model1.eval()
#             player.model2.eval()
            flag = False
        

        player.action_test()
        rgb_frames.append(player.env.render(mode = 'rgb_array'))
        reward_sum += player.reward

        if player.done and not player.info:
            state = player.env.reset()
            player.eps_len += 2
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
        elif player.info:
            flag = True
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_log'.format(args["Training"]["env"])].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}, episodes_total {4}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean, counter.value))
            
#             try:
            if (counter.value-prev_video_at)>100000:
                saveanimation(rgb_frames, address="./runs_gifs/run"+str(counter.value)+"_"+str(architecture_type)+"_rew"+str(reward_sum)+".gif")
                prev_video_at = copy.copy(counter.value) 
            rgb_frames=[]
#             except Exception as e:
#                 print(e)
            
            if args["Training"]["save_max"] and reward_sum >= max_score:
                max_score = reward_sum
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = shared_model.state_dict()
                        torch.save(state_to_save, '{0}{1}{2}_{3}.dat'.format(
                            args["Training"]["save_model_dir"], args["Training"]["env"],architecture_type,reward_sum))
                else:
                    state_to_save = shared_model.state_dict()
                    torch.save(state_to_save, '{0}{1}{2}_{3}.dat'.format(
                        args["Training"]["save_model_dir"], args["Training"]["env"],architecture_type,reward_sum))

            reward_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            player.eps_len += 2
            time.sleep(10)
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

def saveanimation(frames, address="./movie_base.gif"):
    """ 
    This method ,given the frames of images make the gif and save it in the folder
    
    params:
        frames:method takes in the array or np.array of images
        address:(optional)given the address/location saves the gif on that location
                otherwise save it to default address './movie.gif'
    
    return :
        none
    """
    imageio.mimsave(address, frames, fps=5)