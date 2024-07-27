import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTMwithAbaseCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, num_actions):
        """
        Initialize ConvLSTM with action cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMwithAbaseCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.num_actions = num_actions
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim+self.num_actions,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
#         self.conv_action = torch.nn.Conv1d(1, 4*self.hidden_dim, self.num_actions, padding=0)
#         torch.nn.init.xavier_uniform_(self.conv_action.weight)

    def forward(self, input_tensor, cur_state, input_action):
        h_cur, c_cur = cur_state

        action_maps = input_action.squeeze().unsqueeze(1).unsqueeze(2).expand((1,8,10,10))
        
        #spaial features
        #1,32,20,20 + 1,32,20,20 + 1,6,20,20
        combined = torch.cat([input_tensor, h_cur, action_maps], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
#         af_i = af_i.repeat(1,20,20,1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

    
    
class ConvLSTMwithAAbaseCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, num_actions):
        """
        Initialize ConvLSTM with action cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMwithAAbaseCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.num_actions = num_actions
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim+self.num_actions+self.num_actions,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
#         self.conv_action = torch.nn.Conv1d(1, 4*self.hidden_dim, self.num_actions, padding=0)
#         torch.nn.init.xavier_uniform_(self.conv_action.weight)

    def forward(self, input_tensor, cur_state, input_action, input_action2):
        h_cur, c_cur = cur_state

        action_maps = input_action.squeeze().unsqueeze(1).unsqueeze(2).expand((1,6,20,20))
        action_maps2 = input_action.squeeze().unsqueeze(1).unsqueeze(2).expand((1,6,20,20))
        
        #spaial features
        #1,32,20,20 + 1,32,20,20 + 1,6,20,20
        combined = torch.cat([input_tensor, h_cur, action_maps, action_maps2], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
#         af_i = af_i.repeat(1,20,20,1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
    
# class ConvLSTMwithActionCell(nn.Module):

#     def __init__(self, input_dim, hidden_dim, kernel_size, bias, num_actions):
#         """
#         Initialize ConvLSTM with action cell.

#         Parameters
#         ----------
#         input_dim: int
#             Number of channels of input tensor.
#         hidden_dim: int
#             Number of channels of hidden state.
#         kernel_size: (int, int)
#             Size of the convolutional kernel.
#         bias: bool
#             Whether or not to add the bias.
#         """

#         super(ConvLSTMwithActionCell, self).__init__()

#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim

#         self.kernel_size = kernel_size
#         self.padding = kernel_size[0] // 2, kernel_size[1] // 2
#         self.bias = bias
#         self.num_actions = num_actions
        
#         self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
#                               out_channels=4 * self.hidden_dim,
#                               kernel_size=self.kernel_size,
#                               padding=self.padding,
#                               bias=self.bias)
        
#         self.conv_action = torch.nn.Conv1d(1, 4*self.hidden_dim, self.num_actions, padding=0)
#         torch.nn.init.xavier_uniform_(self.conv_action.weight)

#     def forward(self, input_tensor, cur_state, input_action):
#         h_cur, c_cur = cur_state

#         #spaial features
#         combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
#         combined_conv = self.conv(combined)
#         cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
#         #action features (result shape 1,32,1)
# #         print("input_action.shape ", input_action.shape, flush=True)

#         action_features = self.conv_action(input_action)
# #         print("action_features.shape ", action_features.shape, flush=True)
#         af_i, af_f, af_o, af_g = torch.split(action_features.squeeze(), self.hidden_dim, dim=0)
        
# #         print("af_i.shape ", af_i.shape, flush=True)
# #         print("cc_i.shape ", cc_i.shape, flush=True)
#         print("af_i.max() ",af_i.max())
#         print("af_f.max() ",af_f.max())
#         print("af_o.max() ",af_o.max())
#         print("af_g.max() ",af_g.max())
#         print("af_i.min() ",af_i.min())
#         print("af_f.min() ",af_f.min())
#         print("af_o.min() ",af_o.min())
#         print("af_g.min() ",af_g.min())
        
#         print("cc_i.max() ",cc_i.max())
#         print("cc_f.max() ",cc_f.max())
#         print("cc_o.max() ",cc_o.max())
#         print("cc_g.max() ",cc_g.max())
#         print("cc_i.min() ",cc_i.min())
#         print("cc_f.min() ",cc_f.min())
#         print("cc_o.min() ",cc_o.min())
#         print("cc_g.min() ",cc_g.min())
        
#         af_i = af_i.repeat(1,20,20,1)
#         af_i = torch.moveaxis(af_i,3,1)
        
#         af_f = af_f.repeat(1,20,20,1)
#         af_f = torch.moveaxis(af_f,3,1)
        
#         af_o = af_o.repeat(1,20,20,1)
#         af_o = torch.moveaxis(af_o,3,1)
        
#         af_g = af_g.repeat(1,20,20,1)
#         af_g = torch.moveaxis(af_g,3,1)
        
        
#         i = torch.sigmoid(cc_i+af_i)
#         f = torch.sigmoid(cc_f+af_f)
#         o = torch.sigmoid(cc_o+af_o)
#         g = torch.tanh(cc_g+af_g)

#         c_next = f * c_cur + i * g
#         h_next = o * torch.tanh(c_next)

#         return h_next, c_next

#     def init_hidden(self, batch_size, image_size):
#         height, width = image_size
#         return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
#                 torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param