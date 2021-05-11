import numpy as np


def expect(expected: np.ndarray, result: np.ndarray, rtol=1.e-4, **kwargs):
    assert expected.dtype == result.dtype
    assert expected.shape == result.shape
    assert np.allclose(expected, result, rtol=rtol, **kwargs)


class GRU_Helper():
    def __init__(self, **params):
        # GRU Input Names
        X = str('X')
        W = str('W')
        R = str('R')
        B = str('B')
        H_0 = str('initial_h')
        LBR = str('linear_before_reset')
        LAYOUT = str('layout')
        number_of_gates = 3

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, "Missing Required Input: {0}".format(i)

        self.num_directions = params[W].shape[0]

        if self.num_directions == 1:
            for k in params.keys():
                if k != X:
                    params[k] = np.squeeze(params[k], axis=0)

            hidden_size = params[R].shape[-1]
            batch_size = params[X].shape[1]

            layout = params[LAYOUT] if LAYOUT in params else 0
            x = params[X]
            x = x if layout == 0 else np.swapaxes(x, 0, 1)
            b = params[B] if B in params else np.zeros(
                2 * number_of_gates * hidden_size)
            h_0 = params[H_0] if H_0 in params else np.zeros(
                (batch_size, hidden_size))
            lbr = params[LBR] if LBR in params else 0

            self.X = x
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.H_0 = h_0
            self.LBR = lbr
            self.LAYOUT = layout

        else:
            raise NotImplementedError()

    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def g(self, x):
        return np.tanh(x)

    def step(self):
        seq_length = self.X.shape[0]
        hidden_size = self.H_0.shape[-1]
        batch_size = self.X.shape[1]

        Y = np.empty([seq_length, self.num_directions, batch_size, hidden_size])
        h_list = []

        [w_z, w_r, w_h] = np.split(self.W, 3)
        [r_z, r_r, r_h] = np.split(self.R, 3)
        [w_bz, w_br, w_bh, r_bz, r_br, r_bh] = np.split(self.B, 6)
        gates_w = np.transpose(np.concatenate((w_z, w_r)))
        gates_r = np.transpose(np.concatenate((r_z, r_r)))
        gates_b = np.add(
            np.concatenate((w_bz, w_br)),
            np.concatenate((r_bz, r_br)))

        H_t = self.H_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            gates = np.dot(x, gates_w) + np.dot(H_t, gates_r) + gates_b
            z, r = np.split(gates, 2, -1)
            z = self.f(z)
            r = self.f(r)
            h_default = self.g(np.dot(x, np.transpose(
                w_h)) + np.dot(r * H_t, np.transpose(r_h)) + w_bh + r_bh)
            h_linear = self.g(
                np.dot(x, np.transpose(w_h)) + r *
                (np.dot(H_t, np.transpose(r_h)) + r_bh) + w_bh)
            h = h_linear if self.LBR else h_default
            H = (1 - z) * h + z * H_t
            h_list.append(H)
            H_t = H

        concatenated = np.concatenate(h_list)
        if self.num_directions == 1:
            Y[:, 0, :, :] = concatenated

        if self.LAYOUT == 0:
            Y_h = Y[-1]
        else:
            Y = np.transpose(Y, [2, 0, 1, 3])
            Y_h = Y[:, :, -1, :]

        return Y, Y_h


def dropout_reference(X, drop_probability=0.5, seed=0, training_mode=False,
                      return_mask=False):
    if drop_probability == 0 or training_mode is False:
        if return_mask is True:
            return X, np.ones(X.shape, dtype=bool)
        else:
            return X

    np.random.seed(seed)
    mask = np.random.uniform(0, 1.0, X.shape) >= drop_probability
    scale = (1 / (1 - drop_probability))
    if return_mask is True:
        return mask * X * scale, mask.astype(bool)
    else:
        return mask * X * scale


class LSTM_Helper():
    def __init__(self, **params):
        # LSTM Input Names
        X = str('X')
        W = str('W')
        R = str('R')
        B = str('B')
        H_0 = str('initial_h')
        C_0 = str('initial_c')
        P = str('P')
        LAYOUT = str('layout')
        number_of_gates = 4
        number_of_peepholes = 3

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, "Missing Required Input: {0}".format(i)

        self.num_directions = params[W].shape[0]

        if self.num_directions == 1:
            for k in params.keys():
                if k != X:
                    params[k] = np.squeeze(params[k], axis=0)

            hidden_size = params[R].shape[-1]
            batch_size = params[X].shape[1]

            layout = params[LAYOUT] if LAYOUT in params else 0
            x = params[X]
            x = x if layout == 0 else np.swapaxes(x, 0, 1)
            b = params[B] if B in params else np.zeros(
                2 * number_of_gates * hidden_size, dtype=np.float32)
            p = params[P] if P in params else np.zeros(
                number_of_peepholes * hidden_size, dtype=np.float32)
            h_0 = params[H_0] if H_0 in params else np.zeros(
                (batch_size, hidden_size), dtype=np.float32)
            c_0 = params[C_0] if C_0 in params else np.zeros(
                (batch_size, hidden_size), dtype=np.float32)

            self.X = x
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.P = p
            self.H_0 = h_0
            self.C_0 = c_0
            self.LAYOUT = layout

        else:
            raise NotImplementedError()

    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def g(self, x):
        return np.tanh(x)

    def h(self, x):
        return np.tanh(x)

    def step(self):
        seq_length = self.X.shape[0]
        hidden_size = self.H_0.shape[-1]
        batch_size = self.X.shape[1]

        Y = np.empty([seq_length, self.num_directions, batch_size, hidden_size])
        h_list = []

        [p_i, p_o, p_f] = np.split(self.P, 3)
        H_t = self.H_0
        C_t = self.C_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            gates = np.dot(x, np.transpose(self.W)) + \
                np.dot(H_t, np.transpose(self.R)) + np.add(*np.split(self.B, 2))
            i, o, f, c = np.split(gates, 4, -1)
            i = self.f(i + p_i * C_t)
            f = self.f(f + p_f * C_t)
            c = self.g(c)
            C = f * C_t + i * c
            o = self.f(o + p_o * C)
            H = o * self.h(C)
            h_list.append(H)
            H_t = H
            C_t = C

        concatenated = np.concatenate(h_list)
        if self.num_directions == 1:
            Y[:, 0, :, :] = concatenated

        if self.LAYOUT == 0:
            Y_h = Y[-1]
        else:
            Y = np.transpose(Y, [2, 0, 1, 3])
            Y_h = Y[:, :, -1, :]

        return Y, Y_h


def one_hot_reference(indices, depth, axis=-1, dtype=np.float32):
    values = np.asarray(indices)
    rank = len(values.shape)
    depth_range = np.arange(depth)
    if axis < 0:
        axis += (rank + 1)
    ls = values.shape[0:axis]
    rs = values.shape[axis:rank]
    targets = np.reshape(
        depth_range, (1,) * len(ls) + depth_range.shape + (1,) * len(rs))
    values = np.reshape(np.mod(values, depth), ls + (1,) + rs)
    return np.asarray(targets == values, dtype=dtype)


def negative_log_likelihood_loss_reference(
        input, target, weight=None, reduction='mean', ignore_index=None):
    input_shape = input.shape
    if len(input_shape) == 1:
        raise RuntimeError("Unsupported shape")

    target_shape = target.shape
    N = input_shape[0]
    C = input_shape[1]

    # initialize the positional weights when required
    gather_weight = None
    if weight is not None:
        # setting mode='clip' to deal with ignore_index > C or < 0 cases.
        # when the target value is > C or < 0, it doesn't matter which value we are
        # taking in gather_weight, since it will be set to 0 in the following if-block
        # use np.int32 to make it compatible with x86 machines
        gather_weight = np.take(weight, np.array(
            target, dtype=np.int32), mode='clip')
        # set `ignore_index`'s loss weight to 0.
        # The loss tensor will be multiplied by this weight tensor,
        # so `ingore_index`'s loss value will be eliminated.
        if ignore_index is not None:
            gather_weight = np.where(
                target == ignore_index, 0, gather_weight).astype(
                dtype=np.float32)
    elif ignore_index is not None:
        gather_weight = np.where(
            target == ignore_index, 0, 1).astype(
            dtype=np.float32)

    # if input is 4-d and above, make it 3-d
    if len(input_shape) != 3:
        input = input.reshape((N, C, -1))
        target = target.reshape((N, -1))

    # Get a dimension from the reshaped input.
    # If the original input shape is [N, C, H, W],
    # the D here should be H * W because we reshape
    # [N, C, H, W] to [N, C, H * W].
    D = input.shape[2]
    neg_gather_element_input = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        for d in range(D):
            if target[i][d] != ignore_index:
                neg_gather_element_input[i][d] = -input[i][target[i][d]][d]

    loss = neg_gather_element_input

    # if the input was 4-d or above reshape to the right shape
    if len(input_shape) != 3:
        loss = loss.reshape(target_shape)

    # apply the weights when required
    if gather_weight is not None:
        loss = gather_weight * loss
        if reduction == 'mean':
            loss = loss.sum() / gather_weight.sum()
            return loss

    if reduction == 'mean':
        loss = np.mean(loss)
    elif reduction == 'sum':
        loss = np.sum(loss)
    return loss
