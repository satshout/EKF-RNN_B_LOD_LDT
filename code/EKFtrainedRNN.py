### EKF-trained RNN module for IGRF-14 SV candidate model
# Author: Sho Sato
# Date: 2024-09-16
# Based on: 
#   Puskorius & Feldcamp (1994) – doi: <https://doi.org/10.1109/72.279191> 
#   Elman (1990) – doi: <https://doi.org/10.1016/0364-0213(90)90002-E>
#   Kalman (1960) – doi: <https://doi.org/10.1115/1.3662552>

import numpy as np # Numpy 1.23.2

### Recurrent Neural Network (RNN) ------------------------------------------------
""" Elman Network 
Elman (1990) - doi:10.1016/0364-0213(90)90002-E.

# Din : Input vector size 
# Drec: Hidden unit size
# Dout: Output vector size

RNN cell:
h[t] = tanh(Win @ x[t] + Wrec @ h[t-1] + brec)
    input:
        x[t]  : (Din  x 1)
        h[t-1]: (Drec x 1)

    trainable parameters (weights and biases):
        Win : (Drec x Din)
        Wrec: (Drec x Drec)
        brec: (Drec x 1)
    
    output:
        h[t]: (Drec x 1)

Output layer (Affine / Dense):
z[t] = Wout @ h[t] + bout
    input:
        h[t]: (Drec x 1)

    trainable parameters (weights and biases):
        Wout: (Dout x Drec)
        bout: (Dout x 1)
    
    output:
        z[t]: (Dout x 1)
"""

# Activation functions
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

def id(x):
    return x

def id_prime(x):
    return 1

# RNN implementation
def RNNforward(xt, h_prev, Win, Wrec, b_rec, Wout, b_out, 
               h_actF=tanh, z_actF=id):
    """
    Compute forward propagation of a recurrent neural network (Elman, 1990)
    input:
        xt    : input vector of shape (Din, 1)           @ input nodes
        h_prev: previous hidden state of shape (Drec, 1) @ hidden nodes

        weights:
        Win   : input-to-hidden weights of shape (Drec, Din)    @ input layer
        Wrec  : hidden-to-hidden weights of shape (Drec, Drec)  @ hidden layer (RNN cell)
        b_rec : biases of shape (Drec, 1)                       @ hidden layer (RNN cell)

        Wout  : hidden-to-output weights of shape (Dout, Drec)  @ output layer (Affine)
        b_out : biases of shape (Dout, 1)                       @ output layer (Affine)

        activation functions:
        h_actF: hidden activation function (default: tanh)
        z_actF: output activation function (default: identity)
    returns:
        the next hidden state and the output
        ht, zt
    """

    # hidden layer (RNN cell)
    ht = h_actF(Win @ xt + Wrec @ h_prev + b_rec)

    # output layer (Affine)
    zt = z_actF(Wout @ ht + b_out)

    return ht, zt

def RNNforward_tlm(xt,  h_prev,  Win,  Wrec,  b_rec,  Wout,  b_out, 
                   dx, dh_prev, dWin, dWrec, db_rec, dWout, db_out, 
                   h_actF=tanh, z_actF=id, h_actF_prime=tanh_prime, z_actF_prime=id_prime):
    """
    Compute the tangent linear model of the RNNforward function
    input:
        xt    : input vector of shape (Din, p)           @ input nodes
        h_prev: previous hidden state of shape (Drec, 1) @ hidden nodes

        weights:
        Win   : input-to-hidden weights of shape (Drec, Din)    @ input layer
        Wrec  : hidden-to-hidden weights of shape (drec, drec)  @ hidden layer (RNN cell)
        b_rec : biases of shape (Drec, 1)                       @ hidden layer (RNN cell)

        Wout  : hidden-to-output weights of shape (Dout, Drec)  @ output layer (Affine)
        b_out : biases of shape (Dout, 1)                       @ output layer (Affine)

        perturbations:
        dx, dh_prev, dWin, dWrec, db_rec, dWout, db_out

        activation functions:
        h_actF, h_actF_prime: hidden activation function and its deriverate (default: tanh)
        z_actF, z_actF_prime: output activation function and its deriverate (default: identity)

    returns:
        the response characteristics of the model
        dz
    """

    # hidden layer (RNN cell)
    w_rec = Win @ xt + Wrec @ h_prev + b_rec
    dw_rec = (dWin @ xt + Win @ dx) + (dWrec @ h_prev + Wrec @ dh_prev) + db_rec
    ht = h_actF(w_rec)
    dh = h_actF_prime(w_rec) * dw_rec

    # output layer (Affine)
    dz = z_actF_prime(Wout @ ht + b_out) * (dWout @ ht + Wout @ dh + db_out)

    return dz

# Convert matrices to state space for Kalman filter
def matrices_to_statespace(matrices):
    """
    Convert the RNNstep matrices to a state space model
    input:
        matrices: list of matrices
        e.g. [xt, h_prev, Win, Wrec, b_rec, Wout, b_out]
    returns:
        the state space vector and the list of matrix.shape
        X, shapes
    """
    
    concat = []
    shapes = []
    for matrix in matrices:
        concat.append(matrix.flatten())
        shapes.append(matrix.shape)
    X = np.concatenate(concat)
    return X, shapes

def statespace_to_matrices(X, shapes):
    """
    Convert the state space model to RNNstep matrices
    input:
        X: state space vector
        shapes: list of matrix.shape
    returns:
        the matrices
        matrices
    """
    
    matrices = []
    i = 0
    for shape in shapes:
        matrices.append(X[i:i+np.prod(shape)].reshape(shape))
        i += np.prod(shape)
    return matrices


### extended Kalman filter (EKF) ------------------------------------------------
def check_symmetric(P, name='P', epsilon=1e-8):
    """
    A covariance matrix P must be 
        > 1. symmetric
          2. Pierson correlation coefficient |rho| <= 1
    """
    sym_check = np.linalg.norm(P - P.T)
    if sym_check > epsilon:
        print(f'Warning! {name} is not symmetric: {sym_check:.4f}')
        return 0b01
    return 0b00

def check_Rhomatrix(P, name='P', epsilon=1e-8):
    """
    A covariance matrix P must be 
          1. symmetric
        > 2. Pierson correlation coefficient |rho| <= 1
    
        Warning! this function is heavy and slow
    """
    diag_B = 1 / np.sqrt(np.diag(P))
    B = np.diag(diag_B)
    Rho = B @ P @ B - np.eye(P.shape[0]) # extract cross-correlation matrix
    
    rho_min, rho_max = Rho.min(), Rho.max()
    if (rho_min < -1 - epsilon) or (rho_max > 1 + epsilon):
        print(f'Warning! {name} has a correlation coefficient out of range: [{rho_min:.2f} < rho < {rho_max:.2f}]')
        return 0b10
    
    return 0b00

# Observation operator
def Hobs(xt, h_prev, 
         wt, Wshapes):
    Win, Wrec, b_rec, Wout, b_out = statespace_to_matrices(wt, Wshapes)
    
    ht, zt = RNNforward(xt, h_prev, Win, Wrec, b_rec, Wout, b_out)

    return zt

# extended Kalman filter implementation (Kalman, 1960)
def KalmanFilter(wa_prev, Pa_prev, Wshapes,
                 xt, h_prev, H,
                 yo, Ro):
    """
    Kalman filtering designed for RNN training (Puskorius & Feldcamp, 1994)
    input:
        wa_prev: previous state vector of weights and biases flattened into a 1D vector
        Pa_prev: error covariance matrix of the previous estimation of weights and biases
        Wshapes: list of matrix shapes for the state vector
        xt     : current RNN input vector of shape (Din, 1)
        h_prev : previous RNN hidden layer of shape (Drec, 1)
        H      : tangent linear model of observation operator at the time of observation
        yo     : current observation vector (i.e. training data) of shape (Dout, 1)
        Ro     : observation error covariance (i.e. reliability of training data) matrix of shape (Dout, Dout)
    returns:
        the forecast state vector, the analysis state vector, and the analysis covariance matrix
        Pf, wa, Pa
    """
    Pa_prev = (Pa_prev + Pa_prev.T) / 2 # Ensure P is symmetric

    # Forecast
    wf = wa_prev
    Pf = Pa_prev

    # Analysis
    # K = Pf @ H.T @ np.linalg.inv(H @ Pf @ H.T + Ro)
    S = H @ Pf @ H.T + Ro
    invS_H = np.linalg.solve(S, H)
    K = Pf @ invS_H.T

    zf = Hobs(xt, h_prev, wf, Wshapes)
    wa = wf + K @ (yo - zf)

    I_KH = np.eye(wa.size) - K @ H
    # Pa = I_KH @ Pf
    Pa = I_KH @ Pf @ I_KH.T + K @ Ro @ K.T

    return Pf, wa, Pa

# Forecast step
def RNNstep(xt, h_prev,  Px,
            wt, Wshapes, Pw, 
            G, H):
    """
    get the RNN output (zt) and its uncertainity (Pz) by using the current estimation of weights and biases (wt)
    input:
        xt     : current RNN input vector of shape (Din, 1)
        h_prev : previous RNN hidden layer of shape (Drec, 1)
        Px     : error covariance matrix of the RNN input vector
        wt     : current estimation of weights and biases flattened into a 1D vector
        Wshapes: list of matrix shapes for the state vector
        Pw     : error covariance matrix of the current estimation of weights and biases
        G      : tangent linear model of the RNN cell
        H      : tangent linear model of the observation operator
    returns:
        the forecast hidden layer, the forecast observation, and the forecast covariance matrix
        ht, zt, Pz
    """
    
    Win, Wrec, b_rec, Wout, b_out = statespace_to_matrices(wt, Wshapes)

    # Forecast
    ht, zt = RNNforward(xt, h_prev, Win, Wrec, b_rec, Wout, b_out)

    # Forecast covariance
    Pz = G @ Px @ G.T + H @ Pw @ H.T

    return ht, zt, Pz

# utility functions to get the tangent linear model
def get_LzdX(X0, dX, shapes):
    xt,   h_prev,  Win,  Wrec,  b_rec,  Wout,  b_out = statespace_to_matrices(X0, shapes)
    dxt, dh_prev, dWin, dWrec, db_rec, dWout, db_out = statespace_to_matrices(dX, shapes)

    dz = RNNforward_tlm(xt,   h_prev,  Win,  Wrec,  b_rec,  Wout,  b_out,
                     dxt, dh_prev, dWin, dWrec, db_rec, dWout, db_out)
    return dz

def get_LzU(U, X0, shapes):
    concat = []
    for i in range(U.shape[1]):
        Vi = get_LzdX(X0, U[:,i], shapes)
        concat.append(Vi.reshape(-1,1))
    
    LzU = np.concatenate(concat, axis=1)
    return LzU

def get_tlm(xt,  h_prev,
            wt, Wshapes):
    Win, Wrec, b_rec, Wout, b_out = statespace_to_matrices(wt, Wshapes)
    X0, X0shapes = matrices_to_statespace([xt, h_prev, Win, Wrec, b_rec, Wout, b_out])

    E = np.eye(X0.size)
    TLM = get_LzU(E, X0, X0shapes)

    Lz = TLM[:, :xt.size] # as marix G
    Lh = TLM[:, xt.size:xt.size+h_prev.size]
    Lw  = TLM[:, xt.size+h_prev.size:] # as matrix H
    
    return Lz, Lw


### Test -----------------------------------------------------------------------
if __name__ == '__main__':
    print("Test started ...")
    np.set_printoptions(precision=4, floatmode='fixed', suppress=True, sign=' ')

    time = np.linspace(0, 10, 100)
    true = np.zeros((3, time.size))
    true[0, :] = np.sin(time)
    true[1, :] = 2 * np.sin(time) * np.cos(time)
    true[2, :] = time

    # Add noise
    eta = 0.1
    train = true[:, :80].copy()
    train[0, :] = train[0, :] + eta * np.random.randn(train.shape[1])
    train[1, :] = train[1, :] + eta * np.random.randn(train.shape[1])
    R = np.eye(2) * eta**2

    valid = true[:, 80:].copy()

    # initial state
    rng = np.random.RandomState(42)
    mean, stddev = 0.0, 0.1

    Din, Drec, Dout = 2, 3, 2
    Win0   = rng.normal(mean, stddev, (Drec, Din ))
    Wrec0  = rng.normal(mean, stddev, (Drec, Drec))
    b_rec0 = rng.normal(mean, stddev, (Drec))
    Wout0  = rng.normal(mean, stddev, (Dout, Drec))
    b_out0 = rng.normal(mean, stddev, (Dout))

    matrices = [Win0, Wrec0, b_rec0, Wout0, b_out0]
    wa, Wshapes = matrices_to_statespace(matrices)
    Pinf = np.eye(wa.size) * 100

    h_prev = np.zeros(Drec)

    # training loop
    print("Training loop -------------------------")
    RTPS_a = 0.5
    for i in range(1, train.shape[1]):
        print(f'time: {train[-1, i]}')

        yo_prev = train[:-1, i-1].reshape(-1)
        yo_crnt = train[:-1, i  ].reshape(-1)

        # Kalman filter update
        G, H = get_tlm(yo_prev, h_prev, wa, Wshapes)

        check_symmetric(Pinf, 'Pinf')
        check_Rho(Pinf, 'Pinf')

        Pf, wa, Pa = KalmanFilter(wa, Pinf, 
                                  Wshapes, 
                                  yo_prev, h_prev, H,
                                  yo_crnt, R)
        
        Pinf = RTPS_a * Pf + (1 - RTPS_a) * Pa
    
        # Estimate the current observation with the updated state
        ht, zt, Pz = RNNstep(yo_prev, h_prev, R, 
                             wa, Wshapes, Pa, 
                             G, H)

        
        print(f'true : tt={true[:2, i]         }')
        print(f'obs  : yo={yo_crnt[:2]         }, Ro={np.diag(R[:2, :2])       }, sqrR: {np.sqrt(np.sum(np.diag(R[:2, :2])))       :.4f}')
        print(f'estmt: zf={zt[:2]              }, Pz={np.diag(Pz[:2, :2])      }, sqrP: {np.sqrt(np.sum(np.diag(Pz[:2, :2])))      :.4f}')
        print(f'error: er={zt[:2] - true[:2, i]}, e2={(zt[:2] - true[:2, i])**2}, rmse: {np.sqrt(np.sum((zt[:2] - true[:2, i])**2)):.4f}')
    
        h_prev = ht

    print("Training loop is done")

    # Forecast loop
    print("Forecast loop -------------------------")
    xt = train[:-1, -1].reshape(-1)
    Px = R
    for i in range(1, valid.shape[1]):
        print(f'time: {valid[-1, i]}')

        # Forecast
        G, H = get_tlm(xt, ht, wa, Wshapes)        
    
        ht, zt, Pz = RNNstep(xt, ht, Px, 
                             wa, Wshapes, Pa, 
                             G, H)
        
        check_symmetric(Pz, 'Forecast cvar Pz')
        check_Rho(Pz, 'Forecast cvar Pz')

        print(f'true : tt={valid[:2, i]}')
        print(f'estmt: zf={zt[:2]               }, Pz={np.diag(Pz[:2, :2])       }, sqrP: {np.sqrt(np.sum(np.diag(Pz[:2, :2])))       :.4f}')
        print(f'error: er={zt[:2] - valid[:2, i]}, e2={(zt[:2] - valid[:2, i])**2}, rmse: {np.sqrt(np.sum((zt[:2] - valid[:2, i])**2)):.4f}')

        xt = zt
        Px = Pz

    print("Forecast loop is done")
    print("Test is done")
