### Grid search for w0_seed(free), number of hidden unit is fixed to 20
# Author: Sho Sato
# Date: 2024-09-20

import platform                  # Python 3.9.16
import sys

import numpy      as np   # Numpy 1.23.2
import pandas     as pd   # Pandas 2.0.3
import EKFtrainedRNN as EKF_RNN

# parameters
tS0 = 2004.8739 # training Start time (default)
tE0 = 2022.3746 # training End   time (default)
fS0 = 2022.6247 # forecast Start time (default)
fE0 = 2024.3749 # forecast End   time (default)

dt = 0.25 # dt=0.25yrs (more precisely 365.25×0.25 days)
inv_dt = 4
J = 195 # Number of Gauss coefficients up to degree n=13
d_max = 2 # Order of the derivative

Din  = J
Dout = J

def geomagRNNpred(
    coef_raw, Rmatrix_raw,            # loaded MCM model as training data
    tS=tS0, tE=tE0, fS=fS0, fE=fE0,   # training and forecast period
    w0=None, Wshapes=None,            # initial weight space vector and shapes
    Din=J, Drec=None, Dout=J,         # RNN architecture parameters
    Pa0=None, ht0=None                # initial values for EKF-RNN

):
    ### Configuration -----------------------------------------------------------

    ### Program Start ------------------------------------------------------------
    # Create training / validation data
    time = coef_raw[0].index.values

    idx_tS = np.where(time == tS)[0][0]
    idx_tE = np.where(time == tE)[0][0]
    idx_fS = np.where(time == fS)[0][0]
    idx_fE = np.where(time == fE)[0][0]

    print(f"idx_tS={idx_tS:2d}, tS = {tS} = {time[idx_tS]} = {Rmatrix_raw[0][0, 0, idx_tS]}")
    print(f"idx_tE={idx_tE:2d}, tE = {tE} = {time[idx_tE]} = {Rmatrix_raw[0][0, 0, idx_tE]}")
    print(f"idx_fS={idx_fS:2d}, fS = {fS} = {time[idx_fS]}")
    print(f"idx_fE={idx_fE:2d}, fE = {fE} = {time[idx_fE]}")

    L = idx_tE - idx_tS + 1 # Number of training data
    M = idx_fE - idx_fS + 1 # Number of validation data
    N = idx_fE - idx_tS + 1 # Number of all data (training + forecast)

    print(f"training size: L={L}, validation size: M={M}, all data size: N={N}")

    coef_df       = [coef_raw[d].loc[tS:fE]    for d in range(d_max + 1)]
    coef_train_df = [coef_df[d] .loc[tS:tE]    for d in range(d_max + 1)]
    coef_train    = [coef_train_df[d].values.T for d in range(d_max + 1)]

    Rmatrix_train = [Rmatrix_raw[d][:, :, idx_tS:idx_tE + 1] for d in range(d_max + 1)]

    whole_y = coef_df      [d_max]
    train_y = coef_train_df[d_max]
    train_R = Rmatrix_train[d_max]
    valid_y = whole_y.loc[fS:fE]



    ### Assimilation -------------------------------------------------------------

    ### Training Loop -----------------------------------------------------------
    # trace computation
    time_memo   = np.zeros(          N)
    wt_memo     = np.zeros((w0.size, N))
    diagPw_memo = np.zeros((w0.size, N))
    time_memo[:]      = np.nan
    wt_memo[:, :]     = np.nan
    diagPw_memo[:, :] = np.nan

    ht_memo = np.zeros((Drec,       N))
    zt_memo = np.zeros((Dout,       N))
    Pz_memo = np.zeros((Dout, Dout, N))
    ht_memo[:, :]    = np.nan
    zt_memo[:, :]    = np.nan
    Pz_memo[:, :, :] = np.nan

    Pinf_check_memo = np.zeros(N)
    Pz_check_memo   = np.zeros(N)

    # EKF-RNN
    w_prev, Pinf = w0[:], Pa0[:, :]
    h_prev = ht0[:]
    y_prev = train_y.iloc[0, :]
    R_prev = train_R[:, :, 0]

    time_memo[0]      = time[idx_tS]
    wt_memo[:, 0]     = w0[:]
    diagPw_memo[:, 0] = np.diag(Pa0)[:]
    ht_memo[:, 0]     = ht0[:]

    wa = w0[:]
    Pa = Pa0[:, :]

    # EKF-RNN training loop (idx=1, 2, ..., L-1)
    RTPS_a = 0.5
    for i, t in enumerate(time[idx_tS+1 : idx_tE+1], start=1):
        y_crnt = train_y.iloc[i,   :]
        R_crnt = train_R[:, :, i]

        print(f"i = {i}, memo_idx = {i}; time = {t}")
        print(f"y_prev_time = {y_prev.name} --> y_crnt_time = {y_crnt.name}, R_time = {R_crnt[0, 0]}")
        time_memo[i]   = t

        yo_prev = y_prev.values.reshape(-1)
        yo_crnt = y_crnt.values.reshape(-1)
        if np.nan in yo_prev:
            print(f"skipped. (yo_prev is NaN) {yo_prev[:2]}")
            y_prev = y_crnt
            R_prev = R_crnt
            continue 

        G, H = EKF_RNN.get_tlm(yo_prev, h_prev, w_prev, Wshapes)

        Pinf_check_memo[i] += EKF_RNN.check_symmetric(Pinf, "Pinf")
        Pinf_check_memo[i] += EKF_RNN.check_Rhomatrix(Pinf, "Pinf") # This code is heavy and slow!   

        Pf, wa, Pa = EKF_RNN.KalmanFilter(w_prev, Pinf, 
                                          Wshapes, 
                                          yo_prev, h_prev, H,
                                          yo_crnt, R_crnt[1:, 1:])

        Pinf = RTPS_a * Pf + (1 - RTPS_a) * Pa

        wt_memo[:, i]     = wa[:]
        diagPw_memo[:, i] = np.diag(Pa)[:]

        G, H = EKF_RNN.get_tlm(yo_prev, h_prev, wa, Wshapes)
        ht, zt, Pz = EKF_RNN.RNNstep(yo_prev, h_prev, R_prev[1:, 1:], 
                                     wa, Wshapes, Pa, 
                                     G, H)

        Pz_check_memo[i] += EKF_RNN.check_symmetric(Pz, 'Forecast cvar Pz')
        Pz_check_memo[i] += EKF_RNN.check_Rhomatrix(Pz, 'Forecast cvar Pz') # This code is heavy and slow!

        ht_memo[:,    i] = ht[:]
        zt_memo[:,    i] = zt[:]
        Pz_memo[:, :, i] = Pz[:, :]

        print(f'obs  : yo={yo_crnt[:2]         }, Ro={np.diag(R_crnt[1:3, 1:3])}, sqrR: {np.sqrt(np.sum(np.diag(R_crnt[1:, 1:]))):.4f}')
        print(f'estmt: zf={zt[:2]              }, Pz={np.diag(Pz[:2, :2])      }, sqrP: {np.sqrt(np.sum(np.diag(Pz[:, :])))      :.4f} = GRGt: {np.sqrt(np.sum(np.diag(G @ R_prev[1:, 1:] @ G.T))):.4f} + HPHt: {np.sqrt(np.sum(np.diag(H @ Pa @ H.T))):.4f}')
        print(f'error: er={zt[:2] - yo_crnt[:2]}, e2={(zt[:2] - yo_crnt[:2])**2}, rmse: {np.sqrt(np.sum((zt[:] - yo_crnt[:])**2)):.4f}')

        w_prev = wa[:]
        h_prev = ht[:]
        y_prev = y_crnt
        R_prev = R_crnt


    ### Forecast Loop -----------------------------------------------------------
    w_ast  = wa[:]
    Pw_ast = Pa[:, :]

    t_tE = time_memo[L-1]
    y_tE = train_y. iloc[L-1, :]
    R_tE = train_R[:, :, L-1]
    print(f"training End: time: {time[idx_tE]}={t_tE} , y_time = {y_tE.name}, R_time = {R_tE[0, 0]}")

    t_prev  = t_tE
    h_prev  = ht_memo[:, L-1].copy()
    z_prev  = y_tE.values.reshape(-1)
    Pz_prev = R_tE[1:, 1:].copy()

    # EKF-RNN forecast loop (idx=L, L+1, ..., N)
    for i, t_crnt in enumerate(time[idx_fS : idx_fE+1], start=1):
        y_valid = valid_y.iloc[i-1, :]
        print(f"i = {i}, memo_idx = {L-1 + i}; prev_time: {t_prev} --> crnt_time: {t_crnt}, valid_y time = {y_valid.name}")
        time_memo     [L-1 + i] = t_crnt

        wt_memo    [:, L-1 + i] = w_ast
        diagPw_memo[:, L-1 + i] = np.diag(Pw_ast)

        G, H = EKF_RNN.get_tlm(z_prev, h_prev, w_ast, Wshapes)
        ht, zt, Pz = EKF_RNN.RNNstep(z_prev, h_prev, Pz_prev, 
                                     w_ast, Wshapes, Pw_ast, 
                                     G, H)

        Pz_check_memo[L-1 + i] += EKF_RNN.check_symmetric(Pz, 'Forecast cvar Pz')
        Pz_check_memo[L-1 + i] += EKF_RNN.check_Rhomatrix(Pz, 'Forecast cvar Pz') # This code is heavy and slow!

        ht_memo[:,    L-1 + i] = ht[:]
        zt_memo[:,    L-1 + i] = zt[:]
        Pz_memo[:, :, L-1 + i] = Pz[:, :]

        yo_valid = y_valid.values.reshape(-1)
        print(f'valid: yo={yo_valid[:2]         }')
        print(f'estmt: zf={zt[:2]               }, Pz={np.diag(Pz[:2, :2])       }, sqrP: {np.sqrt(np.sum(np.diag(Pz[:, :])))       :.4f} = GRGt: {np.sqrt(np.sum(np.diag(G @ R_prev[1:, 1:] @ G.T))):.4f} + HPHt: {np.sqrt(np.sum(np.diag(H @ Pa @ H.T))):.4f})')
        print(f'error: er={zt[:2] - yo_valid[:2]}, e2={(zt[:2] - yo_valid[:2])**2}, rmse: {np.sqrt(np.sum((zt[:] - yo_valid[:])**2)):.4f}')

        t_prev    = t_crnt
        h_prev[:] = ht[:]
        z_prev[:] = zt[:]
        Pz_prev[:, :] = Pz[:, :]


    ### Integration -------------------------------------------------------------
    d2g_memo = zt_memo
    d2R_memo = Pz_memo

    d1g_memo = np.zeros((J,    N))
    d1R_memo = np.zeros((J, J, N))

    d0g_memo = np.zeros((J,    N))
    d0R_memo = np.zeros((J, J, N))

    d1g_memo[:, :], d1R_memo[:, :, :] = np.nan, np.nan
    d0g_memo[:, :], d0R_memo[:, :, :] = np.nan, np.nan

    print("re-estimation in training period")
    d1g_train = coef_train[1]
    d1R_train = Rmatrix_train[1]
    d0g_train = coef_train[0]
    d0R_train = Rmatrix_train[0]
    for i in range(1, L+1):
        print(f"memo_idx = {i}; time = {time_memo[i]}")

        d1g_memo[:,    i] = d1g_train[:,      i-1] + d2g_memo[:,      i] * dt
        d1R_memo[:, :, i] = d1R_train[1:, 1:, i-1] + d2R_memo[0:, 0:, i] * (dt**2)

        d0g_memo[:,    i] = d0g_train[:,      i-1] + d1g_memo[:,      i] * dt
        d0R_memo[:, :, i] = d0R_train[1:, 1:, i-1] + d1R_memo[0:, 0:, i] * (dt**2)

    print(f"memo_idx = {L}; time = {time_memo[L]} is the first point of validation period, with g_train[{L-1}]")

    print("forecast in validation period")
    for i in range(L+1, N):
        print(f"memo_idx = {i}; time = {time_memo[i]}")

        d1g_memo[:,    i] = d1g_memo[:,      i-1] + d2g_memo[:,      i] * dt
        d1R_memo[:, :, i] = d1R_memo[0:, 0:, i-1] + d2R_memo[0:, 0:, i] * (dt**2)

        d0g_memo[:,    i] = d0g_memo[:,      i-1] + d1g_memo[:,      i] * dt
        d0R_memo[:, :, i] = d0R_memo[0:, 0:, i-1] + d1R_memo[0:, 0:, i] * (dt**2)


    ### Return results -------------------------------------------------------------
    return dict(
             time_memo=time_memo, # NOTE: time_memo is adjusted to Core field time. SV time is [time_memo - 0.125]: SA time is [time_memo - 0.25]
             ref_df=coef_df, ref_train=coef_train_df, Rmatrix_train=Rmatrix_train,
             L_M_N=(L, M, N),

             d2g_memo=d2g_memo, d2R_memo=d2R_memo, 
             d1g_memo=d1g_memo, d1R_memo=d1R_memo, 
             d0g_memo=d0g_memo, d0R_memo=d0R_memo, 
             Pz_check_memo=Pz_check_memo, 

             wt_memo=wt_memo, diagPw_memo=diagPw_memo, 
             w_ast=w_ast,     Pw_ast=Pw_ast,
             ht_memo=ht_memo, zt_memo=zt_memo, Pz_memo=Pz_memo, 
             Pinf_check_memo=Pinf_check_memo
             )
