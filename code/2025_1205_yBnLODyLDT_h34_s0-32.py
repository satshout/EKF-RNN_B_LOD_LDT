### Learn the core field snapshot g(t) and d(LOD)/dt using EKF-RNN:
# Author: Sho Sato
# Date: 2025-12-05, last update: 2025-12-05
# d_max=0-4, Drec=34, w0_seed="00000" - "11111" (32 patterns)
# The estimated execution time for each (d, seed) pair is about 6 hours on M1 Macbook Pro.
# So the total execution time for all d_max (0-4) and seed (0-31) is about 6 hours × 5 × 32 = 960 hours = 40 days.

import os
import platform           # Python 3.11.5
import sys

import numpy      as np   # Numpy 1.24.3
import pandas     as pd   # Pandas 2.1.1
import EKFtrainedRNN as EKF_RNN

np.set_printoptions(precision=5, floatmode='fixed', suppress=True)
pd.options.display.float_format = '{:.5f}'.format

script_name = os.path.basename(__file__)
print(f"Running script: {script_name}")
print(f"Python Platform: {platform.platform()}")
print(f"    Python {sys.version}")
print(f"     Numpy {np.__version__}")
print(f"    Pandas {pd.__version__}")


### Configuration -----------------------------------------------------------
# load/save directories
load_src = "./data/processed"
save_dir = "./output/2025_1205_yBnLODyLDT_h34_s0-32"

# parameters
dt = 0.25 # dt=0.25yrs (more precisely 365.25×0.25 days)
inv_dt = 4
# tS = 2004.8739 # training Start time
# tE = 2014.6247 # training End   time
# fS = 2014.8749 # forecast Start time
# fE = 2019.6254 # forecast End   time

tS = 2004.8739 # training Start time
tE = 2018.6247 # training End   time (15 yrs)
fS = 2018.8749 # forecast Start time
fE = 2023.6254 # forecast End   time

J = 195 + 1 # Number of Gauss coefficients up to degree n=13, + LOD

### Define functions ---------------------------------------------------------
# Define initial state w0 generator
def checkerboard(shape, io):
    # checkerboard pattern for generating initial state w0
    if int(io):
        return (np.indices(shape).sum(axis=0) + 1) % 2
    else:
        return np.indices(shape).sum(axis=0) % 2

def get_energy(coef_array, time_array, gnames):
    energy_array = np.zeros((len(time_array), 14), dtype=float) # n==0 is for time

    for i, epoch in enumerate(time_array):
        #- print(epoch)
        energy_array[i, 0] = epoch

        for l, target in enumerate(gnames):
            n = int(target[2:].split(",")[0])

            energy_array[i, n] += (n+1) * coef_array[l, i] * coef_array[l, i]
    
    return energy_array.T

### Program Start ------------------------------------------------------------
# Load data
d0g_raw = pd.read_csv(f"{load_src}/coef_g.csv")
d1g_raw = pd.read_csv(f"{load_src}/coef_d1g.csv")
d2g_raw = pd.read_csv(f"{load_src}/coef_d2g.csv")
d3g_raw = pd.read_csv(f"{load_src}/coef_d3g.csv")
d4g_raw = pd.read_csv(f"{load_src}/coef_d4g.csv")

d0g_raw.set_index('YEAR', inplace=True)
d1g_raw.set_index('YEAR', inplace=True)
d2g_raw.set_index('YEAR', inplace=True)
d3g_raw.set_index('YEAR', inplace=True)
d4g_raw.set_index('YEAR', inplace=True)

d0R_raw = np.load(f"{load_src}/cvar_R.npy")
d1R_raw = np.load(f"{load_src}/cvar_d1R.npy")
d2R_raw = np.load(f"{load_src}/cvar_d2R.npy")
d3R_raw = np.load(f"{load_src}/cvar_d3R.npy")
d4R_raw = np.load(f"{load_src}/cvar_d4R.npy")

lod_raw = pd.read_csv(f"{load_src}/DlodDT.csv")
lod_raw.set_index('YEAR', inplace=True)

# Create training / validation data
time = d0g_raw.index.values
print("gcf_time", time)
print("d0R_time", d0R_raw[0, 0, :])
print("lod_time", lod_raw.index.values)

idx_tS = np.where(time == tS)[0][0]
idx_tE = np.where(time == tE)[0][0]
idx_fS = np.where(time == fS)[0][0]
idx_fE = np.where(time == fE)[0][0]

print(f"idx_tS={idx_tS:2d}, tS = {tS} = {time[idx_tS]} = {d0R_raw[0, 0, idx_tS]}")
print(f"idx_tE={idx_tE:2d}, tE = {tE} = {time[idx_tE]} = {d0R_raw[0, 0, idx_tE]}")
print(f"idx_fS={idx_fS:2d}, fS = {fS} = {time[idx_fS]} = {d0R_raw[0, 0, idx_fS]}")
print(f"idx_fE={idx_fE:2d}, fE = {fE} = {time[idx_fE]} = {d0R_raw[0, 0, idx_fE]}")

L = idx_tE - idx_tS + 1 # Number of training data
M = idx_fE - idx_fS + 1 # Number of validation data
N = idx_fE - idx_tS + 1 # Number of all data (training + forecast)

print(f"training size: L={L}, validation size: M={M}, all data size: N={N}")

coef_raw      = [d0g_raw.iloc[:, :J], 
                 d1g_raw.iloc[:, :J], 
                 d2g_raw.iloc[:, :J], 
                 d3g_raw.iloc[:, :J], 
                 d4g_raw.iloc[:, :J]]

# 置き換え対象の列インデックス
J_index = J - 1
# 置き換えたい値のシリーズ
replace_values = lod_raw['avg_lod']
# リスト内の各DataFrameに対して処理を実行
for df in coef_raw:
    # DataFrameの行数が置き換え値の数と一致しているか確認
    if len(df) != len(replace_values):
        print(f"⚠️ 警告: DataFrameの行数 ({len(df)}) が 'avg_lod' の数 ({len(replace_values)}) と一致しません。")
        raise ValueError("行数の不一致により、置き換えを実行できません。")
        
    # J列目（インデックス J-1）の値を一括で置き換え
    # loc を使用して、列インデックス J_index にある全ての行の値を代入
    df.iloc[:, J_index] = replace_values.values
    df.columns.values[J_index] = 'lod'  # 列名も更新
    
    
print("J列目（インデックス J-1）の値が 'lod_raw[avg_lod]' の値で置き換えられました。")

coef_df       = [coef_raw[d].loc[tS:fE]    for d in range(5)]
coef_train_df = [coef_df[d] .loc[tS:tE]    for d in range(5)]
coef_train    = [coef_train_df[d].values.T for d in range(5)]

Rmatrix_raw   = [d0R_raw[:J+1, :J+1, :].copy(), 
                 d1R_raw[:J+1, :J+1, :].copy(), 
                 d2R_raw[:J+1, :J+1, :].copy(), 
                 d3R_raw[:J+1, :J+1, :].copy(), 
                 d4R_raw[:J+1, :J+1, :].copy()]

# J+1行目/J+1列目（インデックス J）が操作対象です。
target_index = J 
replace_values_sq = lod_raw['std_lod'].iloc[:80] ** 2

for Rmatrix in Rmatrix_raw:
    # 1. 最終行 (インデックス J) と 最終列 (インデックス J) の要素を0にする
    
    # 最終行全体を0にする
    # Rmatrix[J, :, :]
    Rmatrix[target_index, :, :] = 0
    
    # 最終列全体を0にする
    # Rmatrix[:, J, :]
    Rmatrix[:, target_index, :] = 0
    
    # 2. 右下隅（対角要素 Rmatrix[J, J, :]）を lod_raw['sig_lod']**2 で置き換える
    # この操作で、1で0になった右下隅の要素が置き換え値で上書きされます。
    Rmatrix[target_index, target_index, :] = replace_values_sq.values
    
print("Rmatrix_rawの各行列について、最終行と最終列が操作されました。")
print(f"操作対象の行/列インデックスは {target_index} (J) です。")

Rmatrix_train = [Rmatrix_raw[d][:, :, idx_tS:idx_tE + 1] for d in range(5)]
print(Rmatrix_train[0][:, :, 0])
print(Rmatrix_train[0][:, :, 1])
print(Rmatrix_train[0][:, :, 2])
print(Rmatrix_train[0][:, :, 3])

###　Prepare ddP table ---------------------------------------------------------

columns     = coef_raw[0].columns
time_raw    = coef_raw[0].index.values

coef_raw    = [d0g_raw.iloc[:, :J], d1g_raw.iloc[:, :J], d2g_raw.iloc[:, :J]]
Rmatrix_raw = [d0R_raw[:J+1, :J+1, :], d1R_raw[:J+1, :J+1, :], d2R_raw[:J+1, :J+1, :]]

print("columns:", columns)
print(time_raw)
print(d0R_raw[0, 0, :])

# EKF-RNN setup
p0_r = 10
### d_max 
hidden_unit_size = 34
print(f"hidden_unit_size = {hidden_unit_size}")

Din  = J
Drec = hidden_unit_size
Dout = J

print(f"========================================")

for seed in range(31, -1, -1): # 0 to 31 in a reverse order
    b_seed = format(seed, '05b') # seed No. as 5-bit binary
    print(f"seed = {seed}, b_seed = {b_seed}")
    for d_max in range(5):
        print(f"========================================")
        print(f"d_max = {d_max}")
        print(f"========================================")
        ### Training data ------------------------------------------------

        whole_y = coef_df      [d_max]
        train_y = coef_train_df[d_max]
        train_R = Rmatrix_train[d_max]
        valid_y = whole_y.loc[fS:fE]
        
        ### Initial state w0 -------------------------------------------------------
        Win0   = 0.01 * checkerboard((Drec, Din ), b_seed[0])
        Wrec0  = 0.01 * checkerboard((Drec, Drec), b_seed[1])
        b_rec0 = 0.01 * checkerboard((Drec, 1   ), b_seed[2]).flatten()

        Wout0  = 0.01 * checkerboard((Dout, Drec), b_seed[3])
        b_out0 = 0.01 * checkerboard((Dout, 1   ), b_seed[4]).flatten()

        print(f"Win0 = {Win0[0, :2]}, Wrec0 = {Wrec0[0, :2]}, b_rec0 = {b_rec0[:2]}")
        print(f"Wout0 = {Wout0[0, :2]}, b_out0 = {b_out0[:2]}")

        w0, Wshapes = EKF_RNN.matrices_to_statespace([Win0, Wrec0, b_rec0, Wout0, b_out0])
        Pa0 = p0_r * np.eye(w0.size)

        ht0 = np.zeros(Drec)
        ### Assimilation -------------------------------------------------------------
        print("Assimilation Start")
        print("----------------------------------------")

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
            if i <= d_max: # yo_prev is NaN for i=1, 2(=d_max)
                print(f"skipped. (yo_prev is NaN) {yo_prev[:2]}")
                y_prev = y_crnt
                R_prev = R_crnt
                continue 

            G, H = EKF_RNN.get_tlm(yo_prev, h_prev, w_prev, Wshapes)

            # Pinf_check_memo[i] += EKF_RNN.check_symmetric(Pinf, "Pinf")
            # Pinf_check_memo[i] += EKF_RNN.check_Rhomatrix(Pinf, "Pinf") # This code is heavy and slow!   

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

            # Pz_check_memo[i] += EKF_RNN.check_symmetric(Pz, 'Forecast cvar Pz')
            # Pz_check_memo[i] += EKF_RNN.check_Rhomatrix(Pz, 'Forecast cvar Pz') # This code is heavy and slow!

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

            # Pz_check_memo[L-1 + i] += EKF_RNN.check_symmetric(Pz, 'Forecast cvar Pz')
            # Pz_check_memo[L-1 + i] += EKF_RNN.check_Rhomatrix(Pz, 'Forecast cvar Pz') # This code is heavy and slow!

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
        dg_memo, dR_memo, dRdiag_memo = [], [], []

        for d in range(5):
            dg_memo    .append(np.zeros((J,    N)))
            dR_memo    .append(np.zeros((J, J, N)))
            dRdiag_memo.append(np.zeros((J,    N)))

            dg_memo[d][:, :], dR_memo[d][:, :, :], dRdiag_memo[d][:, :] = np.nan, np.nan, np.nan

        dg_memo[d_max] = zt_memo
        dR_memo[d_max] = Pz_memo

        print("re-estimation in training period")
        dg_train, dR_train = [], []
        for d in range(5):
            dg_train.append(coef_train[d].copy())
            dR_train.append(Rmatrix_train[d].copy())

        for i in range(1, L+1):
            print(f"memo_idx = {i}; time = {time_memo[i]}")

            for d in range(d_max-1, -1, -1):
                dg_memo[d][:,    i] = dg_train[d][:,      i-1] + dg_memo[d+1][:,      i] * dt
                dR_memo[d][:, :, i] = dR_train[d][1:, 1:, i-1] + dR_memo[d+1][0:, 0:, i] * (dt**2)

        print(f"memo_idx = {L}; time = {time_memo[L]} is the first point of validation period, with g_train[{L-1}]")

        print("forecast in validation period")
        for i in range(L+1, N):
            print(f"memo_idx = {i}; time = {time_memo[i]}")

            for d in range(d_max-1, -1, -1):
                dg_memo[d][:,    i] = dg_memo[d][:,      i-1] + dg_memo[d+1][:,      i] * dt
                dR_memo[d][:, :, i] = dR_memo[d][0:, 0:, i-1] + dR_memo[d+1][0:, 0:, i] * (dt**2)

        for d in range(d_max+1):
            dRdiag_memo[d][:, :] = np.diagonal(dR_memo[d], axis1=0, axis2=1).T

            print(f"d{d}g_memo: {dg_memo[d].shape}, d{d}R_memo: {dR_memo[d].shape}, d{d}Rdiag_memo: {dRdiag_memo[d].shape}")

        ### Save results -------------------------------------------------------------
        filename = f"d{d_max}g_{Drec}_{b_seed}_geomag_memos.npz"

        read_file = f"{save_dir}/{filename}"

        np.savez(read_file, 
                    time_memo=time_memo, 
                    tS_tE_fS_fE=[tS, tE, fS, fE],
                    columns=columns,
                    d0g_memo=dg_memo[0], d0R_memo=dRdiag_memo[0],
                    d1g_memo=dg_memo[1], d1R_memo=dRdiag_memo[1],
                    d2g_memo=dg_memo[2], d2R_memo=dRdiag_memo[2],
                    d3g_memo=dg_memo[3], d3R_memo=dRdiag_memo[3],
                    d4g_memo=dg_memo[4], d4R_memo=dRdiag_memo[4],
                    ht_memo=ht_memo
                    )

        print(f"Saved: {read_file}")

        print(f"seed = {seed}: bseed = {b_seed} done.")
        print("----------------------------------------")
    print(f"d_max = {d_max} done.")
print("========================================")

print("Program end.")
