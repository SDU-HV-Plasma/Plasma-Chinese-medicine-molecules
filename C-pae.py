# -*- coding: utf-8 -*-
"""
修正版：NN (sigmoid out) + RandomForest + LightGBM 比较脚本
已修复 OneHotEncoder 在不同 sklearn 版本中的参数兼容问题。
请根据需要修改 train_path / test_path。
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# NN imports
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Try import lightgbm
has_lgb = True
try:
    import lightgbm as lgb
except Exception:
    has_lgb = False
    print("lightgbm not installed. To enable LightGBM, run: pip install lightgbm")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---- helper to construct OneHotEncoder compatibly across sklearn versions ----
def make_ohe(**kwargs):
    """
    Create OneHotEncoder compatible with both older sklearn (sparse) and newer (sparse_output).
    Use: ohe = make_ohe(sparse=False, handle_unknown='ignore')
    """
    try:
        return OneHotEncoder(**kwargs)
    except TypeError:
        # map 'sparse' -> 'sparse_output' if present
        if 'sparse' in kwargs:
            v = kwargs.pop('sparse')
            kwargs['sparse_output'] = v
        return OneHotEncoder(**kwargs)

# ---- data load / transform functions ----
def load_and_unpivot(path):
    """
    读取 csv 并展开为长表:
    假设:
      - 第一行是列名
      - 第2列和第3列为两个剂量（索引 1,2）
      - 第4列及以后为各反应物位置
    返回 DataFrame columns = ['d1','d2','reactant','y']
    """
    df = pd.read_csv(path, header=0)
    # 如果你的 dose 列不是第2/第3，请在此修改
    dose_cols = df.columns[1:3]
    reactant_cols = df.columns[3:]
    rows = []
    for _, r in df.iterrows():
        d1 = r[dose_cols[0]]
        d2 = r[dose_cols[1]]
        for col in reactant_cols:
            val = r[col]
            if pd.isna(val):
                continue
            rows.append((d1, d2, col, float(val)))
    long_df = pd.DataFrame(rows, columns=['d1','d2','reactant','y'])
    return long_df

def make_numeric_features(df, include_poly=True):
    """
    构造数值特征矩阵（可包含多项式/交互项）
    """
    d1 = df['d1'].astype(float).values
    d2 = df['d2'].astype(float).values
    feats = [d1, d2]
    if include_poly:
        feats.extend([d1*d2, d1**2, d2**2, np.log(np.abs(d1)+1e-6), np.log(np.abs(d2)+1e-6)])
    X_num = np.vstack(feats).T
    return X_num

def prepare_for_nn(long_df, le=None, scaler=None, include_poly=True):
    X_num = make_numeric_features(long_df, include_poly=include_poly)
    if scaler is None:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)
    else:
        X_num = scaler.transform(X_num)

    if le is None:
        le = LabelEncoder()
        r_idx = le.fit_transform(long_df['reactant'].values)
    else:
        r_idx = le.transform(long_df['reactant'].values)
    X_react = r_idx.reshape(-1, 1).astype(int)
    y = long_df['y'].astype(float).values
    return X_num, X_react, y, le, scaler

def prepare_for_trees(long_df, ohe=None, include_poly=True):
    """
    为树模型准备特征：numeric + one-hot reactant
    兼容 sklearn 版本差异
    """
    X_num = make_numeric_features(long_df, include_poly=include_poly)
    if ohe is None:
        ohe = make_ohe(sparse=False, handle_unknown='ignore')
        X_react_ohe = ohe.fit_transform(long_df[['reactant']])
    else:
        X_react_ohe = ohe.transform(long_df[['reactant']])
    X = np.hstack([X_num, X_react_ohe])
    y = long_df['y'].astype(float).values
    return X, y, ohe

# ---- model builders / utilities ----
def build_nn(n_reactants, input_dim_num, embed_dim=8, l2_reg=1e-4):
    num_in = Input(shape=(input_dim_num,), name='num_in')
    react_in = Input(shape=(1,), dtype='int32', name='react_in')

    emb = Embedding(input_dim=n_reactants, output_dim=embed_dim, input_length=1, name='emb')(react_in)
    emb_f = Flatten()(emb)

    x_num = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(num_in)
    x = Concatenate()([x_num, emb_f])
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(0.25)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    out = Dense(1, activation='sigmoid', name='out')(x)  # 限制在 (0,1)

    model = Model(inputs=[num_in, react_in], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=['mse'])
    return model

def wide_from_long_with_preds(long_df, preds, out_csv):
    long_df = long_df.copy()
    long_df['pred'] = preds
    wide = long_df.pivot_table(index=['d1','d2'], columns='reactant', values='pred')
    wide.reset_index(inplace=True)
    wide.to_csv(out_csv, index=False)
    return wide

def per_reactant_rmse(long_df, pred_col='pred'):
    res = long_df.groupby('reactant').apply(lambda g: sqrt(mean_squared_error(g['y'], g[pred_col])))
    return res.sort_values(ascending=False)

# ---- main ----
def main():
    # --------- 配置路径 ----------
    train_path = r"E:\mafang\新建文件夹\PAEC-train30.csv"
    test_path  = r"E:\mafang\新建文件夹\PAEC-pre30.csv"

    # 读取并展开
    train_long = load_and_unpivot(train_path)
    test_long = load_and_unpivot(test_path)

    # ----- NN 准备 -----
    Xn_num, Xn_react, y_n, le, scaler = prepare_for_nn(train_long, le=None, scaler=None, include_poly=True)
    input_dim_num = Xn_num.shape[1]
    n_reactants = len(le.classes_)
    embed_dim = min(8, max(2, n_reactants // 2))

    # 划分 validation
    Xn_tr_num, Xn_val_num, Xn_tr_react, Xn_val_react, y_tr, y_val = train_test_split(
        Xn_num, Xn_react, y_n, test_size=0.15, random_state=SEED)

    nn = build_nn(n_reactants=n_reactants, input_dim_num=input_dim_num, embed_dim=embed_dim, l2_reg=1e-3)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6, verbose=0)
    ]

    print("Training NN ...")
    nn.fit({'num_in': Xn_tr_num, 'react_in': Xn_tr_react},
           y_tr,
           validation_data=({'num_in': Xn_val_num, 'react_in': Xn_val_react}, y_val),
           epochs=1000, batch_size=16, callbacks=callbacks, verbose=2)

    # 准备测试集 NN 输入（使用训练集的 encoder & scaler）
    Xtest_num, Xtest_react, y_test_dummy, _, _ = prepare_for_nn(test_long, le=le, scaler=scaler, include_poly=True)
    nn_preds = nn.predict({'num_in': Xtest_num, 'react_in': Xtest_react}).ravel()
    nn_preds = np.clip(nn_preds, 0.0, 1.0)

    test_long_nn = test_long.copy()
    test_long_nn['pred'] = nn_preds
    nn_mse = mean_squared_error(test_long_nn['y'], test_long_nn['pred'])
    print(f"NN Test MSE: {nn_mse:.6f}, RMSE: {sqrt(nn_mse):.6f}")
    print("NN per-reactant RMSE (top worst):")
    print(per_reactant_rmse(test_long_nn, pred_col='pred').head(10))

    wide_nn = wide_from_long_with_preds(test_long, nn_preds, out_csv='predictions_nn.csv')
    print("NN predictions saved to predictions_nn.csv")

    # ----- RandomForest / LightGBM -----
    print("\nPrepare features for tree models (OneHot reactant + numeric poly)...")
    X_tree, y_tree, ohe = prepare_for_trees(train_long, ohe=None, include_poly=True)
    X_test_tree, y_test_tree, _ = prepare_for_trees(test_long, ohe=ohe, include_poly=True)

    # RF Cross-Validation
    print("\nCross-validating RandomForest ...")
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=SEED)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rf_cv_scores = -cross_val_score(rf, X_tree, y_tree, cv=cv, scoring='neg_mean_squared_error')
    print(f"RF CV MSE mean: {rf_cv_scores.mean():.6f}, std: {rf_cv_scores.std():.6f}")

    rf.fit(X_tree, y_tree)
    rf_preds = rf.predict(X_test_tree)
    rf_preds = np.clip(rf_preds, 0.0, 1.0)
    test_long_rf = test_long.copy()
    test_long_rf['pred'] = rf_preds
    rf_mse = mean_squared_error(test_long_rf['y'], test_long_rf['pred'])
    print(f"RF Test MSE: {rf_mse:.6f}, RMSE: {sqrt(rf_mse):.6f}")
    print("RF per-reactant RMSE (top worst):")
    print(per_reactant_rmse(test_long_rf, pred_col='pred').head(10))
    wide_rf = wide_from_long_with_preds(test_long, rf_preds, out_csv='predictions_rf.csv')
    print("RF predictions saved to predictions_rf.csv")

    # LightGBM (若可用)
    if has_lgb:
        print("\nCross-validating LightGBM ...")
        lgb_params = {
            'objective': 'regression',
            'metric': 'mse',
            'verbosity': -1,
            'seed': SEED,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 5
        }
        lgb_train_data = lgb.Dataset(X_tree, label=y_tree)
        cvres = lgb.cv(lgb_params, lgb_train_data, num_boost_round=500, nfold=5, seed=SEED, stratified=False,
                       early_stopping_rounds=50, verbose_eval=False)
        best_iter = len(cvres[next(iter(cvres.keys()))])
        print(f"LGB CV best iters: {best_iter}, CV MSE (last mean): {cvres[next(iter(cvres.keys()))][-1]:.6f}")

        lgbm = lgb.train(lgb_params, lgb_train_data, num_boost_round=best_iter)
        lgb_preds = lgbm.predict(X_test_tree)
        lgb_preds = np.clip(lgb_preds, 0.0, 1.0)
        test_long_lgb = test_long.copy()
        test_long_lgb['pred'] = lgb_preds
        lgb_mse = mean_squared_error(test_long_lgb['y'], test_long_lgb['pred'])
        print(f"LGB Test MSE: {lgb_mse:.6f}, RMSE: {sqrt(lgb_mse):.6f}")
        print("LGB per-reactant RMSE (top worst):")
        print(per_reactant_rmse(test_long_lgb, pred_col='pred').head(10))
        wide_lgb = wide_from_long_with_preds(test_long, lgb_preds, out_csv='predictions_lgb.csv')
        print("LGB predictions saved to predictions_lgb.csv")
    else:
        lgb_preds = None

    # ----- Simple ensemble: average available model predictions -----
    print("\nEnsembling predictions ...")
    preds_list = [nn_preds, rf_preds]
    names = ['NN', 'RF']
    if lgb_preds is not None:
        preds_list.append(lgb_preds)
        names.append('LGB')
    ensemble_preds = np.mean(np.vstack(preds_list), axis=0)
    ensemble_preds = np.clip(ensemble_preds, 0.0, 1.0)

    test_long_ens = test_long.copy()
    test_long_ens['pred'] = ensemble_preds
    ens_mse = mean_squared_error(test_long_ens['y'], test_long_ens['pred'])
    print(f"Ensemble ({', '.join(names)}) Test MSE: {ens_mse:.6f}, RMSE: {sqrt(ens_mse):.6f}")
    print("Ensemble per-reactant RMSE (top worst):")
    print(per_reactant_rmse(test_long_ens, pred_col='pred').head(10))
    wide_ens = wide_from_long_with_preds(test_long, ensemble_preds, out_csv='predictions_ensemble.csv')
    print("Ensemble predictions saved to predictions_ensemble.csv")

    print("\nAll done. Outputs saved:")
    print(" - predictions_nn.csv")
    print(" - predictions_rf.csv")
    if has_lgb:
        print(" - predictions_lgb.csv")
    print(" - predictions_ensemble.csv")

if __name__ == "__main__":
    main()