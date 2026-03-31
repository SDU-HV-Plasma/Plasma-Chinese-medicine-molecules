# -*- coding: utf-8 -*-
"""
神经网络单变量剂量预测 - 改进对比图
实际点与预测曲线重合，优化显示效果
修改：实际值连线、预测点显示间隔更大、位点名称修正
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 设置GPU内存
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义颜色和样式列表
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  
MARKERS = ['o', 's', '^', 'D']  
LINE_STYLES = ['-', '--', '-.', ':']  

def load_and_unpivot(path):
    df = pd.read_csv(path, header=0)
    dose_cols = df.columns[1:3]
    reactant_cols = df.columns[3:]
    
    rows = []
    for _, r in df.iterrows():
        d1 = r[dose_cols[0]]
        d2 = r[dose_cols[1]]
        if pd.isna(d1) or pd.isna(d2):
            continue
        for col in reactant_cols:
            val = r[col]
            if pd.isna(val):
                continue
            rows.append((float(d1), float(d2), str(col), float(val)))
    
    long_df = pd.DataFrame(rows, columns=['d1','d2','reactant','y'])
    return long_df

def prepare_for_nn(long_df, le=None, scaler=None, include_poly=True):
    d1 = long_df['d1'].astype(float).values
    d2 = long_df['d2'].astype(float).values
    
    if include_poly:
        feats = [d1, d2, d1*d2, d1**2, d2**2]
    else:
        feats = [d1, d2]
    
    X_num = np.vstack(feats).T
    
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

def build_mixed_nn(n_reactants, input_dim_num, embed_dim=8, l2_reg=1e-4):
    num_in = Input(shape=(input_dim_num,), name='num_in')
    react_in = Input(shape=(1,), dtype='int32', name='react_in')
    
    emb = Embedding(input_dim=n_reactants, output_dim=embed_dim, 
                    input_length=1, name='emb')(react_in)
    emb_f = Flatten()(emb)
    
    x_num = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(num_in)
    x = Concatenate()([x_num, emb_f])
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(0.25)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    out = Dense(1, activation='sigmoid', name='out')(x)
    
    model = Model(inputs=[num_in, react_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mse']
    )
    return model

def extract_actual_data(full_long, dose_type='O'):
    all_reactants = sorted(full_long['reactant'].unique())
    original_reactants = all_reactants[:4]
    
    reactant_mapping = {}
    for i, orig_name in enumerate(original_reactants):
        new_name = f'C{i+1}'
        reactant_mapping[orig_name] = new_name
    
    actual_data = {}
    for orig_reactant in original_reactants:
        reactant = reactant_mapping[orig_reactant]
        
        if dose_type == 'O':
            mask = (full_long['reactant'] == orig_reactant) & (full_long['d2'] == 0)
        else:
            mask = (full_long['reactant'] == orig_reactant) & (full_long['d1'] == 0)
        
        subset = full_long[mask].copy()
        
        if dose_type == 'O':
            doses = subset['d1'].values
        else:
            doses = subset['d2'].values
        
        values = subset['y'].values
        sorted_indices = np.argsort(doses)
        doses = doses[sorted_indices]
        values = values[sorted_indices]
        
        actual_data[reactant] = {
            'doses': doses,
            'values': values,
            'dose_type': dose_type,
            'original_name': orig_reactant
        }
    return actual_data, reactant_mapping

def generate_predictions(model, le, scaler, full_long, dose_type='O', reactant_mapping=None):
    all_reactants = sorted(full_long['reactant'].unique())
    original_reactants = all_reactants[:4]
    
    continuous_doses = np.arange(0.1, 0.51, 0.01)
    n_points = len(continuous_doses)
    predictions = {}
    
    for i, orig_reactant in enumerate(original_reactants):
        reactant = reactant_mapping[orig_reactant] if reactant_mapping else orig_reactant
        react_code = le.transform([orig_reactant])[0]
        react_input = np.full((n_points, 1), react_code, dtype=int)
        
        if dose_type == 'O':
            d1_values = continuous_doses
            d2_values = np.zeros(n_points)
        else:
            d1_values = np.zeros(n_points)
            d2_values = continuous_doses
        
        d1 = d1_values
        d2 = d2_values
        X_num = np.column_stack([d1, d2, d1*d2, d1**2, d2**2])
        X_num_scaled = scaler.transform(X_num)
        
        react_pred = model.predict(
            {'num_in': X_num_scaled, 'react_in': react_input},
            batch_size=32,
            verbose=0
        ).ravel()
        
        react_pred = np.clip(react_pred, 0.0, 1.0)
        predictions[reactant] = {
            'doses': continuous_doses,
            'predictions': react_pred,
            'dose_type': dose_type,
            'original_name': orig_reactant
        }
    return predictions

def plot_actual_data_single(actual_data, output_dir, dose_type='O'):
    reactants = list(actual_data.keys())
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, reactant in enumerate(reactants):
        data = actual_data[reactant]
        doses = data['doses']
        values = data['values']
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        
        ax.plot(doses, values, '-', linewidth=2, color=color, alpha=0.6, zorder=4)
        ax.scatter(doses, values, s=100, color=color, alpha=0.9, label=reactant, 
                   marker=marker, edgecolors='white', linewidth=1.5, zorder=5)
    
    ax.set_xlabel(f'{dose_type}剂量', fontsize=14)
    ax.set_ylabel('实际值', fontsize=14)
    other_var = 'OH' if dose_type == 'O' else 'O'
    ax.set_title(f'{dose_type}剂量实际数据图 ({other_var}=0)', fontsize=16, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=12, loc='best')
    ax.set_xticks(np.arange(0.1, 0.51, 0.05))
    ax.set_xticklabels([f'{x:.2f}' for x in np.arange(0.1, 0.51, 0.05)], rotation=45)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'actual_points_{dose_type}_{other_var}=0.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path

def plot_predictions_single(predictions, output_dir, dose_type='O'):
    reactants = list(predictions.keys())
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, reactant in enumerate(reactants):
        data = predictions[reactant]
        doses = data['doses']
        preds = data['predictions']
        color = COLORS[idx % len(COLORS)]
        line_style = LINE_STYLES[idx % len(LINE_STYLES)]
        
        doses_dense = np.linspace(doses.min(), doses.max(), 300)
        spline = make_interp_spline(doses, preds, k=3)
        preds_smooth = spline(doses_dense)
        
        ax.plot(doses_dense, preds_smooth, linestyle=line_style, linewidth=2.5, 
                alpha=0.8, color=color, label=reactant)
        
        step = max(1, len(doses) // 11)
        display_indices = np.arange(0, len(doses), step)
        display_doses = doses[display_indices]
        display_preds = preds[display_indices]
        
        ax.scatter(display_doses, display_preds, alpha=0.9, s=70, 
                   color=color, edgecolors='white', linewidth=1.5, zorder=5)
    
    ax.set_xlabel(f'{dose_type}剂量', fontsize=14)
    ax.set_ylabel('预测值', fontsize=14)
    other_var = 'OH' if dose_type == 'O' else 'O'
    ax.set_title(f'{dose_type}剂量预测曲线图 ({other_var}=0) [每隔4%显示点]', fontsize=16, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=12, loc='best')
    ax.set_xticks(np.arange(0.1, 0.51, 0.05))
    ax.set_xticklabels([f'{x:.2f}' for x in np.arange(0.1, 0.51, 0.05)], rotation=45)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'predicted_curve_{dose_type}_{other_var}=0.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path

def plot_comparison_single(actual_data, predictions, output_dir, dose_type='O'):
    reactants = list(actual_data.keys())
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, reactant in enumerate(reactants):
        actual_doses = actual_data[reactant]['doses']
        actual_values = actual_data[reactant]['values']
        pred_doses = predictions[reactant]['doses']
        pred_values = predictions[reactant]['predictions']
        
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        
        pred_doses_dense = np.linspace(pred_doses.min(), pred_doses.max(), 300)
        spline = make_interp_spline(pred_doses, pred_values, k=3)
        pred_smooth = spline(pred_doses_dense)
        
        ax.plot(pred_doses_dense, pred_smooth, '-', linewidth=2.5, alpha=0.7, color=color, label=f'{reactant}')
        ax.scatter(actual_doses, actual_values, s=100, color=color, alpha=0.9, marker=marker, 
                   edgecolors='white', linewidth=1.5, zorder=10, label=f'{reactant} (实际点)')
        
        for actual_dose, actual_value in zip(actual_doses, actual_values):
            pred_idx = np.argmin(np.abs(pred_doses - actual_dose))
            pred_value = pred_values[pred_idx]
            ax.scatter(actual_dose, pred_value, s=60, color='white', alpha=0.8, marker='o', 
                       edgecolors=color, linewidth=1.5, zorder=9)
    
    ax.set_xlabel(f'{dose_type}剂量', fontsize=14)
    ax.set_ylabel('概率值', fontsize=14)
    other_var = 'OH' if dose_type == 'O' else 'O'
    ax.set_title(f'{dose_type}剂量实际点与预测曲线对比 ({other_var}=0)', fontsize=16, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([-0.05, 1.05])
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, label='预测曲线'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='实际数据点')
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='best')
    ax.set_xticks(np.arange(0.1, 0.51, 0.05))
    ax.set_xticklabels([f'{x:.2f}' for x in np.arange(0.1, 0.51, 0.05)], rotation=45)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'comparison_improved_{dose_type}_{other_var}=0.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path

def plot_detailed_comparison(actual_data, predictions, output_dir, dose_type='O'):
    reactants = list(actual_data.keys())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    for idx, reactant in enumerate(reactants):
        actual = actual_data[reactant]
        pred = predictions[reactant]
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        
        pred_doses_dense = np.linspace(pred['doses'].min(), pred['doses'].max(), 300)
        spline = make_interp_spline(pred['doses'], pred['predictions'], k=3)
        pred_smooth = spline(pred_doses_dense)
        
        ax1.plot(pred_doses_dense, pred_smooth, '-', linewidth=2, alpha=0.7, color=color, label=reactant)
        ax1.scatter(actual['doses'], actual['values'], s=80, color=color, alpha=0.9, marker=marker, edgecolors='white', zorder=5)
        
        errors = []
        for actual_dose, actual_value in zip(actual['doses'], actual['values']):
            pred_idx = np.argmin(np.abs(pred['doses'] - actual_dose))
            pred_value = pred['predictions'][pred_idx]
            errors.append(abs(actual_value - pred_value))
        
        ax2.bar(np.arange(len(actual['doses'])) + idx*0.2, errors, width=0.2, alpha=0.7, color=color, label=reactant)
    
    other_var = 'OH' if dose_type == 'O' else 'O'
    ax1.set_title(f'{dose_type}剂量预测曲线与实际点对比 ({other_var}=0)', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'{dose_type}剂量', fontsize=12)
    ax1.set_ylabel('概率值', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xticks(np.arange(0.1, 0.51, 0.05))
    ax1.set_xticklabels([f'{x:.2f}' for x in np.arange(0.1, 0.51, 0.05)], rotation=45)
    
    ax2.set_title('实际点与预测值的绝对误差', fontsize=14, fontweight='bold')
    ax2.set_xlabel('数据点索引', fontsize=12)
    ax2.set_ylabel('绝对误差', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'detailed_comparison_{dose_type}_{other_var}=0.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path

def main():
    full_data_path = r"E:\pycharm_data\PAEC30\PAEC.csv"
    output_dir = r"E:\Users\shuha\PycharmProjects\anjisuan_fivelayers_128\nn_improved_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    full_long = load_and_unpivot(full_data_path)
    X_num, X_react, y, le, scaler = prepare_for_nn(full_long, le=None, scaler=None, include_poly=True)
    Xn_tr_num, Xn_val_num, Xn_tr_react, Xn_val_react, y_tr, y_val = train_test_split(X_num, X_react, y, test_size=0.15, random_state=SEED)
    
    model = build_mixed_nn(n_reactants=len(le.classes_), input_dim_num=X_num.shape[1], embed_dim=8, l2_reg=1e-4)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    ]
    
    model.fit(
        {'num_in': Xn_tr_num, 'react_in': Xn_tr_react},
        y_tr,
        validation_data=({'num_in': Xn_val_num, 'react_in': Xn_val_react}, y_val),
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=2
    )
    
    model.save(os.path.join(output_dir, 'improved_model.h5'))
    
    actual_O, reactant_mapping = extract_actual_data(full_long, dose_type='O')
    pred_O = generate_predictions(model, le, scaler, full_long, dose_type='O', reactant_mapping=reactant_mapping)
    plot_actual_data_single(actual_O, output_dir, dose_type='O')
    plot_predictions_single(pred_O, output_dir, dose_type='O')
    plot_comparison_single(actual_O, pred_O, output_dir, dose_type='O')
    plot_detailed_comparison(actual_O, pred_O, output_dir, dose_type='O')
    
    actual_OH, reactant_mapping_oh = extract_actual_data(full_long, dose_type='OH')
    pred_OH = generate_predictions(model, le, scaler, full_long, dose_type='OH', reactant_mapping=reactant_mapping_oh)
    plot_actual_data_single(actual_OH, output_dir, dose_type='OH')
    plot_predictions_single(pred_OH, output_dir, dose_type='OH')
    plot_comparison_single(actual_OH, pred_OH, output_dir, dose_type='OH')
    plot_detailed_comparison(actual_OH, pred_OH, output_dir, dose_type='OH')

if __name__ == "__main__":
    main()