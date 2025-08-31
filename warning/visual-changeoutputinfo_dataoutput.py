import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def smooth_array(data, window_size=1):
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number")
    pad_size = window_size // 2
    padded_data = np.pad(data, pad_size, mode='edge')
    smoothed = np.zeros(data.shape)
    for i in range(len(data)):
        smoothed[i] = np.mean(padded_data[i:i+window_size])
    return smoothed

def euclidean_distance(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.sqrt(np.sum((a - b) ** 2)))

def best_shapelet_match_euclidean(signal_1d, shapelet_1d):

    x = np.asarray(signal_1d, dtype=np.float32).flatten()
    s = np.asarray(shapelet_1d, dtype=np.float32).flatten()
    n = len(x); m = len(s)
    if m > n:
        return float('inf'), 0, m - 1
    best_dist = float('inf')
    best_start = 0

    for start in range(0, n - m + 1):
        win = x[start:start + m]
        dist = euclidean_distance(win, s)
        if dist < best_dist:
            best_dist = dist
            best_start = start
    return best_dist, best_start, best_start + m - 1


def plot_explanations(dataset, top_k=6, max_sample=100, dpath=None):
    if dpath is None:
        dpath = ('checkpoints/SBM/ours/'
                 'dnn-FCN_seed-42_k-10_div-0.1_reg-0.1_eps-1.0_beta-constant_'
                 'dfunc-euclidean_cls-linear/test_results.pkl')
    with open(dpath, 'rb') as f:
        d = pickle.load(f)

    fontsize = 16
    legend_fontsize = 14
    smooth_window_size = 1  

    for k, v in d.items():
        try:
            print(k, v.shape)
        except Exception:
            print(k)

    num_class = int(d['w'].shape[0])
    num_sample = int(d['x'].shape[0])
    num_channel = int(d['x'].shape[-1])
    length = int(d['x'].shape[1])

    figures = []
    for label in range(num_class):
        fig = plt.figure(figsize=(21, num_channel * 1.5))

        data_axs = []
        for c in range(num_channel, 0, -1):
            ax = fig.add_subplot(num_channel, 3, 1 + 3*(c-1))
            if c != num_channel:
                ax.set_xticklabels([])
            ax.set_ylabel(f"$x^{{{c}}}$", fontsize=fontsize)
            data_axs.append(ax)
        data_axs.reverse()
        if label == 0:
            data_axs[0].set_title(f"Gas risk dataset: Category no risk", fontsize=fontsize)
        else:
            data_axs[0].set_title(f"Gas risk dataset: Category risk", fontsize=fontsize)

        pos_axs = []
        for i in range(top_k, 0, -1):
            ax = fig.add_subplot(top_k, 3, 2 + 3*(i-1))
            if i != top_k:
                ax.set_xticklabels([])
            pos_axs.append(ax)
        pos_axs.reverse()
        pos_axs[0].set_title(f"Top six Shapelets fragments Positive relevant", fontsize=fontsize)

        neg_axs = []
        for i in range(top_k, 0, -1):
            ax = fig.add_subplot(top_k, 3, 3 + 3*(i-1))
            if i != top_k:
                ax.set_xticklabels([])
            neg_axs.append(ax)
        neg_axs.reverse()
        neg_axs[0].set_title(f"Top six Shapelets fragments Negative relevant", fontsize=fontsize)

        figures.append((fig, data_axs, pos_axs, neg_axs))

    for sample_id in range(min(num_sample, max_sample)):
        label = int(d['trues'][sample_id])
        fig, data_axs, pos_axs, neg_axs = figures[label]
        for c in range(int(d['x'].shape[-1])):
            t = np.linspace(0, 1, d['x'].shape[1])
            data_axs[c].plot(t, d['x'][sample_id, :, c].flatten(),
                             color="tab:gray", alpha=0.1, linewidth=2)

    alpha_factor = 0.15
    for label in range(num_class):
        fig, data_axs, pos_axs, neg_axs = figures[label]
        class_w = d['w'][label, :]
        length = d['x'].shape[1]

        top_k_idx = np.argsort(-class_w)[:top_k]
        neg_k_idx = np.argsort(class_w)[:top_k]

        for i, s_id in enumerate(top_k_idx):
            shapelet, s_channel = d['shapelets'][s_id]
            t = np.linspace(0, shapelet.shape[0]/length, shapelet.shape[0])
            pos_axs[i].plot(t, smooth_array(shapelet, window_size=smooth_window_size),
                            color='tab:blue', alpha=1 - alpha_factor*i, linewidth=3,
                            label=f"$s_{{{s_id}}}: w={{{class_w[s_id]:.2f}}}$ on $x^{{{s_channel+1}}}$")
            pos_axs[i].set_xlim(0, 1)
            pos_axs[i].legend(loc='upper right', fontsize=legend_fontsize)

        for i, s_id in enumerate(neg_k_idx):
            shapelet, s_channel = d['shapelets'][s_id]
            t = np.linspace(0, shapelet.shape[0]/length, shapelet.shape[0])
            neg_axs[i].plot(t, smooth_array(shapelet, window_size=smooth_window_size),
                            color='tab:red', alpha=1 - alpha_factor*i, linewidth=3,
                            label=f"$s_{{{s_id}}}: w={{{class_w[s_id]:.2f}}}$ on $x^{{{s_channel+1}}}$")
            neg_axs[i].set_xlim(0, 1)
            neg_axs[i].legend(loc='upper right', fontsize=legend_fontsize)

    return figures, d

def export_per_sample_channel_best(
    d,
    dataset,
    top_m=4,
    keep_top_n_per_class=20,
    max_channels=4,
    tol=0.10,
    save_root='./figures/global'
):

    X = np.asarray(d['x'], dtype=np.float32)    # [N, T, C]
    y = np.asarray(d['trues']).astype(int)      # [N]
    W = np.asarray(d['w'], dtype=np.float32)    # [num_class, num_shapelets]
    shapelets = d['shapelets']                  # list of (shapelet_1d, channel)

    num_classes = int(W.shape[0])
    N, T, C = X.shape
    num_shapelets = int(W.shape[1])

    save_dir = os.path.join(save_root, dataset, 'exports_channel_best')
    os.makedirs(save_dir, exist_ok=True)

    shapelets_long_rows = []
    shapelets_wide_rows = []
    for sid, (s_vals, s_ch) in enumerate(shapelets):
        s_vals = np.asarray(s_vals, dtype=np.float32).flatten()
        s_ch = int(s_ch)
        L = len(s_vals)
        for idx, val in enumerate(s_vals):
            shapelets_long_rows.append({
                'shapelet_id': sid,
                'channel': s_ch,
                'length': L,
                'idx_in_shapelet': idx,
                'value': float(val)
            })
        shapelets_wide_rows.append({
            'shapelet_id': sid,
            'channel': s_ch,
            'length': L,
            'values': ';'.join(str(float(v)) for v in s_vals)
        })
    pd.DataFrame(shapelets_long_rows).to_csv(os.path.join(save_dir, 'shapelets_catalog_long.csv'), index=False)
    pd.DataFrame(shapelets_wide_rows).to_csv(os.path.join(save_dir, 'shapelets_catalog_wide.csv'), index=False)

    w_rows = []
    for cls in range(num_classes):
        for sid in range(num_shapelets):
            w_rows.append({'class': cls, 'shapelet_id': sid, 'w': float(W[cls, sid])})
    pd.DataFrame(w_rows).to_csv(os.path.join(save_dir, 'w_matrix.csv'), index=False)

    all_summary_rows = []

    excel_path = os.path.join(save_dir, f'{dataset}_channelbest_top{top_m}_best{keep_top_n_per_class}.xlsx')
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        pd.DataFrame(shapelets_long_rows).to_excel(writer, sheet_name='shapelets_catalog', index=False)

        for cls in range(num_classes):
            class_w = W[cls, :]
            top_idx = np.argsort(-class_w)[:top_m] 

            cls_idx = np.where(y == cls)[0]
            per_sample_scores = []
            per_sample_cache = {}

            for sid in cls_idx:
                x_seq = X[sid]  # [T, C]

                top_matches = []
                best_overall_for_sort = {'dist': float('inf')}
                for rank, shp_id in enumerate(top_idx, start=1):
                    s_vals, s_ch = shapelets[shp_id]
                    s_vals = np.asarray(s_vals, dtype=np.float32).flatten()
                    s_ch = int(s_ch)
                    dist, st, ed = best_shapelet_match_euclidean(x_seq[:, s_ch], s_vals)
                    top_matches.append({
                        'rank': rank,
                        'shapelet_id': int(shp_id),
                        'shapelet_channel': s_ch,
                        'shapelet_length': int(len(s_vals)),
                        'weight_w': float(class_w[shp_id]),
                        'best_dist': float(dist),
                        'start_idx': int(st),
                        'end_idx': int(ed),
                        'shapelet_values': ';'.join(str(float(v)) for v in s_vals)
                    })
                    if dist < best_overall_for_sort['dist']:
                        best_overall_for_sort['dist'] = dist


                channel_best = []
                for ch in range(min(max_channels, C)):

                    cand_ids = [sid0 for sid0 in top_idx if int(shapelets[sid0][1]) == ch]

                    if len(cand_ids) == 0:
                        cand_ids = [sid0 for sid0 in range(num_shapelets) if int(shapelets[sid0][1]) == ch]

 
                    cand_matches = []
                    for cand in cand_ids:
                        s_vals, s_ch = shapelets[cand]
                        s_vals = np.asarray(s_vals, dtype=np.float32).flatten()

                        if int(s_ch) != ch:
                            continue
                        dist, st, ed = best_shapelet_match_euclidean(x_seq[:, ch], s_vals)
                        cand_matches.append({
                            'shapelet_id': int(cand),
                            'length': int(len(s_vals)),
                            'dist': float(dist),
                            'start': int(st),
                            'end': int(ed),
                            'values': ';'.join(str(float(v)) for v in s_vals),
                            'weight_w': float(class_w[cand]) if cand < class_w.shape[0] else np.nan
                        })

                    if len(cand_matches) == 0:
                        channel_best.append({
                            'channel': ch,
                            'selected_shapelet_id': np.nan,
                            'selected_length': np.nan,
                            'start_idx': np.nan,
                            'end_idx': np.nan,
                            'dist': np.nan,
                            'weight_w': np.nan,
                            'shapelet_values': ''
                        })
                        continue

                    dists = np.array([c['dist'] for c in cand_matches], dtype=np.float32)
                    min_dist = float(np.min(dists[np.isfinite(dists)]) if np.any(np.isfinite(dists)) else float('inf'))
                    thr = min_dist * (1.0 + tol) if np.isfinite(min_dist) and min_dist > 0 else (min_dist + tol)


                    eligible = [c for c in cand_matches if np.isfinite(c['dist']) and c['dist'] <= thr]
                    if len(eligible) == 0:

                        best = min(cand_matches, key=lambda x: (float('inf') if not np.isfinite(x['dist']) else x['dist']))
                    else:

                        max_len = max(c['length'] for c in eligible)
                        longest = [c for c in eligible if c['length'] == max_len]

                        best = min(longest, key=lambda x: x['dist'])

                    channel_best.append({
                        'channel': ch,
                        'selected_shapelet_id': int(best['shapelet_id']),
                        'selected_length': int(best['length']),
                        'start_idx': int(best['start']),
                        'end_idx': int(best['end']),
                        'dist': float(best['dist']),
                        'weight_w': float(best.get('weight_w', np.nan)),
                        'shapelet_values': best.get('values', '')
                    })


                per_sample_scores.append((sid, float(best_overall_for_sort['dist'])))
                per_sample_cache[sid] = {
                    'top_matches': top_matches,
                    'best_for_sort': best_overall_for_sort,
                    'channel_best': channel_best
                }


            per_sample_scores.sort(key=lambda x: x[1])
            selected = per_sample_scores[:min(keep_top_n_per_class, len(per_sample_scores))]


            cls_summary_rows = []
            for sid, bestd in selected:
                bo = per_sample_cache[sid]['best_for_sort']
                cls_summary_rows.append({
                    'class': cls,
                    'sample_id': sid,
                    'best_min_dist_among_topm': float(bo['dist'])
                })
            cls_sum_df = pd.DataFrame(cls_summary_rows)
            if not cls_sum_df.empty:
                sheet_name = f'summary_cls{cls}'
                if len(sheet_name) > 31: sheet_name = sheet_name[:31]
                cls_sum_df.to_excel(writer, sheet_name=sheet_name, index=False)

            for sid, _ in selected:
                x_seq = X[sid]
                cache = per_sample_cache[sid]
                top_matches_df = pd.DataFrame(cache['top_matches'])
                channel_best_df = pd.DataFrame(cache['channel_best'])

                base_info_df = pd.DataFrame([{
                    'dataset': dataset,
                    'class': cls,
                    'sample_id': int(sid),
                    'T': int(T),
                    'C': int(C),
                    'top_m_shapelets': int(top_m)
                }])


                seq_df = pd.DataFrame({'t': np.arange(T, dtype=int)})
                for ch in range(C):
                    seq_df[f'x^{ch+1}'] = x_seq[:, ch].astype(np.float32)


                sheet_name = f'cls{cls}_idx{sid}'
                if len(sheet_name) > 31: sheet_name = sheet_name[:31]
                base_info_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
                top_matches_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=3)
                seq_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=5 + len(top_matches_df) + 2)
                start_r = 5 + len(top_matches_df) + 2 + len(seq_df) + 2
                channel_best_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_r)

                sample_dir = os.path.join(save_dir, f'class_{cls}', f'sample_{sid}')
                os.makedirs(sample_dir, exist_ok=True)
                base_info_df.to_csv(os.path.join(sample_dir, 'sample_meta.csv'), index=False)
                top_matches_df.to_csv(os.path.join(sample_dir, 'matches_topm.csv'), index=False)
                seq_df.to_csv(os.path.join(sample_dir, 'sequence.csv'), index=False)
                channel_best_df.to_csv(os.path.join(sample_dir, 'channel_best_matches.csv'), index=False)


                for rec in cache['top_matches']:
                    sid_shape = int(rec['shapelet_id'])
                    vals = rec['shapelet_values']
                    arr = np.array([float(v) for v in vals.split(';')], dtype=np.float32) if isinstance(vals, str) and vals != '' else np.array([])
                    if arr.size > 0:
                        shp_df = pd.DataFrame({'idx_in_shapelet': np.arange(len(arr)), 'value': arr})
                        shp_df.to_csv(os.path.join(sample_dir, f'shapelet_sid{sid_shape}.csv'), index=False)

                bo = per_sample_cache[sid]['best_for_sort']
                all_summary_rows.append({
                    'class': cls,
                    'sample_id': sid,
                    'best_min_dist_among_topm': float(bo['dist'])
                })

    all_summary_df = pd.DataFrame(all_summary_rows)
    all_summary_csv = os.path.join(save_dir, f'{dataset}_channelbest_top{top_m}_best{keep_top_n_per_class}_summary_all.csv')
    all_summary_df.to_csv(all_summary_csv, index=False)

    print(' Excel:', excel_path)
    print(' CSV :', all_summary_csv)
    print(' shapelets catalog:', os.path.join(save_dir, 'shapelets_catalog_wide.csv'))
    print(' w matrix:', os.path.join(save_dir, 'w_matrix.csv'))
    print(' per-sample folders:', os.path.join(save_dir, 'class_*', 'sample_*'))

if __name__ == '__main__':
    dataset = 'BasicMotions'
    figs, d = plot_explanations(dataset, top_k=6, max_sample=300)
    save_path = f'./figures/global/{dataset}'
    os.makedirs(save_path, exist_ok=True)
    for i, fig in enumerate(figs):
        fig[0].savefig(f'{save_path}/{dataset}_channel{i}.svg', bbox_inches='tight', pad_inches=0.2)
        plt.close(fig[0])

    export_per_sample_channel_best(
        d, dataset, top_m=4, keep_top_n_per_class=20, max_channels=4, tol=0.10, save_root='./figures/global'
    )
