import json
import csv
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
from scipy import stats
from calibration import *
import pandas as pd
from eval_metrics import *
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import pearsonr
import argparse
import itertools
from os import listdir
from os.path import isfile, join
from normalisation import compute_z_norm, compute_fixed_std
import copy
import ast


def get_df(comet_dir, da_dir, nruns=100, docs=False):
    SETUP_PATH = comet_dir
    if args.score_type.lower() == 'da' or args.score_type.lower() == 'mqm':
        if args.ensemble:
            df_all = pd.read_csv(SETUP_PATH)
            df_all.dp_runs_scores = df_all.dp_runs_scores.apply(lambda x: [float(i) for i in x.split('|')])
        else:
            files = [f for f in listdir(SETUP_PATH) if isfile(join(SETUP_PATH, f))]
            sys_files = [f for f in files if (f.split('_')[0] == 'system') and ('Human' not in f)]
            da_scores = pd.read_csv(da_dir)
            da_scores.system = da_scores.system.apply(lambda x: x.split('.')[0])
            if args.score_type.lower() == 'mqm':
                da_scores = da_scores.rename(columns={'source': 'src', 'reference': 'ref', 'score': 'human_score'})
                da_scores.loc[:,'human_score'] = da_scores['human_score'].apply(lambda x:100-x)
            dfs = []
            for s in sys_files:
                f = open(join(SETUP_PATH,s), 'r')
                data = json.loads(f.read()) 
                f.close() 
                system_name = '_'.join(s.split('.')[0].split('_')[1:])
                lines = [[i['src'], i['mt'], i['ref'], i['dp_runs_scores']] for i in data if 'dp_runs_scores' in i.keys()]
                df_ = pd.DataFrame(data=np.array(lines), columns=['src','mt', 'ref', 'dp_runs_scores'])
                da_scores_ = da_scores[da_scores.system == system_name]
                df = df_.merge(da_scores_, how='inner', on=['src', 'mt'])
                # df['dp_runs_scores'] = df['dp_runs_scores'].apply(lambda x: x[:nruns])
                df.drop(['ref_x', 'ref_y'], axis=1, inplace=True)
                dfs.append(df)
            df_all = pd.concat(dfs)
            df_all.reset_index(inplace=True)

    elif args.score_type.lower() == 'hter':
        df_all = pd.read_csv(SETUP_PATH)
        if args.ensemble:
            df_all.dp_runs_scores = df_all.dp_runs_scores.apply(lambda x: [float(i) for i in x.split('|')])
        else:
            # df_all.dp_runs_scores = df_all.dp_runs_scores.apply(lambda x: [float(i) for i in x.split('[')[1].split(']')[0].split(', ')])
            df_all.dp_runs_scores = df_all.dp_runs_scores.apply(lambda x: ast.literal_eval(x))
        df_all.doc_id = df_all.doc_id.apply(lambda x: int(x))
        df_all.drop(['ref', 'src', 'mt', 'pe'], axis=1, inplace=True)
        
    if docs:
        try:
            #print('doc')
            sys_doc_ids = df_all.sys_doc_id.unique().tolist()
            doc_dp_runs_scores = [np.mean(df_all[df_all.sys_doc_id == i].dp_runs_scores.tolist(), axis=0) for i in sys_doc_ids]
            doc_z_score = [np.mean(df_all[df_all.sys_doc_id == i].human_score.tolist()) for i in sys_doc_ids]
            doc_sys = [df_all[df_all.sys_doc_id == i].system.unique()[0] for i in sys_doc_ids]
            df_doc = pd.DataFrame(data=np.array([doc_sys, sys_doc_ids, doc_dp_runs_scores, doc_z_score]).T, columns=['system', 'doc_id','dp_runs_scores', 'human_score'])
            df_doc.reset_index(inplace=True)
            return df_doc
        except:
            return df_all
   
    return df_all

# def map_psqm(psqm_file, df):
#     # file format: Human-B.0       independent.281139      1       1       rater2  Michael Jackson wore tape on his nose to get front pages, former bodyguard claims       Ehemaliger Bodyguard behauptet, Michael Jackson trug Pflaster auf der Nase, um in die Presse zu kommen  4
#     # file format: System   doc_name    system_id?   ?  annot# src mt  score
#     scores = [-1.0]*len(df)
#     print(df.head())
#     df['psqm_score'] = scores

#     with open(psqm_file, 'r') as psqmf:
#         for line in tqdm(psqmf):
#             fields = line.split('\t')
#             system = fields[0].split('.')[0]
#             src = fields[5]
#             score = fields[7]
#             idx_to_change = df.index[(df['system'] == system) & (df['src'] == src)]
#             df.loc[idx_to_change,'psqm_score'] = float(score)
#     #print(len(df))
#     new_dataframe = df[df['psqm_score'] >=0]
#     #print(len(new_dataframe))f
#     return new_dataframe


# def load_mqm_scores_from_df(mqm_df):
#     df = mqm_df
#     print(df.head())
#     systems_comet_scores = {}
#     systems_mqm_scores = {}
#     systems_ext = []
#     for i, row in df.iterrows():
#         if row['psqm_score']>=0.0:
#             sent_id = int(row['index'])
#             system = row['system']
#             system_ext = 'system_'+system+'.json'
#             if not system_ext in systems_ext:
#                 systems_ext.append(system_ext)
#             if not system_ext in systems_comet_scores:
#                 systems_comet_scores[system_ext]={}
#             if not system_ext in systems_mqm_scores:
#                 systems_mqm_scores[system_ext]={}
#             scores = row['dp_runs_scores']
#             systems_comet_scores[system_ext][sent_id] = scores
#             systems_mqm_scores[system_ext][sent_id] = row['psqm_score']
#     return(systems_comet_scores, systems_mqm_scores, systems_ext)


def load_da_scores_from_df(df):
    systems_comet_scores = {}
    systems_human_scores = {}
    systems_ext = {}
    for i, row in df.iterrows():
        sent_id = int(row['index'])
        system = row['system']
        system_ext = 'system_'+system+'.json'
        doc_id = row['doc_id']
        if not system_ext in systems_ext:
            #systems_ext.append(system_ext)
            systems_ext[system_ext]=[]
        systems_ext[system_ext].append(doc_id)
        if not system_ext in systems_comet_scores:
            systems_comet_scores[system_ext]={}
        if not doc_id in systems_comet_scores[system_ext]:
            systems_comet_scores[system_ext][doc_id]=[]
        if not system_ext in systems_human_scores:
            systems_human_scores[system_ext]={}
        if not doc_id in systems_human_scores[system_ext]:
            systems_human_scores[system_ext][doc_id]=[]
        scores = row['dp_runs_scores']
        systems_comet_scores[system_ext][doc_id].append(scores)
        systems_human_scores[system_ext][doc_id].append(row['human_score'])
        
    return(systems_comet_scores, systems_human_scores, systems_ext)

def split_k_fold(comet_scores, scores, systems_list, k=5):
    final_folds = []
    for system in systems_list:
        comet_sys_scores = comet_scores[system]
        sys_scores = scores[system]
        assert(len(comet_sys_scores) == len(sys_scores))
        # split based on the doc level
        zipped = [i for i in zip(comet_sys_scores, sys_scores)]
        folds = np.array_split(zipped, k)
        for i, fold in enumerate(folds):
            comet_scores_fold = {}
            human_scores_fold = {}
            if system not in comet_scores_fold:
                comet_scores_fold[system] = {}
                human_scores_fold[system] = {}
            for comet_doc_id, score_doc_id in fold:
                assert(len(comet_sys_scores[comet_doc_id]) == len(sys_scores[score_doc_id]))
                comet_scores_fold[system][comet_doc_id] = comet_sys_scores[comet_doc_id]
                human_scores_fold[system][score_doc_id] = sys_scores[score_doc_id]

            if len(final_folds) <= i:
                final_folds.append({})
                final_folds[i]['human'] = {}
                final_folds[i]['comet'] = {}

            final_folds[i]['human'][system] = human_scores_fold[system]
            final_folds[i]['comet'][system] = comet_scores_fold[system]
    
    # levels of nested dicts : 
    # 5 folds -> each fold has 2 dicts 'human' and "comet" ->
    # -> each 'human' and "comet" has N keys == systems -> docs -> segments
    return final_folds 


def merge_folds(list_of_folds):
    dev_fold = {}
    for i, fold in enumerate(list_of_folds):
        if i == 0:
            dev_fold = copy.deepcopy(fold)
        else:
            for human_sys, comet_sys in zip(fold['human'].keys(), fold['comet'].keys()):
                dev_fold['human'][human_sys].update(fold['human'][human_sys])
                dev_fold['comet'][comet_sys].update(fold['comet'][comet_sys])
    return dev_fold
        

def batch_data(all_da, all_comet, all_comet_avg, batch_size=1):
    n = len(all_da) - (len(all_da) % batch_size)
    batch_da = all_da[:n].reshape(n//batch_size, batch_size).mean(axis=1)
    #print(all_comet[:n, :].reshape(n//batch_size, batch_size, -1).shape)
    batch_comet_scores = [i.mean(axis=0) for i in all_comet[:n, :].reshape(n//batch_size, batch_size, -1)]
    batch_comet_avg = all_comet_avg[:n].reshape(
        n//batch_size, batch_size).mean(axis=1)
    batch_comet_std = all_comet[:n, :].reshape(
        n//batch_size, batch_size, -1).mean(axis=1).std(axis=-1)
    return batch_da, batch_comet_scores, batch_comet_avg, batch_comet_std


def standardize(scores_test, scores_dev, norm):
    norm_mean = 0.0
    norm_std = 1.0
    if norm:
        norm_mean, norm_std = compute_z_norm(scores_dev)
    #all_scores = np.array([val for _,sys in scores_test.items() for _,doc in sys.items() for _,val in doc.items() ])
    all_scores = np.array([val for _,sys in scores_test.items() for _,doc in sys.items() for val in doc ])
    #print(all_scores.shape)
    all_scores -= norm_mean
    all_scores /= norm_std
    #print(all_scores.shape)
    #np.squeeze(all_scores,axis=1)
    return all_scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process comet outputs')
    parser.add_argument('--comet-setup-file', type=str, 
                        help='path to comet setup to test on')
    parser.add_argument('--scores-file', type=str, 
                        help='path to scores for testing on')
    parser.add_argument('--norm', type=bool, default=True,
                        help='set to true to normalise the std on the ECE')
    parser.add_argument('--score-type', type=str, default='da', 
                        help='Choose type of scores between da | mqm | hter')
    parser.add_argument('--docs', default=False, action='store_true',
                        help= 'select segment or document level eval')
    parser.add_argument('--nruns', type=int, default=100,
                        help= 'select how many drpout runs to evaluate')
    parser.add_argument('--baseline', default=False, action='store_true',
                        help= 'select to evaluate the baseline only')
    parser.add_argument('--ensemble', default=False, action='store_true',
                        help= 'specify that you are evaluating ensemble merged data')
    args = parser.parse_args()
    
    test_year='2020'
    if '2019' in args.comet_setup_file:
        test_year='2019'
    
    #if args.score_type.lower()=='da':
    combined_df = get_df(args.comet_setup_file, args.scores_file, args.nruns, args.docs)
    print(list(combined_df.columns))
    
    systems_comet_scores, systems_human_scores, systems_ext = load_da_scores_from_df(combined_df)
    k_folds = split_k_fold(systems_comet_scores, systems_human_scores, systems_ext)

    cal_avgll_folds = []
    calibration_error_folds = []
    sharpness_cal_folds = []
    epiw_cal_folds = []
    pearson_acc_folds = []
    pearson_d2_cal_folds = []
    pearson_d1_cal_folds = []
    np_calibration_error_folds = []
    np_sharpness_cal_folds = []
    np_pearson_folds = []

    for i, fold in enumerate(k_folds):
        keys = np.arange(len(k_folds))  # 0,1,2,3,4
        dev = keys[keys != i]
        merged_dev = merge_folds([k_folds[k] for k in dev])
        # systems_comet_scores_test, systems_scores_test = k_folds[i]['comet'], k_folds[i]['human']
        systems_comet_scores_test, systems_scores_test = fold['comet'], fold['human']
        systems_comet_scores_dev, systems_scores_dev = merged_dev['comet'], merged_dev['human']
        print()
        print('- - - making dev/test split - - -')
        print('processing as test fold #', i)
        print('processing as dev folds #', dev)

        print()
        print(len(systems_comet_scores_test))
        print(len(systems_scores_test))
        
        norm_human_test = standardize(systems_scores_test, systems_scores_dev, args.norm)
        norm_comet_test = standardize(systems_comet_scores_test, systems_comet_scores_dev, args.norm)
        norm_comet_avg_test = norm_comet_test.mean(axis=1)
    
        # we need to repeat on the dev set to optimise the calibration parameters!
        norm_human_dev = standardize(systems_scores_dev, systems_scores_dev, args.norm)
        norm_comet_dev = standardize(systems_comet_scores_dev, systems_comet_scores_dev, args.norm)
        norm_comet_avg_dev = norm_comet_dev.mean(axis=1)    
    
        # if args.docs:
        #     batch_range = [1]
        # else:
        batch_range = [1]
        for batch_size in batch_range:
            batch_human_test, batch_comet_scores_test, batch_comet_avg_test, batch_comet_std_test = batch_data(
                norm_human_test, norm_comet_test, norm_comet_avg_test, batch_size=batch_size)
            batch_human_dev, batch_comet_scores_dev, batch_comet_avg_dev, batch_comet_std_dev = batch_data(
                norm_human_dev, norm_comet_dev, norm_comet_avg_dev, batch_size=batch_size)

            # Compute fixed std to use as a baseline model
            if args.baseline:
                fixed_std = compute_fixed_std(batch_comet_avg_dev, batch_human_dev)
                batch_baseline_stds_test = np.full_like(batch_comet_std_test, fixed_std)
                batch_baseline_stds_dev = np.full_like(batch_comet_std_dev, fixed_std)

                # Compute Pearson correlation between average COMET and DA.
                pearson_acc = stats.pearsonr(batch_comet_avg_test, batch_human_test)[0]

                base_calibration_error, gammas, base_matches = compute_calibration_error(
                    batch_human_test, batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                mcpe_base, gammas, mcpe_matches_base = compute_mcpe(
                    batch_human_test, batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                sharpness_base = compute_sharpness(batch_baseline_stds_test, std_sum=0, std_scale=1)



                epiw_base, gammas, epiw_matches_base = compute_epiw(
                    batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                # mpiw_base, gammas, mpiw_matches_base = compute_mpiw(
                #     batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                # ence_base, ence_gammas, ence_matches_base = compute_ence(
                #     batch_human_test, batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                # ence_base_rn, ence_gammas_rn, ence_matches_base_rn = compute_ence_rn(
                #     batch_human_test, batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                # ence_base_nn, ence_gammas_nn, ence_matches_base_nn = compute_ence_nn(
                #     batch_human_test, batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                # print("Baseline ECE = %f (baseline std = %f)"  % (base_calibration_error, fixed_std))
                # print("Baseline MCE = %f (baseline std = %f)"  % (mcpe_base, fixed_std))
                # print("Baseline sharpness  = %f (baseline std = %f)"  % (sharpness_base, fixed_std))
                # print("Baseline epiw sharpness  = %f (baseline std = %f)"  % (epiw_base, fixed_std))
                # print("Baseline mpiw sharpness = %f (baseline std = %f)"  % (mpiw_base, fixed_std))
                # print("Baseline ENCE = %f (baseline std = %f)"  % (ence_base, fixed_std))
                # print("Baseline ENCE_RN = %f (baseline std = %f)"  % (ence_base_rn, fixed_std))
                # print("Baseline ENCE_NN = %f (baseline std = %f)"  % (ence_base_nn, fixed_std))
                # print("Baseline Parametric CE = %f (baseline std = %f)"  % (base_calibration_error, fixed_std))
                # Compute Baseline ALL and NLL
                base_avgll, base_negll = compute_avgll(batch_human_test, batch_comet_avg_test, batch_baseline_stds_test)
                print("Baseline ALL = %f" % base_avgll)
                print("Baseline NLL = %f" % base_negll)
                print()

                medians = [np.median(i) for i in batch_comet_scores_test]
                np_pearson = stats.pearsonr(medians, batch_human_test)[0]

                np_s_vals = np.linspace(0, 5, 500)
                _, best_s = optimize_calibration_error_non_parametric_base(
                    batch_human_dev, batch_comet_avg_dev, s_vals=np_s_vals)
                np_calibration_error, np_gammas, np_matches = compute_calibration_error_non_parametric_base(
                    batch_human_test, batch_comet_avg_test, best_s)
                print("Non-parametric CE baseline = %f (best_s = %f)" % (np_calibration_error, best_s))

                np_sharpness_cal, gammas, matches = compute_epiw_np_base(batch_comet_avg_test, best_s)
                print("Non-parametric Sharpness = %f " % np_sharpness_cal)

                cal_avgll_folds.append(base_avgll)
                calibration_error_folds.append(base_calibration_error)
                sharpness_cal_folds.append(sharpness_base)
                epiw_cal_folds.append(epiw_base)
                pearson_acc_folds.append(pearson_acc)

                # np_calibration_error_folds.append(np_calibration_error)
                # np_sharpness_cal_folds.append(np_sharpness_cal)
                np_pearson_folds.append(np_pearson)

                ############# LATEX #################
                print('----------LATEX OUTPUTS----------')
                print('& average NLL & ECE & Sharpness & EPIW \\\\')
                print('& %f & %f & %f & %f \\\\' % (base_avgll, base_calibration_error, sharpness_base, epiw_base))
                print('& r(human, pred) & r(|pred-human|,std) \\\\')
                print('& %f & %f   \\\\' % (pearson_acc, 0))
                # print('& ECE_np & Sharpness_np \\\\')
                # print('& %f & %f   \\\\' % (np_calibration_error, np_sharpness_cal))
                print('& np_pearson \\\\')
                print('& %f   \\\\' % np_pearson)

            else:
                # Parametric CE
                # It assumes a parametric Gaussian distribution for the COMET scores.
                
                calibration_error, gammas, matches = compute_calibration_error(
                    batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1)
                # mcpe, gammas, mcpe_matches = compute_mcpe(
                #     batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1) 
                # sharpness = compute_sharpness(batch_comet_std_test, std_sum=0, std_scale=1)
                # epiw, gammas, epiw_matches = compute_epiw(
                #     batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1)
                # mpiw, gammas, mpiw_matches = compute_mpiw(
                #     batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1)
                # ence, ence_gammas, ence_matches = compute_ence(
                #     batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1)
                # ence_rn, ence_gammas_rn, ence_matches_rn = compute_ence_rn(
                #     batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1)
                # ence_nn, ence_gammas_nn, ence_matches_nn = compute_ence_nn(
                #     batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1)

                # print("ECE = %f" % calibration_error)
                # print("MCE = %f" % mcpe)
                # print("Sharpness = %f" % sharpness)
                # print("EPIW = %f" % epiw)
                # print("MPIW = %f" % mpiw)
                # print("ENCE = %f" % ence)
                # print("ENCE_RN = %f" % ence_rn)
                # print("ENCE_NN = %f" % ence_nn)
                # print("Parametric CE  = %f" % calibration_error)
                
                # Seek the best post-calibration to minimize calibration error.
                # The correction is std_transformed**2 = std_sum**2 + (std_scale*std)**2,
                # where std_sum and std_scale are correction parameters.
                std_sums = np.linspace(0, 2, 100)
                std_scales = np.linspace(1, 10, 100)
                _, std_sum, std_scale, = optimize_calibration_error(
                    batch_human_dev, batch_comet_avg_dev, batch_comet_std_dev,
                    std_sums=std_sums, std_scales=std_scales)
                
                # Compute Pearson correlation between average COMET and DA.
                pearson_acc = stats.pearsonr(batch_comet_avg_test, batch_human_test)[0]
                # print("Pearson (COMET, MQM) batch size =%d - r= %f" % (
                #     batch_size, pearson_acc))

                # Compute Pearson correlation between |COMET - DA| and COMET_std
                abs_diff = [abs(da - cs) for da, cs in zip(batch_human_test, batch_comet_avg_test)]
                pearson_d1 = stats.pearsonr(abs_diff, batch_comet_std_test)[0]
                # print("Pearson (|COMET-MQM|, COMET_std) batch size =%d - r= %f" % (
                #     batch_size, pearson_d1))
                
                # Compute Pearson correlation between |COMET - DA| and COMET_std
                abs_diff_sq = [(da - cs)**2 for da, cs in zip(batch_human_test, batch_comet_avg_test)]
                batch_comet_std_test_sq = [ x**2 for x in batch_comet_std_test]
                pearson_d2 = stats.pearsonr(abs_diff_sq, batch_comet_std_test_sq)[0]
                # print("Pearson ((COMET-MQM)**2, COMET_std**2) batch size =%d - r= %f" % (
                #     batch_size, pearson_d2))

                ## Calibrated pearsons
                batch_comet_std_test_transformed = np.sqrt(std_sum**2 + (std_scale*batch_comet_std_test)**2)
                # Compute Pearson correlation between |COMET - DA| and COMET_std
                abs_diff = [abs(da - cs) for da, cs in zip(batch_human_test, batch_comet_avg_test)]
                pearson_d1_cal = stats.pearsonr(abs_diff, batch_comet_std_test_transformed)[0]
                # print("Calibrated Pearson (|COMET-MQM|, COMET_std) batch size =%d - r= %f" % (
                #     batch_size, pearson_d1_cal))
                
                # Compute Pearson correlation between |COMET - DA| and COMET_std
                abs_diff_sq = [(da - cs)**2 for da, cs in zip(batch_human_test, batch_comet_avg_test)]
                batch_comet_std_test_transformed_sq = [ x**2 for x in batch_comet_std_test_transformed]
                pearson_d2_cal = stats.pearsonr(abs_diff_sq, batch_comet_std_test_transformed_sq)[0]
                # print("Calibrated Pearson ((COMET-MQM)**2, COMET_std**2) batch size =%d - r= %f" % (
                #     batch_size, pearson_d2_cal ))

                calibration_error, gammas, matches_cal = compute_calibration_error(
                    batch_human_test, batch_comet_avg_test, batch_comet_std_test,
                    std_sum=std_sum, std_scale=std_scale)
                
                # mcpe_cal, gammas, mcpe_matches_cal = compute_mcpe(
                #     batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)
                sharpness_cal = compute_sharpness(batch_comet_std_test, std_sum, std_scale)
                epiw_cal, gammas, epiw_matches_cal = compute_epiw(
                    batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)
                # mpiw_cal, gammas, mpiw_matches_cal = compute_mpiw(
                #     batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)
                # ence_cal, ence_gammas, ence_matches_cal = compute_ence(
                #     batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)
                # ence_cal_rn, ence_gammas_rn, ence_matches_cal_rn = compute_ence_rn(
                #     batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)
                # ence_cal_nn, ence_gammas_nn, ence_matches_cal_nn = compute_ence_nn(
                #     batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)

                # print("Calibrated ECE = %f (calibrated std_sum=%f, std_scale=%f)"  % (calibration_error, std_sum, std_scale))
                # print("Calibrated MCE = %f (calibrated std_sum=%f, std_scale=%f)"  % (mcpe_cal, std_sum, std_scale))
                # print("Calibrated sharpness  = %f (calibrated std_sum=%f, std_scale=%f)"  % (sharpness_cal, std_sum, std_scale))
                # print("Calibrated epiw sharpness  = %f (calibrated std_sum=%f, std_scale=%f)"  % (epiw_cal, std_sum, std_scale))
                # print("Calibrated mpiw sharpness = %f (calibrated std_sum=%f, std_scale=%f)"  % (mpiw_cal,  std_sum, std_scale))
                # print("Calibrated ENCE = %f (calibrated std_sum=%f, std_scale=%f)"  % (ence_cal,  std_sum, std_scale))
                # print("Calibrated ENCE_RN = %f (calibrated std_sum=%f, std_scale=%f)"  % (ence_cal_rn,  std_sum, std_scale))
                # print("Calibrated ENCE_NN = %f (calibrated std_sum=%f, std_scale=%f)"  % (ence_cal_nn,  std_sum, std_scale))
                # print("Calibrated Parametric CE = %f (calibrated std_sum=%f, std_scale=%f)" %
                #     (calibration_error, std_sum, std_scale))
                
                # Compute ALL and NLL
                avgll, negll = compute_avgll(batch_human_test, batch_comet_avg_test, batch_comet_std_test)       
                # print("ALL = %f" % avgll)
                # print("NLL = %f" % negll)
                # Compute ALL and NLL
                cal_avgll, cal_negll = compute_avgll(batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)       
                print("Calibrated ALL = %f" % cal_avgll)
                print("Calibrated NLL = %f" % cal_negll)


                ################### NON PRARAMETRIC ##################
                # Compute calibration error by binning different confidence intervals.
                # Non-parametric CE
                np_calibration_error, np_gammas, np_matches = compute_calibration_error_non_parametric(
                    batch_human_test, batch_comet_scores_test)
                print("Non-parametric CE = %f" % np_calibration_error)
                # Best non-parametric CE 

                medians = [np.median(i) for i in batch_comet_scores_test]
                np_pearson = stats.pearsonr(medians, batch_human_test)[0]

                # np_scaling_vals = np.linspace(0.05, 3, 20)
                # np_scaling_sums = np.linspace(-1,  1, 11)

                # _, best_scale_val, best_scale_sum = optimize_calibration_error_non_parametric(
                #     batch_human_dev, batch_comet_scores_dev, scaling_vals=np_scaling_vals, scaling_sums=np_scaling_sums)
                # np_calibration_error, np_gammas, np_matches = compute_calibration_error_non_parametric(
                #     batch_human_test, batch_comet_scores_test, scaling_val=best_scale_val, scaling_sum=best_scale_sum)
                # print("Non-parametric CE = %f (calibrated, best_scaling_val=%f, best_scaling_sum=%f)" %
                #     (np_calibration_error, best_scale_val, best_scale_sum))
                
                # np_sharpness_cal, gammas, matches = compute_epiw_np(batch_comet_scores_test, std_scale=best_scale_val, std_sum=best_scale_sum)
                # print("Non-parametric Sharpness = %f (calibrated, best_scaling_val=%f, best_scaling_sum=%f)" %
                #     (np_sharpness_cal, best_scale_val, best_scale_sum))

                ############# LATEX #################
                print()
                print('----------LATEX OUTPUTS----------')
                print('& average NLL & ECE & Sharpness & EPIW \\\\')
                print('& %f & %f & %f & %f \\\\' % (cal_avgll, calibration_error, sharpness_cal, epiw_cal))
                print('& r(human, pred) & r(|pred-human|,std) \\\\')
                print('& %f & %f   \\\\' % (pearson_acc, pearson_d1_cal))
                # print('& ECE_np & EPIW_np \\\\')
                # print('& %f & %f   \\\\' % (np_calibration_error, np_sharpness_cal))
                # print('& np_pearson \\\\')
                # print('& %f   \\\\' % np_pearson)
        
                cal_avgll_folds.append(cal_avgll)
                calibration_error_folds.append(calibration_error)
                sharpness_cal_folds.append(sharpness_cal)
                epiw_cal_folds.append(epiw_cal)
                pearson_acc_folds.append(pearson_acc)
                pearson_d2_cal_folds.append(pearson_d2_cal)
                pearson_d1_cal_folds.append(pearson_d1_cal)
                # np_calibration_error_folds.append(np_calibration_error)
                # np_sharpness_cal_folds.append(np_sharpness_cal)
                np_pearson_folds.append(np_pearson)

    if args.baseline:
        print()
        print('------AVERAGED OVER k FOLDS------')
        print('----------LATEX BASELINE OUTPUTS----------')
        print('& r(human, pred) & r(|pred-human|,std) & average NLL & ECE & Sharpness & EPIW\\\\')
        print('& %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\' % (round(np.mean(pearson_acc_folds),3), 0, round(np.mean(cal_avgll_folds),3), round(np.mean(calibration_error_folds),3), round(np.mean(sharpness_cal_folds),3), round(np.mean(epiw_cal_folds),3)))
        # print('& ECE_np & EPIW_np \\\\')
        # print('& %.3f & %.3f   \\\\' % (round(np.mean(np_calibration_error_folds),3), round(np.mean(np_sharpness_cal_folds),3)))
        print('& np_pearson \\\\')
        print('& %.3f   \\\\' % round(np.mean(np_pearson_folds),3))
    else:
        print()
        print('------AVERAGED OVER k FOLDS------')
        print('----------LATEX OUTPUTS----------')
        print('& PPS & UPS & average NLL & ECE & Sharpness  \\\\')
        print('& %.3f & %.3f & %.3f & %.3f & %.3f \\\\' % (round(np.mean(pearson_acc_folds),3), round(np.mean(pearson_d1_cal_folds),3), round(np.mean(cal_avgll_folds),3), round(np.mean(calibration_error_folds),3), round(np.mean(sharpness_cal_folds),3)))#, round(np.mean(epiw_cal_folds),3)))
        # print('& ECE_np & EPIW_np \\\\')
        # print('& %.3f & %.3f   \\\\' % (round(np.mean(np_calibration_error_folds),3), round(np.mean(np_sharpness_cal_folds),3)))
        # print('& np_pearson \\\\')
        # print('& %.3f   \\\\' % round(np.mean(np_pearson_folds),3))

        ################## FIGURES #################
        # matplotlib.rc('xtick', labelsize=15)
        # matplotlib.rc('ytick', labelsize=15)
        # plt.figure(figsize=(6.2,2.2))
        # plt.xlabel('Confidence level $\gamma$', fontsize=15)
        # plt.ylabel('ECE', fontsize=15)
        # # plt.title(args.score_type.upper() + ': 1719 on '+test_year+' - Batch size = %d' % batch_size)
        # plt.plot(gammas, matches, 'royalblue', label="Original ECE")
        # plt.plot(gammas, matches_cal, 'orangered', linestyle='dotted', label="Calibrated ECE", linewidth=3)
        # # plt.plot(gammas, base_matches, 'g', label="Baseline ECE")
        # plt.plot([0, 1], [0, 1], 'k', linewidth=0.9)
        # plt.legend()
        # plt.legend(prop={'size': 14})
        # plt.show()
        # plt.savefig('/media/hdd1/glushkovato/comet/COMET_uncertainty/ue_eval_scripts/figures/ECE_bs_final_squeeze_'+'.png', bbox_inches = "tight")
        # plt.close()

        # sample = [-0.7036617994308472, -0.39346814155578613, -0.4354693293571472, -0.48200106620788574,
        #   -0.6583844423294067, -0.6528894305229187, -0.4381236135959625, -0.11167386919260025,
        #   -0.28474313020706177, -0.40106481313705444, -0.2071761190891266, -0.3169260025024414,
        #   -0.42727798223495483, -0.2133534699678421, -0.37169918417930603, 0.02465342916548252,
        #   -0.4433746635913849, -0.2109990417957306, -0.3115224242210388, -0.12913624942302704,
        #   -0.3110971450805664, -0.2711679935455322, -0.2629014551639557, -0.161701962351799,
        #   -0.31409913301467896, -0.28766417503356934, -0.4218456745147705, -0.4927760362625122,
        #   -0.4791868329048157, -0.5151439309120178, -0.4783304035663605, -0.2807827591896057,
        #   -0.4361232817173004, -0.796786367893219, -0.2349693924188614, -0.2692130208015442,
        #   -0.5983560681343079, -0.3687020540237427, -0.3561617434024811, -0.35035240650177,
        #   -0.34771427512168884, -0.24625299870967865, -0.36683908104896545, -0.33239245414733887, 
        #   -0.4329518973827362, -0.3675892949104309, -0.6854426860809326, -0.21822558343410492, 
        #   -0.23549531400203705, -0.32744812965393066, -0.37420371174812317, -0.35194385051727295, 
        #   -0.2507886588573456, -0.6340183615684509, -0.40667828917503357 -0.20614176988601685,
        #   -0.24489286541938782, -0.4341568648815155, -0.37508007884025574, -0.5427935719490051,
        #   -0.46071887016296387, -0.3867534101009369, -0.30441683530807495, -0.15482938289642334,
        #   -0.3157658874988556, -0.2350553274154663, -0.5219535231590271, -0.7520877718925476,
        #   -0.39036089181900024, -0.39128726720809937, -0.09702187776565552, -0.3885476291179657,
        #   -0.35855793952941895, -0.10762306302785873, -0.32352709770202637, -0.3512462377548218,
        #   -0.32870057225227356, -0.4129355549812317, -0.38273707032203674, -0.5623825788497925,
        #   0.0948031097650528, -0.3213968575000763, -0.23260238766670227, -0.47009772062301636,
        #   -0.5744777321815491, -0.509739100933075, -0.15552622079849243, -0.29284384846687317,
        #   -0.19066350162029266, -0.44607532024383545, -0.5014781951904297, -0.4129786491394043,
        #   -0.40598946809768677, -0.4015771150588989, -0.29395225644111633, -0.4239853620529175,
        #   -0.4720333516597748, 0.004249433055520058, -0.540823757648468, -0.21973752975463867]
        
        sample = [0.5979697704315186,
                    0.5243543982505798,
                    0.6262829899787903,
                    0.5946624875068665,
                    0.5717122554779053,
                    0.6173045635223389,
                    0.6162536144256592,
                    0.5854749083518982,
                    0.5606750845909119,
                    0.6813172101974487,
                    0.7215273380279541,
                    0.7598370909690857,
                    0.5823168754577637,
                    0.6067937612533569,
                    0.6279692649841309,
                    0.5281223058700562,
                    0.5969744324684143,
                    0.5853649377822876,
                    0.48922640085220337,
                    0.5766693949699402,
                    0.5745446085929871,
                    0.5487832427024841,
                    0.647320032119751,
                    0.6826099157333374,
                    0.6288926005363464,
                    0.8613891005516052,
                    0.5903519988059998,
                    0.5309380292892456,
                    0.6349964141845703,
                    0.4533690810203552,
                    0.6135360598564148,
                    0.8157045841217041,
                    0.41160082817077637,
                    0.5631874799728394,
                    0.5206363797187805,
                    0.6249307990074158,
                    0.6297017931938171,
                    0.6902846097946167,
                    0.6883143782615662,
                    0.705655038356781,
                    0.5418302416801453,
                    0.6571133136749268,
                    0.7079156041145325,
                    0.6279858946800232,
                    0.6430858373641968,
                    0.5736710429191589,
                    0.6936737298965454,
                    0.634878396987915,
                    0.6792322397232056,
                    0.32287514209747314,
                    0.6860127449035645,
                    0.6514933109283447,
                    0.5734164714813232,
                    0.6523839235305786,
                    0.67072993516922,
                    0.7287837266921997,
                    0.6147819757461548,
                    0.701930582523346,
                    0.5400246381759644,
                    0.5519304275512695,
                    0.7217748165130615,
                    0.6027462482452393,
                    0.6484041810035706,
                    0.6087967753410339,
                    0.5354230403900146,
                    0.6050034165382385,
                    0.5663189888000488,
                    0.516261875629425,
                    0.6997227668762207,
                    0.6717677712440491,
                    0.5033883452415466,
                    0.5382351875305176,
                    0.6828071475028992,
                    0.6036621928215027,
                    0.5766533017158508,
                    0.5570096969604492,
                    0.5567960143089294,
                    0.47815167903900146,
                    0.7115316390991211,
                    0.6047992706298828,
                    0.6455367207527161,
                    0.648800253868103,
                    0.7322787642478943,
                    0.6291396021842957,
                    0.6151097416877747,
                    0.6088402271270752,
                    0.6159900426864624,
                    0.6520301699638367,
                    0.5357393622398376,
                    0.552027702331543,
                    0.47821640968322754,
                    0.6975051760673523,
                    0.6235313415527344,
                    0.5580254793167114,
                    0.5078786611557007,
                    0.625751256942749,
                    0.6019373536109924,
                    0.6296380758285522,
                    0.6481614708900452,
                    0.6562914848327637]

        plt.figure(figsize=(7,4))
        mu = np.mean(sample)
        sigma = np.std(sample)
        n, bins, patches = plt.hist(sample, bins=20, color='royalblue', density=1)
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
            np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

        plt.plot(bins, y, '--', color='orangered')
        # plt.hist(sample, bins=20, color='royalblue', density=1, alpha=0.7)
        plt.xlabel("Predicted values", size=15)
        plt.ylabel("Counts", size=15)
        plt.show()
        plt.savefig('/media/hdd1/glushkovato/comet/COMET_uncertainty/ue_eval_scripts/figures/sample_distr11.png', bbox_inches = "tight")
        plt.close()


        # matplotlib.rc('xtick', labelsize=15)
        # matplotlib.rc('ytick', labelsize=15)
        # plt.xlabel('N', fontsize=15)
        # plt.ylabel('Recall@N', fontsize=15)
        # plt.plot(recall, rec_comet_unc_prism_1, 'darkgreen',linestyle='dotted', label="UA-COMET")
        # plt.plot(recall, rec_comet_mean_prism_1, 'darkorange',linestyle='dashed', label="MCD COMET mean")
        # plt.plot(recall, rec_comet_original_prism_1, 'royalblue', label="COMET original")
        # plt.legend()
        # plt.legend(prop={'size': 15})
        # plt.show()
        # plt.savefig('figures/NNEW_combined_MQM_Recall@N_relevant_'+str(1)+'_perc.png')
        # plt.close()

        # plt.xlabel('Confidence level $\gamma$')
        # plt.ylabel('Sharpness')
        # plt.title(args.score_type.upper() + ': 1719 on '+test_year+' - Batch size = %d' % batch_size)
        # plt.plot(gammas, epiw_matches, 'b', label="Original sharpness")
        # plt.plot(gammas, epiw_matches_cal, 'r', label="Calibrated sharpness")
        # plt.plot(gammas, epiw_matches_base, 'g', label="Baseline sharpness")
        # plt.plot(gammas, mpiw_matches, 'b:', label="Original max sharpness")
        # plt.plot(gammas, mpiw_matches_cal, 'r:', label="Calibrated max sharpness")
        # #plt.plot([0, 1], [0, 1], 'k--')
        # plt.legend()
        # plt.show()
        # plt.savefig('figures/'+args.score_type.upper()+'_1719-'+test_year+'-SHARP_bs_'+str(batch_size)+'.png')
        # plt.close()
       
        # plt.xlabel('Bins (ascending std values) $')
        # plt.ylabel('ENCE')
        # plt.title(args.score_type.upper() + ': 1719 on '+test_year+' - Batch size = %d' % batch_size)
        # plt.plot(ence_gammas, ence_matches, 'b', label="Original ENCE")
        # plt.plot(ence_gammas, ence_matches_cal, 'r', label="Calibrated ENCE")
        # plt.plot(ence_gammas, ence_matches_base, 'g', label="Baseline ENCE")
        # #plt.plot([0, 1], [0, 1], 'k--')
        # plt.legend()
        # plt.show()
        # plt.savefig('figures/'+args.score_type.upper()+'_1719-'+test_year+'-ENCE_bs_'+str(batch_size)+'.png')
        # plt.close()

        # plt.xlabel('Bins (ascending std values) $')
        # plt.ylabel('ENCE RN')
        # plt.title(args.score_type.upper() + ': 1719 on '+test_year+' - Batch size = %d' % batch_size)
        # plt.plot(ence_gammas_rn, ence_matches_rn, 'b', label="Original ENCE_RN")
        # plt.plot(ence_gammas_rn, ence_matches_cal_rn, 'r', label="Calibrated ENCE_RN")
        # plt.plot(ence_gammas_rn, ence_matches_base_rn, 'g', label="Baseline ENCE_RN")
        # #plt.plot([0, 1], [0, 1], 'k--')
        # plt.legend()
        # plt.show()
        # plt.savefig('figures/'+args.score_type.upper()+'_1719-'+test_year+'-ENCE_RN_bs_'+str(batch_size)+'.png')
        # plt.close()


        # plt.xlabel('Bins (ascending std values) $')
        # plt.ylabel('ENCE NN')
        # plt.title(args.score_type.upper() + ': 1719 on '+test_year+' - Batch size = %d' % batch_size)
        # plt.plot(ence_gammas_nn, ence_matches_nn, 'b', label="Original ENCE_NN")
        # plt.plot(ence_gammas_nn, ence_matches_cal_nn, 'r', label="Calibrated ENCE_NN")
        # plt.plot(ence_gammas_nn, ence_matches_base_nn, 'g', label="Baseline ENCE_NN")
        # #plt.plot([0, 1], [0, 1], 'k--')
        # plt.legend()
        # plt.show()
        # plt.savefig('figures/'+args.score_type.upper()+'_1719-'+test_year+'-ENCE_NN_bs_'+str(batch_size)+'.png')
        # plt.close()


