import json
import csv
import numpy as np
import matplotlib.pyplot as plt
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
                df.drop(['ref_x', 'ref_y', 'lp', 'annotators'], axis=1, inplace=True)
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
    pearson_acc_folds = []
    pearson_d1_cal_folds = []

    np_calibration_error_folds = []
    np_sharpness_cal_folds = []

    for i, fold in enumerate(k_folds):
        keys = np.arange(len(k_folds))  # 0,1,2,3,4
        dev = keys[keys != i]
        merged_dev = merge_folds([k_folds[k] for k in dev])
        # systems_comet_scores_test, systems_scores_test = k_folds[i]['comet'], k_folds[i]['human']
        systems_comet_scores_test, systems_scores_test = fold['comet'], fold['human']
        systems_comet_scores_dev, systems_scores_dev = merged_dev['comet'], merged_dev['human']
        # print()
        # print('_____ ! _______')
        # np_scores = np.array([val for _,sys in systems_comet_scores_test.items() for _,doc in sys.items() for val in doc])
        # print(np_scores.shape)
        # np_scores = np.array([val for _,sys in systems_comet_scores_dev.items() for _,doc in sys.items() for val in doc])
        # print(np_scores.shape)

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

                base_calibration_error, gammas, base_matches = compute_calibration_error(
                    batch_human_test, batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                mcpe_base, gammas, mcpe_matches_base = compute_mcpe(
                    batch_human_test, batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                sharpness_base = compute_sharpness(batch_baseline_stds_test, std_sum=0, std_scale=1)
                epiw_base, gammas, epiw_matches_base = compute_epiw(
                    batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                mpiw_base, gammas, mpiw_matches_base = compute_mpiw(
                    batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                ence_base, ence_gammas, ence_matches_base = compute_ence(
                    batch_human_test, batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                ence_base_rn, ence_gammas_rn, ence_matches_base_rn = compute_ence_rn(
                    batch_human_test, batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                ence_base_nn, ence_gammas_nn, ence_matches_base_nn = compute_ence_nn(
                    batch_human_test, batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
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

                cal_avgll_folds.append(base_avgll)
                calibration_error_folds.append(base_calibration_error)
                sharpness_cal_folds.append(sharpness_base)

                ############# LATEX #################
                print('----------LATEX OUTPUTS----------')
                print('& average NLL & ECE & Sharpness \\\\')
                print('& %f & %f & %f  \\\\' % (np.round(base_avgll, 3), np.round(base_calibration_error, 3), np.round(sharpness_base, 3)))
                
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
                # epiw_cal, gammas, epiw_matches_cal = compute_epiw(
                #     batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)
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
                np_scaling_vals = np.linspace(0.05, 1, 10)
                np_scaling_sums = np.linspace(-10, 0, 10)
                _, best_scale_val, best_scale_sum = optimize_calibration_error_non_parametric(
                    batch_human_dev, batch_comet_scores_dev, scaling_vals=np_scaling_vals, scaling_sums=np_scaling_sums)
                np_calibration_error, np_gammas, np_matches = compute_calibration_error_non_parametric(
                    batch_human_test, batch_comet_scores_test, scaling_val=best_scale_val, scaling_sum=best_scale_sum)
                print("Non-parametric CE = %f (calibrated, best_scaling_val=%f, best_scaling_sum=%f)" %
                    (np_calibration_error, best_scale_val, best_scale_sum))

                np_sharpness_cal = compute_epiw(batch_comet_scores_test, batch_comet_avg_test, batch_comet_std_test, std_sum=best_scale_sum, std_scale=best_scale_val)
                print("Non-parametric Sharpness = %f (calibrated, best_scaling_val=%f, best_scaling_sum=%f)" %
                    (np_sharpness_cal, best_scale_val, best_scale_sum))

                ############# LATEX #################
                print()
                print('----------LATEX OUTPUTS----------')
                print('& average NLL & ECE & Sharpness \\\\')
                print('& %f & %f & %f  \\\\' % (cal_avgll, calibration_error, sharpness_cal))
                print('& r(human, pred) & r(|pred-human|,std) \\\\')
                print('& %f & %f   \\\\' % (pearson_acc, pearson_d1_cal))
                # print('& ECE_np & Sharpness_np \\\\')
                # print('& %f & %f   \\\\' % (np_calibration_error, np_sharpness_cal))
        
                cal_avgll_folds.append(cal_avgll)
                calibration_error_folds.append(calibration_error)
                sharpness_cal_folds.append(sharpness_cal)
                pearson_acc_folds.append(pearson_acc)
                pearson_d1_cal_folds.append(pearson_d1_cal)
                # np_calibration_error_folds.append(np_calibration_error)
                # np_sharpness_cal_folds.append(np_sharpness_cal)

    if args.baseline:
        print()
        print('------AVERAGED OVER k FOLDS------')
        print('----------LATEX OUTPUTS----------')
        print('& average NLL & ECE & Sharpness \\\\')
        print('& %.3f & %.3f & %.3f  \\\\' % (round(np.mean(cal_avgll_folds),3), round(np.mean(calibration_error_folds),3), round(np.mean(sharpness_cal_folds),3)))
    else:
        print()
        print('------AVERAGED OVER k FOLDS------')
        print('----------LATEX OUTPUTS----------')
        print('& average NLL & ECE & Sharpness \\\\')
        print('& %.3f & %.3f & %.3f  \\\\' % (round(np.mean(cal_avgll_folds),3), round(np.mean(calibration_error_folds),3), round(np.mean(sharpness_cal_folds),3)))
        print('& r(human, pred) & r(|pred-human|,std) \\\\')
        print('& %.3f & %.3f   \\\\' % (round(np.mean(pearson_acc_folds),3), round(np.mean(pearson_d1_cal_folds),3)))
        # print('& ECE_np & Sharpness_np \\\\')
        # print('& %.3f & %.3f   \\\\' % (round(np.mean(np_calibration_error_folds),3), round(np.mean(np_sharpness_cal_folds),3)))

        ################## FIGURES #################
        # plt.xlabel('Confidence level $\gamma$')
        # plt.ylabel('ECE')
        # plt.title(args.score_type.upper() + ': 1719 on '+test_year+' - Batch size = %d' % batch_size)
        # plt.plot(gammas, matches, 'b', label="Original ECE")
        # plt.plot(gammas, matches_cal, 'r', label="Calibrated ECE")
        # plt.plot(gammas, base_matches, 'g', label="Baseline ECE")
        # #plt.plot(gammas, mcpe_matches, 'b:', label="Original MPE")
        # #plt.plot(gammas, mcpe_matches_cal, 'r:', label="Calibrated MPE")
        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.legend()
        # plt.show()
        # plt.savefig('figures/'+args.score_type.upper()+'_1719-'+test_year+'-ECE_bs_'+str(batch_size)+'.png')
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


