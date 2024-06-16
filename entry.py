import mlflow
from azureml.core import Workspace
from azureml.core import Run

from pathlib import Path
import argparse
import copy
import json

import pandas as pd
import numpy as np
np.random.seed(2048)

import seaborn as sns
sns.reset_defaults()
sns.set_context(context="talk", font_scale=1)
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import torch
#torch.set_printoptions(sci_mode=False)
torch.manual_seed(42)
from torch import nn
import torch.nn.functional as F
import pytorch_warmup as warmup

from utils import (
    TokenPlacement,
    InstancingType,
    BatchLogger,
    MLFlowDummy,
    df_to_ix_tensors,
    diversity_score,
    dropout_activation,
    extract_possible_values, 
    generated_to_data_ix_tensors,
    get_target_tensor,
    get_torch_device,
    filter_by_impressions,
    calculate_cosine_similarity,
    calculate_hamming_distance,
    build_generated_df,
    save_fig_json_png,
)

from estimator import ipinyouPriceMdeler
from generator import (
    ASGAR, 
    make_ASGAR,
    L, 
    score_positiveness,
    save_loss_surfaces_plot,
    positive_curve,
)
# from aligner import CLIPModel, CLIPCosineLoss
from aligner import TripletModel, AlignerLoss

device = get_torch_device()

def main(args, mlflow):

    current_run = Run.get_context()
    ws = current_run.experiment.workspace

    print(f'Torch Device {device}')
    # ----------------------------------------- Logging args as job parameters -----------------------------------------
    for arg_name, arg_value in vars(args).items():
        if arg_name == 'data':
            continue
        mlflow.log_param(arg_name, arg_value)
    
    _, plot_path = save_loss_surfaces_plot(args, 'loss_curves.html')
    mlflow.log_artifact(plot_path)
        
    # ----------------------------------------- read data -----------------------------------------
    print('Preparing Data...')
    import glob
    train_path = glob.glob(args.data+'/train_*.csv')[0]
    validation_path = glob.glob(args.data+'/validation_*.csv')[0]
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    datas = {'train':train, 'validation':validation}


    # ----------------------------------------- features -----------------------------------------
    strategy_columns = args.strategy_columns 
    context_columns = args.context_columns
    performance_column = args.score_column


    # ----------------------------------------- data preparation -----------------------------------------
    # region data preparation
    norm_functs = {
        'log': {'normalize': lambda x: np.log(x), 'denormalize': lambda x: torch.exp(x)},
        'exp': {'normalize': lambda x: np.exp(x), 'denormalize': lambda x: torch.log(x)},
        'scaled_log': {'normalize': lambda x: 10*np.log(x), 'denormalize': lambda x: torch.exp(x/10)},
        'shifted_log': {'normalize': lambda x: 10+np.log(x), 'denormalize': lambda x: torch.exp(x-10)},
        'min_max': {'normalize': lambda x: (x-args.score_min)/(args.score_max-args.score_min), 'denormalize': lambda x: x*(args.score_max-args.score_min)+args.score_min},
        'sqrt': {'normalize': lambda x: np.sqrt(x), 'denormalize': lambda x: torch.power(x,2)},
        'none': {'normalize': lambda x: x, 'denormalize': lambda x: x},
    }

    normalizers = {
        'score': norm_functs[args.normalize_score]['normalize']
    }

    denormalizers = {
        'score': norm_functs[args.normalize_score]['denormalize']
    }

    for df_title in datas:
        df = datas[df_title]
        df[context_columns + strategy_columns] = df[context_columns + strategy_columns].astype(str)
        data = df[context_columns + strategy_columns + [performance_column]].copy()
        
        data.rename(columns={col_name: col_name+'_context' for col_name in context_columns }, inplace=True)
        data.rename(columns={performance_column: 'score'}, inplace=True)

        for col in data:
            if data[col].dtype=="object":
                data.fillna({col:''},inplace=True)
            else:
                data.fillna({col:0},inplace=True)
        datas[df_title] = data

    context_columns = [col_name+'_context' for col_name in context_columns]
    train, validation = datas['train'], datas['validation']
    del datas

    # score filtering
    if args.filter_score:
        train = train[(train['score']>=args.score_min) & (train['score']<=args.score_max)].copy()
        validation = validation[(validation['score']>=args.score_min) & (validation['score']<=args.score_max)].copy()

    features = extract_possible_values(pd.concat([train, validation]))
    torch.save(features, "features")
    mlflow.log_artifact('features', '.')

    feature_cols = features['context_features']['features_order'] + features['strategy_features']['features_order']

    if args.groupby_features:
        train = train.groupby(feature_cols).mean().reset_index()
        validation = validation.groupby(feature_cols).mean().reset_index()

    stats = train.groupby(features['context_features']['features_order']).agg(
                min=pd.NamedAgg(column='score', aggfunc='min'), 
                mean=pd.NamedAgg(column='score', aggfunc='mean'),
                max=pd.NamedAgg(column='score', aggfunc='max')
            ) 
    
    minimize_mode = True if args.min_anchor_positiveness > 0.50 else False

    # context balancing
    max_count = train.groupby(context_columns).count()['score'].max()
    subsets = list()
    for group, idx in train.reset_index(drop=True).groupby(context_columns).groups.items():
        subset = train.iloc[idx].copy()
        subset.loc[:, 'is_better'] = (subset['score'] >= stats.loc[group]['mean']).values
        if minimize_mode:
            subset['is_better'] = ~subset['is_better']
        subsets.append(subset)
        positives = subset[subset.is_better]
        positives = [positives]*(max_count//len(positives))
        subsets += positives
    train = pd.concat(subsets).sample(frac=1)

    max_count = validation.groupby(context_columns).count()['score'].max()
    subsets = list()
    for group, idx in validation.reset_index(drop=True).groupby(context_columns).groups.items():
        subset = validation.iloc[idx].copy()
        subset.loc[:, 'is_better'] = (subset['score'] > stats.loc[group]['mean']).values
        if minimize_mode:
            subset['is_better'] = ~subset['is_better']
        subsets.append(subset)
        positives = subset[subset.is_better]
        positives = [positives]*(max_count//len(positives))
        subsets += positives
    validation = pd.concat(subsets).sample(frac=1)

    print('Train Shape', train.shape)
    print('Validation Shape', validation.shape)
    # endregion

    # ----------------------------------------- Estimator loading -----------------------------------------
    # region Estimator Loading
    if args.embeddings_from_estimator == 1:
        assert args.estimator_emb_dim == args.emb_size

    uri = args.estimator_model_path
    run_id = uri.split('dcid.')[1].split('/')[0]
    run = ws.get_run(run_id)
    file_to_download = uri.split('dcid.')[1][len(run_id)+1:]
    run.download_file(file_to_download, 'estimator_model')
    mlflow.log_artifact('estimator_model', '.')

    strategy_estimator = ipinyouPriceMdeler(features, args.estimator_emb_dim, args.estimator_latent_dim, normalizers, denormalizers, output_size=1).to(device)
    strategy_estimator.load_state_dict(torch.load('estimator_model', map_location=device))
    strategy_estimator.requires_grad_(False)
    strategy_estimator.eval()

    # endregion


    # ----------------------------------------- Aligner loading -----------------------------------------
    # region Aligner loading
    uri = args.aligner_model_path
    run_id = uri.split('dcid.')[1].split('/')[0]
    run = ws.get_run(run_id)
    file_to_download = uri.split('dcid.')[1][len(run_id)+1:]
    run.download_file(file_to_download, 'aligner_model')
    mlflow.log_artifact('aligner_model', '.')

    aligner_model = TripletModel(strategy_estimator, strategy_embedding=args.estimator_latent_dim, projection_dim=args.aligner_projection_dim)
    aligner_model.load_state_dict(torch.load('aligner_model', map_location=device))
    aligner_model.requires_grad_(False)
    aligner_loss = AlignerLoss(aligner_model, args.cosine_weight)
    # endregion

    # ----------------------------------------- Training Loop -----------------------------------------
    #LOSS
    # criterions = [nn.CrossEntropyLoss(reduction='none') for i in range(len(features['strategy_features']['features_order']))]
    # mtlwrapper = MultiTaskLossWrapper(task_num=2, reduction='none')
    
    # region Params
    min_anchor_positiveness = args.min_anchor_positiveness
    max_anchor_positiveness = None
    sampling = False
    context_attn_factor = 1
    train_df = train
    validation_df = validation

    # endregion

    # region Initialization

    continue_training = False
    if continue_training:
        print('Continuing training from last training !')
    else:
        model, optimizer, scheduler = make_ASGAR(args, strategy_estimator, aligner_model, features)

    # just init model, this is for local testing later
    if args.just_init_model:
        return locals()

    schedule_ce_combine = list()
    for sched in args.aligner_schedule.split(','):
        ce_combine, epoch_switch = sched.split(':')
        if epoch_switch == 'end':
            break
        schedule_ce_combine += [ce_combine]*(int(epoch_switch)-len(schedule_ce_combine))
    schedule_ce_combine += [ce_combine]*(int(args.epochs)-len(schedule_ce_combine))

    if -1.1 < args.masking_rate and args.masking_rate < 0:
        # masking_rate == -1.0: epoch level random rate
        masks = np.random.uniform(size=args.epochs)
    else:
        # masking_rate == -2.0: batch level random rate
        # OR
        # just forward a fixed masking rate in range [0,1.0]
        masks = [args.masking_rate]*args.epochs
        
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=args.warmup_period)

    best_val_model = None
    best_train_model = None
    last_model = None
    best_val_loss = np.inf
    best_train_loss = np.inf
    state_dicts = list()
    bl = BatchLogger()

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    Path("plots").mkdir(parents=True, exist_ok=True)

    # endregion

    print('Started Training Loop')
    
    # region Training Loop
    
    for epoch in range(args.epochs):
        args.ce_combine = schedule_ce_combine[epoch]
        current_masking_rate = masks[epoch]

        # region Train
        model.train()
        for i in range(0, len(train_df), args.batch_size):
            optimizer.zero_grad()
            
            batch_data = train_df[i: i+args.batch_size]

            data_ix_tensors = df_to_ix_tensors(batch_data, features)
            decoder_target_tensor = get_target_tensor(data_ix_tensors, features)

            score_tensor = torch.tensor(batch_data['score'].values, dtype=torch.float, device=device)
            p = score_positiveness(batch_data, score_tensor, stats, features,
                                min_anchor_positiveness=min_anchor_positiveness, max_anchor_positiveness=max_anchor_positiveness, c_scaling=args.c_scaling)

            output, variational_mean, variational_log_var, attentions = model(data_ix_tensors, p=p, candidating=args.candidating, masking_rate=current_masking_rate)
                
            diversity = 0
            reconstruction_loss = 0
            for feature_ix, strategy_feature in enumerate(features['strategy_features']['features_order']):
                predicted_distribution = output[strategy_feature]/args.temperature
                diversity += diversity_score(predicted_distribution.detach())
                if args.ce_combine:
                    n_classes = len(features['strategy_features']['features'][strategy_feature]['values'])
                    target_class = decoder_target_tensor[:,feature_ix]
                    # soft_tgt = F.softmax((1-F.one_hot(target_class, num_classes=n_classes).float())*10, dim=-1)
                    soft_tgt = 1-F.one_hot(target_class, num_classes=n_classes).float()
                    # soft_tgt = (1-F.one_hot(target_class, num_classes=n_classes).float())*(soft_tgt_ref[strategy_feature].detach().softmax(-1))
                    soft_tgt = soft_tgt/soft_tgt.sum(-1).unsqueeze(1)

                    old_loss = F.cross_entropy(predicted_distribution, target_class, reduction='none')
                    new_loss = F.cross_entropy(predicted_distribution, soft_tgt, reduction='none')
                    
                    mix_loss = old_loss
                    if epoch >= args.avoid_epoch:
                        mix_loss[p<0.75] = new_loss[p<0.75]
                    reconstruction_loss += mix_loss

            if args.ce_combine == 'mean':
                reconstruction_loss = reconstruction_loss/len(features['strategy_features']['features_order'])
                reconstruction_loss += aligner_loss.from_logits(output, data_ix_tensors, softmax_weight=args.softmax_weight).flatten()
                reconstruction_loss = reconstruction_loss/2
            elif args.ce_combine == 'weight':
                reconstruction_loss = reconstruction_loss/len(features['strategy_features']['features_order'])
                reconstruction_loss += args.aligner_weight*aligner_loss.from_logits(output, data_ix_tensors, softmax_weight=args.softmax_weight).flatten()
            elif args.ce_combine == 'product':
                reconstruction_loss = reconstruction_loss/len(features['strategy_features']['features_order'])
                reconstruction_loss *= aligner_loss.from_logits(output, data_ix_tensors, softmax_weight=args.softmax_weight).flatten()
            elif args.ce_combine == 'ce_only':
                reconstruction_loss = reconstruction_loss/len(features['strategy_features']['features_order'])
            elif args.ce_combine == 'aligner_only':
                reconstruction_loss = aligner_loss.from_logits(output, data_ix_tensors, softmax_weight=args.softmax_weight).flatten()


            train_aligner_loss = aligner_loss.from_logits(output, data_ix_tensors, softmax_weight=args.softmax_weight).flatten()

            pos_mask = (p>=0.75)
            neut_mask = (p>=0.25)&(p<0.75)
            neg_mask = p<0.25

            bl.accumulate_batch_list('train_pos_rec_losses', reconstruction_loss[pos_mask])
            bl.accumulate_batch_list('train_neut_rec_losses', reconstruction_loss[neut_mask])
            bl.accumulate_batch_list('train_neg_rec_losses', reconstruction_loss[neg_mask])
            
            bl.accumulate_batch_list('train_pos_aligner_losses', train_aligner_loss[pos_mask])
            bl.accumulate_batch_list('train_neut_aligner_losses', train_aligner_loss[neut_mask])
            bl.accumulate_batch_list('train_neg_aligner_losses', train_aligner_loss[neg_mask])
            
            generated_data_ix_tensors, _ = generated_to_data_ix_tensors(output, data_ix_tensors, features, sampling=sampling)

            # estimator_output = strategy_estimator(generated_data_ix_tensors)
            # estimator_output = F.softplus(strategy_estimator(generated_data_ix_tensors, from_logits=True, logits_dict=output, softmax_weight=args.softmax_weight))
            estimator_output = strategy_estimator(generated_data_ix_tensors, from_logits=True, logits_dict=output, softmax_weight=args.softmax_weight)
            estimated_scores = denormalizers["score"](estimator_output).flatten()
            
            estimated_score_loss = 1 - score_positiveness(batch_data, estimated_scores, stats, features,
                                                        min_anchor_positiveness=min_anchor_positiveness, max_anchor_positiveness=max_anchor_positiveness, c_scaling=args.c_scaling)
            
            bl.accumulate_batch_list('train_pos_gen_positiveness', (1 - estimated_score_loss[pos_mask]))
            bl.accumulate_batch_list('train_neut_gen_positiveness', (1 - estimated_score_loss[neut_mask]))
            bl.accumulate_batch_list('train_neg_gen_positiveness', (1 - estimated_score_loss[neg_mask]))
            
            # KLD = KLD.mean(1)
            if epoch >= args.avoid_epoch and args.ce_combine != 'aligner_only':
                generative_loss = positive_curve(reconstruction_loss, estimated_score_loss, eiw=args.peiw, dfw=args.pdfw)
                generative_loss[p<0.25] = positive_curve(reconstruction_loss[p<0.25], estimated_score_loss[p<0.25], eiw=0.2, dfw=0.1)
                generative_loss[(p<0.75)&(p>=0.25)] = positive_curve(reconstruction_loss[(p<0.75)&(p>=0.25)], estimated_score_loss[(p<0.75)&(p>=0.25)], eiw=0.2, dfw=0.07)
            else:
                generative_loss = L(reconstruction_loss, estimated_score_loss, p, peiw=args.peiw, pdfw=args.pdfw, aeiw=args.aeiw, aaw=args.aaw, nfw=args.nfw, new=args.new, interpolating=args.interpolating)

            if args.variational:
                variational_mean = variational_mean.view(generative_loss.size(0), -1)
                variational_log_var = variational_log_var.view(generative_loss.size(0), -1)
                KLD = - 0.5 * torch.sum(1+ variational_log_var - variational_mean.pow(2) - variational_log_var.exp(), dim=-1)
                loss_formula = generative_loss + args.kl_weight*KLD
            else:
                loss_formula = generative_loss

            if not args.encoder_only:
                model.first_print('Constrained Decoder Quantization Attention')
                decoder_q_attn_loss = 0
                for _, attn_l in enumerate(attentions['decoder_self_attention']):
                    decoder_q_attn_loss += torch.clamp(attn_l[:, :,-1].mean(-1) - args.attn_threshold, 0)
                loss_formula += decoder_q_attn_loss
            
            if args.encoder_context_attn_forcing and (not args.use_special_embedding):
                model.first_print('Constrained Encoder Context Attention')
                encoder_attnl_loss = 0
                for _, attn_l in enumerate(attentions['encoder_attention']):
                    context_attn_threshold = context_attn_factor/attn_l.shape[-1]
                    # context_attn_weights = attn_l[:,:,[0,1]].sum(-1).max(-1).values
                    context_attn_weights = attn_l[:,:,[0,1]].mean(-1).mean(-1)
                    encoder_attnl_loss += torch.clamp(context_attn_weights - context_attn_threshold, 0)
                loss_formula += encoder_attnl_loss
            
            # TO REMOVE LATEER !
            # loss_formula += div_loss

            bl.accumulate_batch_list('train_loss', loss_formula)
            bl.accumulate_batch_list('train_estimated_scores', estimated_scores)
            bl.accumulate_batch_list('train_estimated_score_loss', estimated_score_loss)

            batch_loss = torch.mean(loss_formula)
            batch_loss.backward()
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # torch.nn.utils.clip_grad_value_(params, clip_value=5.0)
            optimizer.step()
            # freeing gpu memory
            del output, loss_formula, data_ix_tensors, score_tensor, estimated_scores, estimator_output, estimated_score_loss
            del reconstruction_loss, generative_loss, batch_loss, p, predicted_distribution, generated_data_ix_tensors

        bl.summarize_epoch()
        # endregion

        # region Validation 
        generated = None
        df = validation_df
        with torch.no_grad():
            total_loss = 0
            total_score = 0
            model.eval()

            validation_batch_size = 512
            for i in range(0, len(df), validation_batch_size):
                batch_data = df[i: i+validation_batch_size]

                data_ix_tensors = df_to_ix_tensors(batch_data, features)
                decoder_target_tensor = get_target_tensor(data_ix_tensors, features)

                score_tensor = torch.tensor(batch_data['score'].values, dtype=torch.float, device=device)
                p = score_positiveness(batch_data, score_tensor, stats, features,
                                    min_anchor_positiveness=min_anchor_positiveness, max_anchor_positiveness=max_anchor_positiveness, c_scaling=args.c_scaling)

                output, variational_mean, variational_log_var, attentions = model(data_ix_tensors, p=None)
                
                diversity = 0
                reconstruction_loss = 0
                for feature_ix, strategy_feature in enumerate(features['strategy_features']['features_order']):
                    predicted_distribution = output[strategy_feature]/args.temperature
                    diversity += diversity_score(predicted_distribution.detach())
                    if args.ce_combine:
                        target_class = decoder_target_tensor[:,feature_ix]
                        reconstruction_loss += F.cross_entropy(predicted_distribution, target_class, reduction='none')

                if args.ce_combine == 'mean':
                    reconstruction_loss = reconstruction_loss/len(features['strategy_features']['features_order'])
                    reconstruction_loss += aligner_loss.from_logits(output, data_ix_tensors, softmax_weight=args.softmax_weight).flatten()
                    reconstruction_loss = reconstruction_loss/2
                elif args.ce_combine == 'weight':
                    reconstruction_loss = reconstruction_loss/len(features['strategy_features']['features_order'])
                    reconstruction_loss += args.aligner_weight*aligner_loss.from_logits(output, data_ix_tensors, softmax_weight=args.softmax_weight).flatten()
                elif args.ce_combine == 'product':
                    reconstruction_loss = reconstruction_loss/len(features['strategy_features']['features_order'])
                    reconstruction_loss *= aligner_loss.from_logits(output, data_ix_tensors, softmax_weight=args.softmax_weight).flatten()
                elif args.ce_combine == 'ce_only':
                    reconstruction_loss = reconstruction_loss/len(features['strategy_features']['features_order'])
                elif args.ce_combine == 'aligner_only':
                    reconstruction_loss = aligner_loss.from_logits(output, data_ix_tensors, softmax_weight=args.softmax_weight).flatten()

                validation_aligner_loss = aligner_loss.from_logits(output, data_ix_tensors, softmax_weight=args.softmax_weight).flatten()

                pos_mask = p>=0.75
                neut_mask = (p>=0.25)&(p<0.75)
                neg_mask = p<0.25

                bl.accumulate_batch_list('validation_pos_aligner_losses', validation_aligner_loss[pos_mask])
                bl.accumulate_batch_list('validation_neut_aligner_losses', validation_aligner_loss[neut_mask])
                bl.accumulate_batch_list('validation_neg_aligner_losses', validation_aligner_loss[neg_mask])
                
                bl.accumulate_batch_list('validation_pos_rec_losses', reconstruction_loss[pos_mask])
                bl.accumulate_batch_list('validation_neut_rec_losses', reconstruction_loss[neut_mask])
                bl.accumulate_batch_list('validation_neg_rec_losses', reconstruction_loss[neg_mask])

                generated_data_ix_tensors, confidences = generated_to_data_ix_tensors(output, data_ix_tensors, features, sampling=sampling)

                # estimator_output = F.softplus(strategy_estimator(generated_data_ix_tensors))
                estimator_output = strategy_estimator(generated_data_ix_tensors)
                estimated_scores = denormalizers["score"](estimator_output).flatten()
                estimated_score_loss = 1 - score_positiveness(batch_data, estimated_scores, stats, features,
                                                            min_anchor_positiveness=min_anchor_positiveness, max_anchor_positiveness=max_anchor_positiveness, c_scaling=args.c_scaling)
                
                bl.accumulate_batch_list('validation_pos_gen_positiveness', (1 - estimated_score_loss[pos_mask]))
                bl.accumulate_batch_list('validation_neut_gen_positiveness', (1 - estimated_score_loss[neut_mask]))
                bl.accumulate_batch_list('validation_neg_gen_positiveness', (1 - estimated_score_loss[neg_mask]))

                bl.accumulate_batch_list('validation_estimated_scores', estimated_scores)
                bl.accumulate_batch_list('validation_diversity', diversity)
                bl.accumulate_batch_list('validation_reconstruction_loss', reconstruction_loss)
                
                generated = build_generated_df(generated, output, batch_data, estimated_scores, estimated_score_loss, features)

            bl.summarize_epoch()

            generated = generated.set_index(features['context_features']['features_order'])

            bl.log_one_value('validation_mean_hammings', calculate_hamming_distance(generated, df, features)[0].item())
            bl.log_one_value('validation_mean_cosines', calculate_cosine_similarity(generated, df, features, aligner_model)[0].item())
            bl.log_one_value('validation_generated_counts', generated.shape[0])
            bl.log_one_value('validation_generated_proportions', generated.shape[0]/df.groupby(feature_cols).first().shape[0])


        generated = None
        df = train_df[~train_df.is_better]
        with torch.no_grad():
            total_loss = 0
            total_score = 0
            model.eval()

            validation_batch_size = 512
            for i in range(0, len(df), validation_batch_size):
                batch_data = df[i: i+validation_batch_size]

                data_ix_tensors = df_to_ix_tensors(batch_data, features)
                decoder_target_tensor = get_target_tensor(data_ix_tensors, features)

                score_tensor = torch.tensor(batch_data['score'].values, dtype=torch.float, device=device)
                p = score_positiveness(batch_data, score_tensor, stats, features,
                                    min_anchor_positiveness=min_anchor_positiveness, max_anchor_positiveness=max_anchor_positiveness, c_scaling=args.c_scaling)

                output, variational_mean, variational_log_var, attentions = model(data_ix_tensors, p=p, guiding='avoid')
                
                diversity = 0
                reconstruction_loss = 0
                for feature_ix, strategy_feature in enumerate(features['strategy_features']['features_order']):
                    predicted_distribution = output[strategy_feature]/args.temperature
                    diversity += diversity_score(predicted_distribution.detach())
                    if args.ce_combine:
                        target_class = decoder_target_tensor[:,feature_ix]
                        reconstruction_loss += F.cross_entropy(predicted_distribution, target_class, reduction='none')

                if args.ce_combine == 'mean':
                    reconstruction_loss = reconstruction_loss/len(features['strategy_features']['features_order'])
                    reconstruction_loss += aligner_loss.from_logits(output, data_ix_tensors, softmax_weight=args.softmax_weight).flatten()
                    reconstruction_loss = reconstruction_loss/2
                elif args.ce_combine == 'weight':
                    reconstruction_loss = reconstruction_loss/len(features['strategy_features']['features_order'])
                    reconstruction_loss += args.aligner_weight*aligner_loss.from_logits(output, data_ix_tensors, softmax_weight=args.softmax_weight).flatten()
                elif args.ce_combine == 'product':
                    reconstruction_loss = reconstruction_loss/len(features['strategy_features']['features_order'])
                    reconstruction_loss *= aligner_loss.from_logits(output, data_ix_tensors, softmax_weight=args.softmax_weight).flatten()
                elif args.ce_combine == 'ce_only':
                    reconstruction_loss = reconstruction_loss/len(features['strategy_features']['features_order'])
                elif args.ce_combine == 'aligner_only':
                    reconstruction_loss = aligner_loss.from_logits(output, data_ix_tensors, softmax_weight=args.softmax_weight).flatten()

                generated_data_ix_tensors, confidences = generated_to_data_ix_tensors(output, data_ix_tensors, features, sampling=sampling)

                # estimator_output = F.softplus(strategy_estimator(generated_data_ix_tensors))
                estimator_output = strategy_estimator(generated_data_ix_tensors)
                estimated_scores = denormalizers["score"](estimator_output).flatten()
                estimated_score_loss = 1 - score_positiveness(batch_data, estimated_scores, stats, features,
                                                            min_anchor_positiveness=min_anchor_positiveness, max_anchor_positiveness=max_anchor_positiveness, c_scaling=args.c_scaling)

                generated = build_generated_df(generated, output, batch_data, estimated_scores, estimated_score_loss, features)

            bl.log_one_value('train_avoid_counts', generated.shape[0])
        # endregion
        
        # region Logging
        bl.log_one_value('learning_rate', optimizer.param_groups[0]['lr'])
        print(
            ' | '.join(
                [
                    f"Ep {epoch}",
                    f"Loss: {bl.last('train_loss'):0.2f}",
                    f"Score: {bl.last('train_estimated_scores'):0.2f} val {bl.last('validation_estimated_scores'):0.2f}",
                    f"Pos: {1-bl.last('train_estimated_score_loss'):0.3f} val +{bl.last('validation_pos_gen_positiveness'):0.2f} ~{bl.last('validation_neut_gen_positiveness'):0.2f} -{bl.last('validation_neg_gen_positiveness'):0.2f}",
                    f"Rec: +{bl.last('train_pos_rec_losses'):0.2f} ~{bl.last('train_neut_rec_losses'):0.2f} -{bl.last('train_neg_rec_losses'):0.2f} val +{bl.last('validation_pos_rec_losses'):0.2f} ~{bl.last('validation_neut_rec_losses'):0.2f} -{bl.last('validation_neg_rec_losses'):0.2f}",
                    f"Ham/Cos: val {bl.last('validation_mean_hammings'):0.2f}/{bl.last('validation_mean_cosines'):0.2f}",
                    f"Diversity: val {bl.last('validation_diversity'):0.2f}",
                    f"Count: val {bl.last('validation_generated_counts')}",
                    f"Avoid Effect: {bl.last('train_avoid_counts')}",
                    f"Prop: val {bl.last('validation_generated_proportions'):0.3f}",
                    f"LR {optimizer.param_groups[0]['lr']:0.8f}",
                ]
            )
        )

        current_logs = bl.all_lasts()
        for metric_name, metric_value in current_logs.items():
            mlflow.log_metric(metric_name, metric_value, step=epoch)


        if bl.last('train_loss') < best_train_loss:
            best_train_loss = bl.last('train_loss')
            best_train_model = copy.deepcopy(model.state_dict())
            checkpoint_path = f'checkpoints/ASGAR_epoch_{epoch}'
            torch.save(best_train_model, checkpoint_path)
            mlflow.log_artifact(checkpoint_path, 'checkpoints')
        
        with warmup_scheduler.dampening():
            if epoch > warmup_scheduler.warmup_params[0]['warmup_period']:
                if args.lr_scheduler == 'ReduceLROnPlateau':
                    if scheduler.num_bad_epochs == scheduler.patience and (not scheduler.is_better(bl.last('train_loss'), scheduler.best)): #is about to become bad next epoch
                        print(f'Reducing learning rate and backtracking to the best train model yet with loss {scheduler.best:0.5f}')
                        model.load_state_dict(best_train_model)
                    scheduler.step(bl.last('train_loss'))
                elif args.lr_scheduler in ['CyclicLR']:
                    scheduler.step()

        # last_model = copy.deepcopy(model.state_dict())
        # endregion

        # region Plots
        if bl.last('train_loss') <= best_train_loss:
            metrics = bl.main_lists

            # Scores and Generated Count
            plot_path = f'plots/scores_and_count_{epoch}'
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(y=metrics['train_estimated_scores'], name='Train', line=dict(color='royalblue', width=2)))
            fig.add_trace(go.Scatter(y=metrics['validation_estimated_scores'], name='Validation', line=dict(color='darkorange', width=2)))
            fig.add_trace(go.Scatter(y=metrics['validation_generated_counts'], name='Count', line=dict(color='black', width=1)), secondary_y=True)
            fig.update_layout(title='Generated Score',
                            xaxis_title='Epochs',
                            yaxis_title='Score')
            fig.update_yaxes(rangemode='tozero')
            save_fig_json_png(plot_path, fig, mlflow)

            # Validation Positiveness and Count
            plot_path = f'plots/validation_positiveness_and_count_{epoch}'
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(y=metrics['validation_pos_gen_positiveness'], name='Validation Pos', line=dict(color='forestgreen', width=2)))
            fig.add_trace(go.Scatter(y=metrics['validation_neut_gen_positiveness'], name='Validation Neut', line=dict(color='slategray', width=2)))
            fig.add_trace(go.Scatter(y=metrics['validation_neg_gen_positiveness'], name='Validation Neg', line=dict(color='lightcoral', width=2)))
            fig.add_trace(go.Scatter(y=metrics['validation_generated_counts'], name='Count', line=dict(color='black', width=1)), secondary_y=True)
            fig.update_layout(title='Validation Generated Positiveness',
                            xaxis_title='Epochs',
                            yaxis_title='Positiveness')
            fig.update_yaxes(rangemode='tozero')
            save_fig_json_png(plot_path, fig, mlflow)

            # Train Reconstruction Losses
            plot_path = f'plots/train_reconstruction_{epoch}'
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=metrics['train_pos_rec_losses'], name='Pos', line=dict(color='forestgreen', width=2)))
            fig.add_trace(go.Scatter(y=metrics['train_neut_rec_losses'], name='Neut', line=dict(color='slategray', width=2)))
            fig.add_trace(go.Scatter(y=metrics['train_neg_rec_losses'], name='Neg', line=dict(color='lightcoral', width=2)))
            fig.update_layout(title='Train Reconstruction',
                            xaxis_title='Epochs',
                            yaxis_title='Train Reconstruction Loss')
            save_fig_json_png(plot_path, fig, mlflow)

            # Validation Reconstruction Losses
            plot_path = f'plots/validation_reconstruction_{epoch}'
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=metrics['validation_pos_rec_losses'], name='Pos', line=dict(color='forestgreen', width=2)))
            fig.add_trace(go.Scatter(y=metrics['validation_neut_rec_losses'], name='Neut', line=dict(color='slategray', width=2)))
            fig.add_trace(go.Scatter(y=metrics['validation_neg_rec_losses'], name='Neg', line=dict(color='lightcoral', width=2)))
            fig.update_layout(title='Validation Reconstruction',
                            xaxis_title='Epochs',
                            yaxis_title='Validation Reconstruction Loss')
            save_fig_json_png(plot_path, fig, mlflow)

            # Train Aligner Losses
            plot_path = f'plots/train_aligner_{epoch}'
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=metrics['train_pos_aligner_losses'], name='Pos', line=dict(color='forestgreen', width=2)))
            fig.add_trace(go.Scatter(y=metrics['train_neut_aligner_losses'], name='Neut', line=dict(color='slategray', width=2)))
            fig.add_trace(go.Scatter(y=metrics['train_neg_aligner_losses'], name='Neg', line=dict(color='lightcoral', width=2)))
            fig.update_layout(title='Aligner Losses',
                            xaxis_title='Epochs',
                            yaxis_title='Aligner Loss')
            save_fig_json_png(plot_path, fig, mlflow)

            # Validation Aligner Losses
            plot_path = f'plots/validation_aligner_{epoch}'
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=metrics['validation_pos_aligner_losses'], name='Pos', line=dict(color='forestgreen', width=2)))
            fig.add_trace(go.Scatter(y=metrics['validation_neut_aligner_losses'], name='Neut', line=dict(color='slategray', width=2)))
            fig.add_trace(go.Scatter(y=metrics['validation_neg_aligner_losses'], name='Neg', line=dict(color='lightcoral', width=2)))
            fig.update_layout(title='Aligner Losses',
                            xaxis_title='Epochs',
                            yaxis_title='Aligner Loss')
            save_fig_json_png(plot_path, fig, mlflow)

            # Positives Reconstruction Losses
            plot_path = f'plots/positives_reconstruction_{epoch}'
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=metrics['train_pos_rec_losses'], name='Train', line=dict(color='royalblue', width=2)))
            fig.add_trace(go.Scatter(y=metrics['validation_pos_rec_losses'], name='Validation', line=dict(color='darkorange', width=2)))
            fig.update_layout(title='Reconstruction for Positive Strategies',
                            xaxis_title='Epochs',
                            yaxis_title='Reconstruction Loss')
            save_fig_json_png(plot_path, fig, mlflow)

            # Positives Aligner Losses
            plot_path = f'plots/positives_aligner_{epoch}'
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=metrics['train_pos_aligner_losses'], name='Train', line=dict(color='royalblue', width=2)))
            fig.add_trace(go.Scatter(y=metrics['validation_pos_aligner_losses'], name='Validation', line=dict(color='darkorange', width=2)))
            fig.update_layout(title='Aligner Loss for Positive Strategies',
                            xaxis_title='Epochs',
                            yaxis_title='Aligner Loss')
            save_fig_json_png(plot_path, fig, mlflow)

            # Validation Overall
            plot_path = f'plots/validation_overall_{epoch}'
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(y=metrics['validation_pos_gen_positiveness'], name='Pos Positiveness', line=dict(color='forestgreen', width=2)))
            fig.add_trace(go.Scatter(y=metrics['validation_neut_gen_positiveness'], name='Neut Positivness', line=dict(color='slategray', width=2)))
            fig.add_trace(go.Scatter(y=metrics['validation_neg_gen_positiveness'], name='Neg Positivness', line=dict(color='lightcoral', width=2)))
            fig.add_trace(go.Scatter(y=metrics['validation_diversity'], name='Diversity', line=dict(color='darkorange', width=2)))
            fig.add_trace(go.Scatter(y=metrics['validation_generated_proportions'], name='Proportion', line=dict(color='salmon', width=2)))
            fig.add_trace(go.Scatter(y=metrics['validation_generated_counts'], name='Count', line=dict(color='black', width=1)), secondary_y=True)
            fig.update_layout(title='Validation Generated Positiveness / Diversity',
                            xaxis_title='Epochs',
                            yaxis_title='Positiveness')
            save_fig_json_png(plot_path, fig, mlflow)

            # Validation Cosine Hamming
            plot_path = f'plots/validation_cosine_hamming_{epoch}'
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(y=metrics['validation_mean_cosines'], name='Cosine Similarity', line=dict(color='mediumpurple', width=2)))
            fig.add_trace(go.Scatter(y=metrics['validation_mean_hammings'], name='Hamming Distance', line=dict(color='salmon', width=2)))
            fig.add_trace(go.Scatter(y=metrics['validation_generated_counts'], name='Count', line=dict(color='grey', width=1)), secondary_y=True)
            fig.update_layout(title='Validation Similarity Metrics',
                            xaxis_title='Epochs',
                            yaxis_title='Value',)
            save_fig_json_png(plot_path, fig, mlflow)
        # endregion
    # endregion


def get_argparser():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--data", type=str)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--lr_scheduler", type=str, default='None')
    parser.add_argument("--base_lr", type=float, default=0.001)
    parser.add_argument("--max_lr", type=float, default=0.001)
    parser.add_argument("--lr_scheduler_step_size", type=int, default=10)
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_patience", type=int, default=10)
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--type_embedding_weight', type=float, default=1.0)
    parser.add_argument("--transformer_layers", type=int, default=3)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--kl_weight', type=float, default=1.0)
    parser.add_argument('--mtlw', type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--peiw', type=float)
    parser.add_argument('--pdfw', type=float)
    parser.add_argument('--aeiw', type=float)
    parser.add_argument('--aaw', type=float)
    parser.add_argument('--nfw', type=float)
    parser.add_argument('--new', type=float)
    parser.add_argument('--c_scaling', type=float, default=1.0)
    parser.add_argument('--min_anchor_positiveness', type=float, default=0.02)
    parser.add_argument('--interpolating', type=int)
    parser.add_argument('--candidating', type=int)
    parser.add_argument('--candidating_members', type=int, default=1)
    parser.add_argument('--ce_combine', type=str)
    parser.add_argument('--embeddings_from_estimator', type=int, default=0)
    parser.add_argument('--softmax_weight', type=float, default=10.0)
    parser.add_argument('--cosine_weight', type=float, default=4.0)
    parser.add_argument('--wide_ffns', type=int)
    parser.add_argument('--variational', type=str)
    parser.add_argument('--context_placement', type=str)
    parser.add_argument('--quantized_placement', type=str)
    parser.add_argument('--instancing_type', type=str)
    parser.add_argument('--fsq_levels', type=lambda x: list(map(int,x.split(','))), default='8,6,5')
    parser.add_argument('--n_quantized_tokens', type=int)
    parser.add_argument('--context_decoder_mask', type=int, default=1)
    parser.add_argument('--use_aligner_latent', type=int)

    parser.add_argument("--encoder_only", type=int, default=0)
    parser.add_argument("--use_special_embedding", type=int, default=0)
    parser.add_argument("--temperature", type=int, default=1)
    parser.add_argument("--cross_attn_mask_encoder_context", type=int, default=1)
    parser.add_argument("--cross_attn_mask_encoder_instance_q", type=int, default=0)
    parser.add_argument("--use_feature_type_memory_mask", type=int, default=0)
    parser.add_argument('--align_attentions', type=int, default=0)
    parser.add_argument('--encoder_context_attn_forcing', type=int, default=0)
    parser.add_argument('--masking_rate', type=float, default=0.5)
    parser.add_argument('--just_init_model', type=int)

    parser.add_argument('--warmup_period', type=int, default=20)
    parser.add_argument('--attn_threshold', type=float, default=0.2)
    parser.add_argument('--avoid_epoch', type=int, default=0)
    parser.add_argument('--aligner_schedule', type=str, default='ce_only:end')
    parser.add_argument('--aligner_weight', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=100)
    
    # Data Prep
    parser.add_argument("--strategy_columns", type=lambda x: x.split(','), required=True)
    parser.add_argument("--context_columns", type=lambda x: x.split(','), required=True)
    parser.add_argument("--score_column", type=str, required=True)
    parser.add_argument("--score_max", type=float, default=10.0)
    parser.add_argument("--score_min", type=float, default=0.01)
    parser.add_argument("--normalize_score", type=str, default='none')
    parser.add_argument("--filter_score", type=int, default=1)
    parser.add_argument("--groupby_features", type=int, default=1)

    # Estimator & Aligner
    parser.add_argument("--estimator_latent_dim", type=int, default=128)
    parser.add_argument("--estimator_emb_dim", type=int, default=128)
    parser.add_argument("--estimator_model_path", type=str, required=True)
    parser.add_argument("--aligner_projection_dim", type=int, default=128)
    parser.add_argument("--aligner_model_path", type=str, required=True)
    return parser



# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    parser = get_argparser()
    args = parser.parse_args()

    # run main function
    if args.just_init_model:
        local_vars = main(args, MLFlowDummy())
    else:
        local_vars = main(args, mlflow)


    # add space in logs
    print("*" * 60)
    print("\n\n")