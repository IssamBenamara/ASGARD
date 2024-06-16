import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.spatial.distance import hamming

class TokenPlacement:
    ENCODER_DECODER = 'encoder_decoder'
    ENCODER_ONLY = 'encoder_only'
    DECODER_ONLY = 'decoder_only'
    SUMMED_TO_ENCODER = 'summed_to_encoder'
    SUMMED_TO_ENCODER_DECODER = 'summed_to_encoder_decoder'
    NO_PLACE = None

class InstancingType:
    ALIGNER_PROJECTION = 'aligner_projection'
    QUANTIZATION = 'quantization'
    NO_INSTANCING = None

class MLFlowDummy():
    '''Dummy object that can be called with everything.'''
    def __getattribute__(self, name):
        return lambda *args, **kwargs: None

def get_torch_device():
    
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_possible_values(df):

    context_features = dict()
    strategy_features = dict()

    for col in df.columns:
        values = df[col].unique().tolist()
        value_to_ix = {v:i for i,v in enumerate(values)}
        ix_to_value = {i:v for i,v in enumerate(values)}
        values_dict = {
            'values':values,
            'value_to_ix': value_to_ix,
            'ix_to_value': ix_to_value
        }
        if col.endswith('context'):
            context_features[col] = values_dict
        elif col.endswith('score'):
            # Nothing for score
            continue
        else:
            strategy_features[col] = values_dict
    
    result = {'context_features': {'features_order': list(context_features.keys()), 
                                   'features': context_features}, 
              'strategy_features': {'features_order': list(strategy_features.keys()), 
                                   'features': strategy_features}}
    return result


def df_to_ix_tensors(df, features):

    device = get_torch_device()
    result_df = df.copy()
    result = {'context_features': dict(), 'strategy_features': dict()}

    for feature_name in features['context_features']['features_order']:
        v_to_ix_f = np.vectorize(features['context_features']['features'][feature_name]['value_to_ix'].__getitem__)
        result['context_features'][feature_name] = torch.from_numpy(v_to_ix_f(result_df[feature_name].values)).long().to(device).view(-1,1)

    for feature_name in features['strategy_features']['features_order']:
        v_to_ix_f = np.vectorize(features['strategy_features']['features'][feature_name]['value_to_ix'].__getitem__)
        result['strategy_features'][feature_name] = torch.from_numpy(v_to_ix_f(result_df[feature_name].values)).long().to(device).view(-1,1)

    return result


def construct_embedding_module(features, emb_dim=512, positional=False):
    
    item_embeddings = nn.ModuleDict()
    all_features = list()
    for feature_type in features:
        for col in features[feature_type]['features_order']:
            all_features.append(col)
            item_embeddings[col] = nn.Embedding(len(features[feature_type]['features'][col]['values']), emb_dim)

    type_embeddings = nn.Embedding(len(all_features), emb_dim)
    type_to_ix = {v:i for i, v in enumerate(all_features)}
    
    embeddings = nn.ModuleDict({'item_embeddings': item_embeddings,
                             'type_embeddings': type_embeddings})
    return embeddings, type_to_ix


def map_tensor(t, mapping):

    device = get_torch_device()
    palette, key = zip(*sorted(mapping.items()))
    key = torch.tensor(key, device=device)
    palette = torch.tensor(palette, device=device)
    index = torch.bucketize(t, palette)
    return key[index]


def get_target_tensor(data_ix_tensors, features):
    
    decoder_target_tensor = list()
    for col in features['strategy_features']['features_order']:
        decoder_target_tensor.append(data_ix_tensors['strategy_features'][col].view((-1,1)))
    decoder_target_tensor = torch.cat(decoder_target_tensor,1) #shape: (batch_size, sequence_len)
    return decoder_target_tensor


def generated_to_data_ix_tensors(output, data_ix_tensors, features, sampling=False):
    
    generated = {
        'strategy_features': dict(),
        'context_features': dict()
    }
    confidences = 0
    for k in features['strategy_features']['features_order']:
        softmaxes = F.softmax(output[k].detach(), dim=1)
        if sampling:
            value = torch.multinomial(softmaxes, 1, replacement=True)
        else:
            value = softmaxes.argmax(1).unsqueeze(1)
        confidences += torch.take(softmaxes, value)
        generated['strategy_features'][k] = value
    confidences = confidences/len(features['strategy_features']['features_order'])
    generated['context_features'] = data_ix_tensors['context_features']
    return generated, confidences


class MultiTaskLossWrapper(nn.Module):
    
    def __init__(self, task_num, reduction='none'):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.reduction = reduction
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, losses_list):

        loss = 0
        for i, loss_value in enumerate(losses_list):
            precision = torch.exp(-self.log_vars[i])
            diff = loss_value
            loss += precision * diff + self.log_vars[i]

        if self.reduction == 'none':
            return loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def diversity_score(predicted_distribution):

    source_dist = F.softmax(predicted_distribution, 1)
    mean_dist = F.softmax(predicted_distribution,1).mean(0).expand(predicted_distribution.size(0),-1)
    return F.mse_loss(source_dist, mean_dist, reduction='none').mean(1).unsqueeze(1)


def dropout_activation(model, activate_dropout):
    
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Dropout):
            if activate_dropout == 'eval':
                layer.eval()
            if activate_dropout == 'train':
                layer.train()


def filter_by_impressions(df, impression_threshold, feature_cols):
    
    x = df.groupby(feature_cols).agg(
        score=pd.NamedAgg(column='score', aggfunc='mean'), 
        volume=pd.NamedAgg(column='score', aggfunc='count'),
    )
    x = x[x['volume'] >= impression_threshold]
    return x.reset_index().sample(frac=1)


def ix_tensors_to_ix_df(ix_tensors):

    tmp_dict = dict()
    for feature_type, sub_d in ix_tensors.items():
        for feature_name, ix_tensor in sub_d.items():
            tmp_dict[feature_name] = ix_tensor.cpu().numpy().flatten()
    return pd.DataFrame.from_dict(tmp_dict)


def sample_batch_target_pair(batch_data, train_df, features, within_score=0.2, min_hamming=0.8, anti_example=False):

    batch = batch_data
    targets = train_df

    ix_batch = ix_tensors_to_ix_df(df_to_ix_tensors(batch, features))
    ix_targets = ix_tensors_to_ix_df(df_to_ix_tensors(targets, features))

    A = ix_batch.values
    B = ix_targets.values
    result = 1 - (A[:, None, :] != B).mean(2)
    score_proportions = np.abs(targets.score.values[:,None].dot(1/batch.score.values[:,None].T)-1).T

    if anti_example:
        is_within_score = score_proportions >= within_score
    else:
        is_within_score = score_proportions <= within_score

    keepers = (result<1)&(result>=min_hamming)&(is_within_score)
    batch_i = np.arange(len(batch))[keepers.any(1)]
    a = np.zeros(keepers.shape)-1
    ki,kj=keepers.nonzero()
    a[ki,kj]=np.random.random(size=kj.shape)

    target_j = a.argmax(1)[batch_i]

    batch = batch.iloc[batch_i]
    target = targets.iloc[target_j]
    return batch, target


def get_validation_set(batch_data, df, features, within_score=0.2, min_hamming=0.8, anti_example=False):
    batch = batch_data
    targets = df

    ix_batch = ix_tensors_to_ix_df(df_to_ix_tensors(batch, features))
    ix_targets = ix_tensors_to_ix_df(df_to_ix_tensors(targets, features))

    A = ix_batch.values
    B = ix_targets.values
    result = 1 - (A[:, None, :] != B).mean(2)
    score_proportions = np.abs(batch.score.values[:,None].dot(1/targets.score.values[:,None].T)-1)
    if anti_example:
        is_within_score = score_proportions >= within_score
    else:
        is_within_score = score_proportions <= within_score

    keepers = (result<1)&(result>=min_hamming)&(is_within_score)
    ki, kj = keepers.nonzero()

    validation_dict = dict()
    for i in range(len(ki)):
        validation_dict[ki[i]] = validation_dict.get(ki[i], []) + [kj[i]]
    # filling missing targets
    for i in range(len(batch)):
        if i not in validation_dict:
            validation_dict[i] = []

    validation_set = dict()
    for source_i, targets_j in validation_dict.items():
        validation_set[source_i] = {
            'source': batch.iloc[[source_i]],
            'targets': targets.iloc[targets_j],
        }

    return validation_set


def reconstruction_loss(output, data_ix_tensors, generated_data_ix_tensors, criterions, features, mode='cross_entropy'):

    if mode == 'cross_entropy':
        diversity = 0
        reconstruction_loss = 0
        decoder_target_tensor = get_target_tensor(data_ix_tensors, features)
        for feature_ix, strategy_feature in enumerate(features['strategy_features']['features_order']):
            loss = criterions[feature_ix]
            predicted_distribution = output[strategy_feature]
            target_class = decoder_target_tensor[:,feature_ix]
            reconstruction_loss += loss(predicted_distribution, target_class)
            diversity += diversity_score(predicted_distribution)
        reconstruction_loss = reconstruction_loss/len(features['strategy_features']['features_order'])
    elif mode == '1_cosine':
        cliploss = criterions
        reconstruction_loss = cliploss(data_ix_tensors, generated_data_ix_tensors)


def boolean_indexing(v, fillval=np.nan):

    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out


def patch_attention(m):

    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return forward_orig(*args, **kwargs)
    
    m.forward = wrap


def save_outputs(output_dict, name):
    
	def hook(self, module_in, module_out):
		output_dict[name] = module_out[1]
	return hook


def calculate_hamming_distance(generated, df, features):
    
    cols = features['strategy_features']['features_order']
    context_cols = features['context_features']['features_order']
    top_k_tmp = generated.copy()
    our_hammings_list = list()
    # for current_context in top_k_tmp.index.unique():
    for current_context in top_k_tmp.reset_index().groupby(context_cols).groups:
        my_results = top_k_tmp.loc[[current_context]]
        current_context_dfs = [my_results]
        # cdf = df[(df.advertiser_category_context==current_context[0])&(df.adexchange_context==current_context[1])].drop_duplicates()
        cdf = df.query(' & '.join([ f'`{context_col}`=="{current_context[i]}"' for i, context_col in enumerate(context_cols)])).drop_duplicates()
        cdf = cdf.set_index(context_cols)
        cdf = cdf[cdf['is_better']]
        
        cdfmat = cdf[cols].values
        if len(cdfmat)==0:
            continue
        all_our_hammings = list()
        for i in range(len(current_context_dfs)):
            hammings = list()
            for _, garr in enumerate(current_context_dfs[i][cols].values):
                tmp = []
                for cdfarr in cdfmat:
                    hamming_distance = 1-(garr == cdfarr).mean()
                    tmp.append(hamming_distance)
                hammings.append(min(tmp))
            our_hamming = np.mean(hammings)
            all_our_hammings.append(our_hamming)
        
        tmp_dict = dict()
        for i, context_col in enumerate(context_cols):
            tmp_dict[context_col] = current_context[i]
        tmp_dict['hamming_distance'] = np.mean(all_our_hammings)
        our_hammings_list.append(tmp_dict)
        
    hammings = pd.DataFrame(our_hammings_list)
    return hammings['hamming_distance'].mean(), hammings


def calculate_cosine_similarity(generated, df, features, aligner_model):
    context_cols = features['context_features']['features_order']
    top_k_tmp = generated.copy()
    cosines_dfs = list()
    for current_context in top_k_tmp.index.unique():
        my_results = top_k_tmp.loc[[current_context]]
        cdf = df.query(' & '.join([ f'`{context_col}`=="{current_context[i]}"' for i, context_col in enumerate(context_cols)])).drop_duplicates()
        cdf = cdf.set_index(context_cols)
        cdf = cdf[cdf['is_better']]
        if len(cdf)==0:
            continue
        all_out_cosine = list()

        for i in range(len(my_results)):
            gen_ix_tensors = df_to_ix_tensors(my_results.iloc[[i]*len(cdf)].reset_index(), features)
            dat_ix_tensors = df_to_ix_tensors(cdf.reset_index(), features)

            cosines = aligner_model.cosine_similarity(gen_ix_tensors, dat_ix_tensors)
            max_cosine_sim = cosines.max().item()
        
            all_out_cosine.append(max_cosine_sim)
        my_results['cosine_sim'] = all_out_cosine
        cosines_dfs.append(my_results)

    cosine_df = pd.concat(cosines_dfs).groupby(features['context_features']['features_order'])[['cosine_sim']].mean()
    return cosine_df.mean(), cosine_df


def build_generated_df(last_generation, output, batch_data, estimated_scores, estimated_score_loss, features):
    tmp = dict()
    for s_feature in features['strategy_features']['features_order']:
        tmp[s_feature] = output[s_feature].detach().argmax(1).cpu().numpy()
    generated = pd.DataFrame.from_dict(tmp)
    generated[features['context_features']['features_order']] = batch_data[features['context_features']['features_order']].values
    generated['generated_score'] = estimated_scores.detach().cpu().numpy()
    generated['generated_positiveness'] = 1 - estimated_score_loss.cpu().numpy()
    generated = generated[features['context_features']['features_order']+features['strategy_features']['features_order']+['generated_score', 'generated_positiveness']]
    for feat, feat_dict in list(features['strategy_features']['features'].items()) + list(features['context_features']['features'].items()):
        generated.replace({feat: feat_dict['ix_to_value']}, inplace=True)

    if last_generation is not None:
        generated = pd.concat([last_generation, generated])

    generated = generated.groupby(features['context_features']['features_order'] + features['strategy_features']['features_order']).mean().reset_index()
    return generated


class BatchLogger:
    
    def __init__(self):
        self.main_lists = dict()
        self.tmp_lists = dict()
    
    def accumulate_batch_list(self, name, values):
        if isinstance(values, np.ndarray):
            values = values.tolist()
        elif isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy().tolist()
        self.tmp_lists[name] = self.tmp_lists.get(name, []) + values
    
    def log_one_value(self, name, value):
        self.main_lists[name] = self.main_lists.get(name, []) + [value]

    def summarize_epoch(self):
        for var_name, values in self.tmp_lists.items():
            self.main_lists[var_name] = self.main_lists.get(var_name, []) + [np.mean(values)]
        self.tmp_lists = dict()

    def get_main_values(self, name):
        return self.main_lists.get(name, [])
    
    def get_batch_values(self, name):
        return self.tmp_lists.get(name, [])
    
    def last(self, name):
        return self.main_lists.get(name, ['No_Monitoring'])[-1]
    
    def all_lasts(self):
        d = dict()
        for k, l in self.main_lists.items():
            d[k] = l[-1]
        return d
    

def save_fig_json_png(plot_path, fig, mlflow):
    fig.write_json(f'{plot_path}.json')
    fig.write_image(f'{plot_path}.png')
    mlflow.log_artifact(f'{plot_path}.json', 'plots')
    mlflow.log_artifact(f'{plot_path}.png', 'plots')