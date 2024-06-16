import torch
from torch import nn
import torch.nn.functional as F
from mlframework.pytorch.models import CrossNetMix
from utils import construct_embedding_module, map_tensor


class ipinyouPriceMdeler(nn.Module):

    def __init__(self, features, embedding_dim, latent_dim, normalizers, denormalizers, output_size = 2):
        super(ipinyouPriceMdeler, self).__init__()

        self.features = features
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.normalizers = normalizers
        self.denormalizers = denormalizers

        embeddings, type_to_ix = construct_embedding_module(features, emb_dim=embedding_dim)
        self.embeddings = embeddings
        #nn.init.xavier_uniform_(self.embeddings.weight)
        
        nb_numerical = len(normalizers) - 1   # -1 to not count goal dimension
        nb_categorical = len(features['context_features']['features_order'] + features['strategy_features']['features_order']) - nb_numerical 
        input_dim = (nb_categorical * embedding_dim) + nb_numerical

        self.dnn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, int(input_dim/2)),
            nn.ReLU(),
            nn.Linear(int(input_dim/2), latent_dim)
        )

        self.latent = nn.Linear(latent_dim + input_dim, latent_dim)

        self.final_layer = nn.Linear(latent_dim, output_size)
        
        self.dcn = CrossNetMix(input_dim, layer_num=3, low_rank=32, num_experts=4)
    

    def forward(self, data_ix_tensors, latent_only=False, from_logits=False, logits_dict=None, softmax_weight=10):
        
        if from_logits:
            input_tensor = self.logits_to_input(logits_dict, data_ix_tensors, softmax_weight=softmax_weight)
        else:
            input_tensor = self.data_ix_tensors_to_input(data_ix_tensors)
        
        lt = self.to_latent(input_tensor)
        if latent_only:
            return lt
        
        out = self.final_layer(lt)
        
        return out

    def data_ix_tensors_to_input(self, data_ix_tensors):
        batch_size = next(iter(next(iter(data_ix_tensors.items()))[1].values())).size(0)
        input_tensor = list()
        for col in self.features['context_features']['features_order']:
            t = data_ix_tensors['context_features'][col]
            value_embedding = self.embeddings['item_embeddings'][col](t).view(batch_size, -1)
            input_tensor.append(value_embedding)

        for col in self.features['strategy_features']['features_order']:
            if col in self.normalizers:
                mapping = self.features['strategy_features']['features'][col]['ix_to_value']
                tn = data_ix_tensors['strategy_features'][col]
                tn = self.normalizers[col](map_tensor(tn, mapping)).view(batch_size, -1)
                input_tensor.append(tn)
            else:
                t = data_ix_tensors['strategy_features'][col]
                value_embedding = self.embeddings['item_embeddings'][col](t).view(batch_size, -1)
                input_tensor.append(value_embedding)

        input_tensor = torch.cat(input_tensor,1)
        return input_tensor


    def logits_to_input(self, logits_dict, data_ix_tensors, softmax_weight=10):
        
        batch_size = next(iter(next(iter(data_ix_tensors.items()))[1].values())).size(0)
        input_tensor = list()
        for col in self.features['context_features']['features_order']:
            t = data_ix_tensors['context_features'][col]
            value_embedding = self.embeddings['item_embeddings'][col](t).view(batch_size, -1)
            input_tensor.append(value_embedding)

        for col in self.features['strategy_features']['features_order']:
            if col in self.normalizers:
                mapping = self.features['strategy_features']['features'][col]['ix_to_value']
                tn = data_ix_tensors['strategy_features'][col]
                tn = self.normalizers[col](map_tensor(tn, mapping)).view(batch_size, -1)
                input_tensor.append(tn)
            else:
                # value_embedding = torch.matmul( 
                #     torch.softmax(logits_dict[col]*softmax_weight, dim=-1), 
                #     self.embeddings['item_embeddings'][col].weight
                # ).view(batch_size, -1)
                value_embedding = torch.matmul( 
                    F.gumbel_softmax(logits_dict[col], tau=softmax_weight, hard=True, dim=-1), 
                    self.embeddings['item_embeddings'][col].weight
                ).view(batch_size, -1)
                input_tensor.append(value_embedding)

        input_tensor = torch.cat(input_tensor,1)
        return input_tensor


    def to_latent(self, input_tensor):

        out2 = self.dnn(input_tensor)

        dcn_out = self.dcn(input_tensor)
        if len(dcn_out.size()) == 1:
            dcn_out = dcn_out.view(1,-1)
            
        combined = F.relu(torch.cat([dcn_out, out2], dim=-1))
        
        #out = self.linear3(out2)
        lt = torch.sigmoid(self.latent(combined))
        return lt