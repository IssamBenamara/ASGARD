import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from vector_quantize_pytorch import FSQ
from aligner import CLIPModel
from utils import construct_embedding_module, get_torch_device, patch_attention, save_outputs
from utils import TokenPlacement, InstancingType, boolean_indexing
from custom_layers import CustomTransformerEncoder, CustomTransformerEncoderLayer, CustomTransformerDecoder, CustomTransformerDecoderLayer

class ASGAR(nn.Module):

    def __init__(self, 
                 features: dict, 
                 emb_size: int, 
                 n_head:int, 
                 n_layers:int,
                 ffn_expansion: float = 4,
                 dropout: float = 0.2,
                 use_interaction=True,
                 positional_encodings=False,
                 n_sampling_choices=0,
                 pretrained_embeddings=None,
                 aligner_model:CLIPModel=None,
                 type_embedding_weight:float=1.0,
                 use_bias=True,
                 wide_ffns=True,
                 variational=None,
                 cross_attn_mask_encoder_context=True,
                 cross_attn_mask_encoder_instance_q=True,
                 use_feature_type_memory_mask=False,
                 context_placement:TokenPlacement=TokenPlacement.ENCODER_DECODER,
                 quantized_placement:TokenPlacement=TokenPlacement.DECODER_ONLY,
                 instancing_type:InstancingType=InstancingType.QUANTIZATION,
                 levels:list = [7,5,5,5,5],
                 n_quantized_tokens:int = 16,
                 candidating_members = None,
                 encoder_only = False,
                 use_special_embedding = False,
                 masking_rate = 0.3,
                 **kwargs,
                 ):
        super().__init__()
        self.model_type = 'Transformer'
        self.encoder_only = encoder_only
        self.use_special_embedding = use_special_embedding
        self.device = get_torch_device()
        self.features = features
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.n_head = n_head
        self.ffn_expansion = ffn_expansion
        self.n_sampling_choices = n_sampling_choices
        self.type_embedding_weight = type_embedding_weight
        self.wide_ffns = wide_ffns
        self.variational = variational
        self.context_placement = context_placement
        if self.context_placement in [TokenPlacement.DECODER_ONLY, 
                                      TokenPlacement.NO_PLACE, 
                                      TokenPlacement.SUMMED_TO_ENCODER, 
                                      TokenPlacement.SUMMED_TO_ENCODER_DECODER]: 
            self.cross_attn_mask_encoder_context = False
        else:
            self.cross_attn_mask_encoder_context = cross_attn_mask_encoder_context
        
        self.quantized_placement = quantized_placement
        if self.quantized_placement in [TokenPlacement.DECODER_ONLY, 
                                        TokenPlacement.NO_PLACE, 
                                        TokenPlacement.SUMMED_TO_ENCODER, 
                                        TokenPlacement.SUMMED_TO_ENCODER_DECODER]: 
            self.cross_attn_mask_encoder_instance_q = False
        else:
            self.cross_attn_mask_encoder_instance_q = cross_attn_mask_encoder_instance_q

        if aligner_model:
            self.aligner_model = aligner_model
            self.aligner_model.requires_grad_(False)
            self.aligner_model.eval()
            self.aligner_projection_dim = self.aligner_model.source_projection.fc.in_features
            self.aligner_to_emb = nn.Linear(self.aligner_projection_dim, self.emb_size)
        else:
            self.aligner_model = None
        self.instancing_type = instancing_type

        self.candidating_members = candidating_members

        # QUANTIZATION
        self.fsq_levels = levels
        self.n_quantized_tokens = n_quantized_tokens
        self.quantizer = FSQ(self.fsq_levels)
        self.encoder_fsq = nn.Sequential(
            nn.Linear(self.emb_size, self.aligner_projection_dim, bias=use_bias),
            nn.GELU(),
            nn.Linear(self.aligner_projection_dim, len(self.fsq_levels)*n_quantized_tokens, bias=use_bias),
        )
        self.decoder_fsq = nn.Sequential(
            nn.Linear(len(self.fsq_levels)*n_quantized_tokens, self.aligner_projection_dim, bias=use_bias),
            nn.GELU(),
            nn.Linear(self.aligner_projection_dim, self.emb_size, bias=use_bias),
        )

        #EMBEDDINGS
        embeddings, type_to_ix = construct_embedding_module(features, emb_dim=emb_size, positional=positional_encodings)
        self.embeddings = embeddings
        self.type_to_ix = type_to_ix
        if pretrained_embeddings:
            self.fill_pretrained_item_embeddings(pretrained_embeddings)
        self.special_embedding = nn.Embedding(1, emb_size)
        self.special_tokens = nn.ModuleDict({feat: nn.Embedding(4, emb_size) for feat in features['strategy_features']['features_order']}) # 0 neutral, 1 favor, 2 meh, 3 avoid
        # self.masking_rate = masking_rate

        #SAMPLING
        if self.n_sampling_choices > 0:
            self.sampling_embeddings = nn.Embedding(n_sampling_choices, emb_size)
            self.sampling_embeddings.weight.requires_grad = False
            #self.register_buffer('sampling_embeddings', sampling_embeddings)
        
        #VARIATIONAL CONTEXT
        if self.variational:
            self.variational_base = nn.Sequential(
                    nn.Linear(self.emb_size, ffn_expansion*self.emb_size, bias=True),
                    nn.LeakyReLU(),
                    nn.Linear(ffn_expansion*self.emb_size, self.emb_size, bias=True),
                    nn.LeakyReLU(),
                    )
            self.variational_mean = nn.Linear(self.emb_size, emb_size, bias=True)
            self.variational_log_var = nn.Linear(self.emb_size, emb_size, bias=True)
        # if self.context_placement in [TokenPlacement.SUMMED_TO_ENCODER, TokenPlacement.SUMMED_TO_ENCODER_DECODER]:
        #     n_context_tokens = len(features['context_features']['features_order'])
        #     self.context_projection = nn.Linear(n_context_tokens*self.emb_size, self.emb_size)

        n_context_tokens = len(features['context_features']['features_order'])
        self.context_projection = nn.Sequential(
                    nn.Linear(n_context_tokens*self.emb_size, ffn_expansion*self.emb_size, bias=False),
                    nn.LeakyReLU(),
                    nn.Linear(ffn_expansion*self.emb_size, self.emb_size, bias=False),
                    )
            
        #ENCODER
        encoder_layer = CustomTransformerEncoderLayer(d_model=emb_size, nhead=n_head, batch_first=True, norm_first=True, dropout=dropout)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers=n_layers)

        #DECODER
        decoder_layer = CustomTransformerDecoderLayer(d_model=emb_size, nhead=n_head, batch_first=True, norm_first=True, dropout=dropout)
        self.transformer_decoder = CustomTransformerDecoder(decoder_layer, num_layers=n_layers)

        #patching attention
        self.decoder_cross_attn = dict()
        for i, l in enumerate(self.transformer_decoder.layers):
            attention_layer = l.multihead_attn
            patch_attention(attention_layer)
            attention_layer.register_forward_hook(save_outputs(self.decoder_cross_attn, f'layer_{i}_attn_weights'))
            
        #memory_mask
        self.use_feature_type_memory_mask = use_feature_type_memory_mask
        if self.use_feature_type_memory_mask:
            assert self.candidating_members is None, "Can't use feature type memory mask if candidating. All tokens should be included to encoder."
            mask = [[]]
            for feat in features['strategy_features']['features_order']:
                e = [0]*len(mask[-1]) + [1]*len(features['strategy_features']['features'][feat]['values'])
                mask.append(e)
            mask = boolean_indexing(mask[1:], fillval=0)
            mask = torch.from_numpy(mask).float().to(self.device)
            self.memory_feature_mask = mask.masked_fill(mask != 1, float('-inf'))

        #FFNs
        self.ffns = nn.ModuleDict()
        if self.wide_ffns:
            ffn_input_size = len(features['strategy_features']['features_order'])*emb_size
        else:
            ffn_input_size = emb_size
        for strategy_feature in features['strategy_features']['features_order']:
            vocab_size = len(features['strategy_features']['features'][strategy_feature]['values'])
            self.ffns[strategy_feature] =  nn.Sequential(
                nn.Linear(ffn_input_size, int(emb_size*ffn_expansion), bias=use_bias),
                nn.LeakyReLU(),
                nn.Linear(int(emb_size*ffn_expansion), vocab_size, bias=use_bias),
                )
            
        #self.interaction_layer = CrossNetMix(emb_size*len(features['strategy_features']['features_order']), layer_num=3, low_rank=32, num_experts=4)

        self.init_weights()

        self.first_printed = 0

    def init_weights(self) -> None:
        pass

    def reparameterization(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std).to(self.device)        # sampling epsilon        
        z = mean + std*epsilon                          # reparameterization trick
        return z

    def first_print(self, s):
        if self.first_printed < 2:
            print(s)

    def forward(self, data_ix_tensors, p=None, candidating=False, from_indices=None, guiding=None, masking_rate=0.5):
        chosen_samples=None
        indices = None
        variational_mean = None
        variational_log_var = None
        #CONTEXT
        if self.context_placement in [TokenPlacement.SUMMED_TO_ENCODER, TokenPlacement.SUMMED_TO_ENCODER_DECODER]:
            context_tensor = self.get_context_embeddings(data_ix_tensors,concatenated=True) #shape: (batch_size, 1, n_context_features*embedding_dim)
            context_tensor = self.context_projection(context_tensor) #shape: (batch_size, 1, embedding_dim)
        else:
            context_tensor = self.get_context_embeddings(data_ix_tensors,concatenated=False) #shape: (batch_size, n_context_features, embedding_dim)
        #VARIATIONAL LATENT
        if self.variational:
            latent_variational = self.variational_base(context_tensor)
            variational_mean = self.variational_mean(latent_variational)
            variational_log_var = self.variational_log_var(latent_variational)
            variational_z = self.reparameterization(variational_mean, variational_log_var) #shape: (batch_size, 1, embedding_dim)
        

        # Input Tensor
        input_tensor = self.get_encoder_input_tensors(data_ix_tensors, p=p, candidating=candidating, guiding=guiding, masking_rate=masking_rate) #shape: (batch_size, sequence_len, embedding_dim)
        batch_size = input_tensor.shape[0]
        if self.context_placement in [TokenPlacement.SUMMED_TO_ENCODER, TokenPlacement.SUMMED_TO_ENCODER_DECODER]:
            self.first_print(f'Context Placement Summed to Encoder Input')
            input_tensor = input_tensor + context_tensor.expand_as(input_tensor)
        self.first_print(f'Input Tensor: {input_tensor.shape}')

        # Instance Projection
        if self.aligner_model:
            if from_indices is None:
                instance_projection, _, _ = self.aligner_model(
                        {
                            "source":data_ix_tensors,
                            "target":data_ix_tensors,
                        },
                        inference = True,
                    )
                instance_projection = self.aligner_to_emb(instance_projection)
                instance_projection = instance_projection.view(batch_size, 1, self.emb_size)
            if self.instancing_type == InstancingType.QUANTIZATION:
                self.first_print(f'Quantization Instancing')
                if from_indices is None:
                    instance_projection = self.encoder_fsq(instance_projection).view(batch_size,len(self.fsq_levels),1,self.n_quantized_tokens)
                    quantized, indices = self.quantizer(instance_projection)
                else:
                    quantized = self.quantizer.indices_to_codes(from_indices)
                quantized_projected = quantized.reshape(batch_size, 1, len(self.fsq_levels)*self.n_quantized_tokens)
                quantized_projected = self.decoder_fsq(quantized_projected)
            elif self.instancing_type == InstancingType.ALIGNER_PROJECTION:
                self.first_print(f'Aligner Instancing')
                quantized_projected = instance_projection
        if self.quantized_placement in [TokenPlacement.ENCODER_ONLY, TokenPlacement.ENCODER_DECODER]:
            self.first_print(f'Quantized Placement in Encoder')
            input_tensor = torch.cat([quantized_projected, input_tensor], dim=1) # inject context variational latent to decoder instead
        elif self.quantized_placement in [TokenPlacement.SUMMED_TO_ENCODER, TokenPlacement.SUMMED_TO_ENCODER_DECODER]:
            self.first_print(f'Quantized Placement Summed to Encoder Input')
            input_tensor = input_tensor + quantized_projected.expand_as(input_tensor)
        if self.variational in [TokenPlacement.ENCODER_ONLY, TokenPlacement.ENCODER_DECODER]:
            self.first_print(f'Variational Placement in Encoder')
            input_tensor = torch.cat([variational_z, input_tensor], dim=1)
        if self.encoder_only or self.use_special_embedding:
            self.first_print('Special Embedding in Encoder')
            spec = self.special_embedding(torch.zeros(1, dtype=torch.int, device=self.device)).expand((batch_size, 1, self.emb_size))
            input_tensor = torch.cat([input_tensor, spec], dim=1)


        self.first_print(f'Final Input Tensor: {input_tensor.shape}')

        # ENCODER
        encoder_output, encoder_attn_weights = self.transformer_encoder(input_tensor) #shape: (batch_size, sequence_len, embedding_dim)
        self.first_print(f'Encoder Output Tensor: {input_tensor.shape}')

        if self.encoder_only:
            result = dict()
            for feature_ix, strategy_feature in enumerate(self.features['strategy_features']['features_order']):
                if self.wide_ffns:
                    result[strategy_feature] = self.ffns[strategy_feature](encoder_output[:,:len(self.features['strategy_features']['features_order']),:].view(batch_size,-1)) 
                else:
                    # result[strategy_feature] = self.ffns[strategy_feature](encoder_output[:,2+feature_ix,:]) 
                    result[strategy_feature] = self.ffns[strategy_feature](encoder_output[:,-1,:]) 
                    # self.first_print(f'Encoder Output {strategy_feature} in {2+feature_ix} position')

            self.first_print('\n\n')
            self.first_printed += 1
            attentions = {
                'encoder_attention': encoder_attn_weights,
                'decoder_self_attention': None,
                'decoder_cross_attention': None,
            }
            return result, variational_mean, variational_log_var, attentions
    


        #DECODER
        decoder_list = list()
        batch_size = input_tensor.shape[0]

        decoder_input_tensor = self.get_decoder_input_tensor(batch_size)
        if self.context_placement in [TokenPlacement.SUMMED_TO_ENCODER_DECODER]:
            self.first_print(f'Context Placement Summed to Decoder Input')
            decoder_input_tensor = decoder_input_tensor + context_tensor.expand_as(decoder_input_tensor)
        self.first_print(f'Decoder Input Tensor: {decoder_input_tensor.shape}')
        #decoder_input_tensor = decoder_input_tensor + compressed_context.expand_as(decoder_input_tensor)
        # variational_z = torch.randn_like(decoder_input_tensor, device=self.device)
        decoder_list.append(decoder_input_tensor)
        if self.context_placement in [TokenPlacement.DECODER_ONLY, TokenPlacement.ENCODER_DECODER]:
            self.first_print(f'Context Placement in Decoder')
            decoder_list.append(context_tensor)
        if self.variational in [TokenPlacement.DECODER_ONLY, TokenPlacement.ENCODER_DECODER]:
            self.first_print(f'Variational Placement in Decoder')
            decoder_list.append(variational_z)
            
        has_q = 0
        if self.aligner_model:
            if self.quantized_placement in [TokenPlacement.DECODER_ONLY, TokenPlacement.ENCODER_DECODER]:
                self.first_print(f'Quantized Placement in Decoder')
                decoder_list.append(quantized_projected)
            if self.quantized_placement in [TokenPlacement.ENCODER_ONLY, TokenPlacement.ENCODER_DECODER]:
                if self.cross_attn_mask_encoder_instance_q:
                    has_q = 1
            #decoder_list.append(instance_projection)

        decoder_input_tensor = torch.cat(decoder_list, dim=1)
        self.first_print(f'Final Decoder Input Tensor: {decoder_input_tensor.shape}')

        if self.use_special_embedding:
            decoder_memory_mask = None
        else:
            decoder_memory_mask = torch.zeros(decoder_input_tensor.size(1), encoder_output.size(1), device=self.device) # shape: (nb_decoder_input_tokens, nb_encoder_output_tokens)
            if self.quantized_placement in [TokenPlacement.ENCODER_ONLY, TokenPlacement.ENCODER_DECODER]: 
                if self.cross_attn_mask_encoder_context: 
                    if self.cross_attn_mask_encoder_instance_q:
                        decoder_memory_mask[:, 0:1+int(context_tensor.size(1))] = float('-inf') # There is Q AND Ctx_mask AND Q_mask
                        self.first_print(f'MASKING in Encoder: Quantized and Context')
                    else:
                        decoder_memory_mask[:, 1:1+int(context_tensor.size(1))] = float('-inf') # There is Q AND Ctx_mask
                        self.first_print(f'MASKING in Encoder: Context')
                else:
                    if self.cross_attn_mask_encoder_instance_q:
                        self.first_print(f'MASKING in Encoder: Quantized')
                        decoder_memory_mask[:, 0:1] = float('-inf') # There is Q AND Q_mask
            else:
                if self.cross_attn_mask_encoder_context:
                    self.first_print(f'MASKING in Encoder: Context')
                    decoder_memory_mask[:, 0:int(context_tensor.size(1))] = float('-inf') # Ctx_mask
            if self.use_feature_type_memory_mask:
                decoder_memory_mask[:self.memory_feature_mask.shape[0], -self.memory_feature_mask.shape[1]:] = self.memory_feature_mask
            

        if self.use_special_embedding:
            encoder_output = encoder_output[:,[-1],:]
            self.first_print(f'USING only SPECIAL EMBEDDING in Cross Attention: shape {encoder_output.shape}')

        decoder_output, decoder_self_attn_weights, decoder_cross_attn_weights = self.transformer_decoder(decoder_input_tensor, encoder_output, memory_mask=decoder_memory_mask)
        self.first_print(f'Decoder Output Tensor: {decoder_output.shape}')
        #decoder_output = decoder_output + variational_z.expand_as(decoder_output)

        #FFNs
        # Careful because the last part of the decoder_output is the variational_z output
        result = dict()
        for feature_ix, strategy_feature in enumerate(self.features['strategy_features']['features_order']):
            if self.wide_ffns:
                result[strategy_feature] = self.ffns[strategy_feature](decoder_output[:,:len(self.features['strategy_features']['features_order']),:].view(batch_size,-1)) 
            else:
                result[strategy_feature] = self.ffns[strategy_feature](decoder_output[:,feature_ix,:]) 

        self.first_print('\n\n')
        self.first_printed += 1

        attentions = {
            'encoder_attention': encoder_attn_weights,
            'decoder_self_attention': decoder_self_attn_weights,
            'decoder_cross_attention': decoder_cross_attn_weights,
        }
        return result, variational_mean, variational_log_var, attentions

    def get_batch_size(self, data_ix_tensors):
        first_context_column = self.features['context_features']['features_order'][0]
        batch_size = len(data_ix_tensors['context_features'][first_context_column])
        return batch_size

    def quantize(self, data_ix_tensors):
        indices = None

        # Input Tensor
        batch_size = self.get_batch_size(data_ix_tensors)

        # Instance Projection
        if self.aligner_model:
            instance_projection, _, _ = self.aligner_model(
                    {
                        "source":data_ix_tensors,
                        "target":data_ix_tensors,
                    },
                    inference = True,
                )
            instance_projection = self.aligner_to_emb(instance_projection)
            instance_projection = instance_projection.view(batch_size, 1, self.emb_size)
            instance_projection = self.encoder_fsq(instance_projection).view(batch_size,len(self.fsq_levels),1,self.n_quantized_tokens)
            quantized, indices = self.quantizer(instance_projection)
            quantized_projected = quantized.reshape(batch_size, 1, len(self.fsq_levels)*self.n_quantized_tokens)
            quantized_projected = self.decoder_fsq(quantized_projected)
        
        return quantized, indices, instance_projection, quantized_projected
        

    def get_context_embeddings(self, data_ix_tensors, concatenated=True):

        first_context_column = self.features['context_features']['features_order'][0]
        batch_size = len(data_ix_tensors['context_features'][first_context_column])
        input_tensors = list()

        for col in self.features['context_features']['features_order']:
            t = data_ix_tensors['context_features'][col]
            value_embedding = self.embeddings['item_embeddings'][col](t).view(batch_size, self.emb_size) #shape: (batch_size, emb_dim)

            type_ix_t = torch.tensor(self.type_to_ix[col], dtype=torch.int, device=self.device).expand(batch_size)
            type_embedding = self.embeddings['type_embeddings'](type_ix_t).view(batch_size, self.emb_size) #shape: (batch_size, emb_dim)

            final_embedding = value_embedding + self.type_embedding_weight*type_embedding #shape: (batch_size, emb_dim)
            input_tensors.append(final_embedding.view(batch_size, 1, self.emb_size)) #shape: (batch_size, 1, emb_dim)

        input_tensors = torch.cat(input_tensors, 1) #shape: (batch_size, n_context_features, embedding_dim)
        if concatenated:
            input_tensors = input_tensors.view( (-1, 1, len(self.features['context_features']['features_order'])*self.emb_size) ) #shape: (batch_size, n_context_features * embedding_dim)
        return input_tensors 


    def get_encoder_input_tensors(self, data_ix_tensors, p=None, candidating=None, guiding=None, masking_rate=0.0):
        
        first_context_column = self.features['context_features']['features_order'][0]
        batch_size = len(data_ix_tensors['context_features'][first_context_column])
        # if p >= 0.75 -----> code 1 means RECONSTRUCT
        # if 0 <= p < 0.75 -----> code 2 means AVOID
        # if p < 0  -----> code 0 means NEUTRAL
        if p is None:
            self.first_print('No Positiveness provided, everything is NEUTRAL')
            p = torch.ones(batch_size, device=self.device)*-1
        if guiding == 'favor':
            self.first_print('Guidance: Favor')
            p = torch.ones(batch_size, device=self.device)
            masking_rate = -1
        elif guiding == 'avoid':
            self.first_print('Guidance: Avoid')
            p = torch.ones(batch_size, device=self.device)*0.1
            masking_rate = -1
        elif guiding == 'soft_avoid':
            self.first_print('Guidance: Soft Avoid')
            p = torch.ones(batch_size, device=self.device)*0.5
            masking_rate = -1
        # else:
        #     masking_rate = self.masking_rate

        if -2.5 < masking_rate and masking_rate < -1.5:
            # masking_rate == -2.0 : batch level random, we generate compare a random to a random for each batch.
            masking_rate = torch.rand((batch_size,1),device=self.device)

        input_tensors = list()

        if self.context_placement in [TokenPlacement.ENCODER_ONLY, TokenPlacement.ENCODER_DECODER]: 
            self.first_print(f'Context Placement in Encoder')
            for col in self.features['context_features']['features_order']:
                t = data_ix_tensors['context_features'][col]
                value_embedding = self.embeddings['item_embeddings'][col](t).view(batch_size, self.emb_size) #shape: (batch_size, emb_dim)

                type_ix_t = torch.tensor(self.type_to_ix[col], dtype=torch.int, device=self.device).expand(batch_size)
                type_embedding = self.embeddings['type_embeddings'](type_ix_t).view(batch_size, self.emb_size) #shape: (batch_size, emb_dim)

                final_embedding = value_embedding + self.type_embedding_weight*type_embedding #shape: (batch_size, emb_dim)
                input_tensors.append(final_embedding.view(batch_size, 1, self.emb_size)) #shape: (batch_size, 1, emb_dim)

        if candidating:
            for col in self.features['strategy_features']['features_order']:
                vals = self.features['strategy_features']['features'][col]['values']
                dis = torch.ones(batch_size, len(vals), device=self.device)
                if candidating == 'members':
                    # dis[torch.arange(0, dis.size(0), dtype=torch.long, device=self.device), data_ix_tensors['strategy_features'][col]] = 0
                    dis.scatter_(1, data_ix_tensors['strategy_features'][col], 0)
                    members = self.candidating_members if self.candidating_members is not None else int(len(vals)/2+1)
                    self.first_print(f'Candidating {col} with {members} Members')
                    if members == 0:
                        # if no other candidate, just take the instance original token
                        candidates = data_ix_tensors['strategy_features'][col].view(batch_size, 1)
                    else:
                        candidates = torch.multinomial(dis, members)
                        candidates = torch.cat([data_ix_tensors['strategy_features'][col].view(batch_size, 1), candidates], dim=1)
                elif candidating == 'shuffle':
                    self.first_print(f'Candidating {col} with Shuffle')
                    candidates = torch.multinomial(dis, len(vals))
                value_embedding = self.embeddings['item_embeddings'][col](candidates).view(batch_size, -1, self.emb_size) #shape: (batch_size, current_feature_vocab_len, emb_dim)

                type_ix_t = torch.tensor(self.type_to_ix[col], dtype=torch.int, device=self.device).expand(batch_size, candidates.size(1))
                type_embedding = self.embeddings['type_embeddings'](type_ix_t).view(batch_size, -1, self.emb_size) #shape: (batch_size, current_feature_vocab_len, emb_dim)

                final_embedding = value_embedding + self.type_embedding_weight*type_embedding #shape: (batch_size, current_feature_vocab_len, emb_dim)
                input_tensors.append(final_embedding)
                
                # freeing gpu memory
                del candidates, dis
        else:
            # contxt = torch.cat(input_tensors, dim=1).flatten(start_dim=1, end_dim=2).unsqueeze(1)
            # context_projection = self.context_projection(contxt)
            
            # if p >= 0.75 -----> code 1 means RECONSTRUCT
            # if 0.25 <= p < 0.75 -----> code 2 means AVOID
            # if p < 0.25  -----> code 0 means NEUTRAL

            # pos_code = ((p>=0.75)*1 + ((p<0.75)&(p>=0.25))*2).unsqueeze(1).to(self.device) # (batch_size, 1) 1 if positive, 2 otherwise
            
            # if p >= 0.75 -----> code 1 means RECONSTRUCT
            # if p <= 0.25 -----> code 2 means AVOID
            # if 0.25 < p < 0.75  -----> code 0 means NEUTRAL

            # 0 neutral, 1 favor, 2 meh, 3 avoid
            # pos_code = ((p>=0.75)*1 + ((p>=0.25)&(p<0.75))*2 + ((p>=0.0)&(p<0.25))*3).unsqueeze(1).to(self.device)
            pos_code = ((p>=0.75)*1 + ((p>=0.25)&(p<0.75))*2 + ((p>=0.0)&(p<0.25))*2).unsqueeze(1).to(self.device) #just using avoid for neutral AND negative for now

            for col in self.features['strategy_features']['features_order']:
                d = self.features['strategy_features']['features'][col]
                vocab_ixs = torch.arange(0, len(d['values']), dtype=torch.int, device=self.device).expand(batch_size,-1)
                instance_ixs = data_ix_tensors['strategy_features'][col]
                data_mask = torch.where((vocab_ixs - instance_ixs) == 0, 1, 0).to(self.device) # (batch_size, vocab_size) vocab where 1 is instance value and 0 is other values
                data_mask = data_mask*((torch.rand((batch_size,1),device=self.device)>=masking_rate)*1) # randomly consider an instance token as neutral
                reconst_avoid_embedding = self.special_tokens[col](data_mask*pos_code) #(batch_size, vocab_size, emb_size) all vocab but with [reconstruct], [avoid], [neutral] embeddings
                value_embedding = self.embeddings['item_embeddings'][col](vocab_ixs).view(batch_size, -1, self.emb_size) #shape: (batch_size, current_feature_vocab_len, emb_dim)

                type_ix_t = torch.tensor(self.type_to_ix[col], dtype=torch.int, device=self.device).expand(batch_size,len(d['values']))
                type_embedding = self.embeddings['type_embeddings'](type_ix_t).view(batch_size, -1, self.emb_size) #shape: (batch_size, current_feature_vocab_len, emb_dim)

                final_embedding = value_embedding + self.type_embedding_weight*type_embedding + reconst_avoid_embedding #shape: (batch_size, current_feature_vocab_len, emb_dim)
                # final_embedding += context_projection 
                input_tensors.append(final_embedding)

        input_tensors = torch.cat(input_tensors, 1) #shape: (batch_size, all_vocab_len, embedding_dim)

        return input_tensors #shape: (batch_size, all_vocab_len, embedding_dim)


    def get_decoder_input_tensor(self, batch_size):

        decoder_input_tensor = list()
        
        for col in self.features['strategy_features']['features_order']:
            type_ix_t = torch.tensor([[self.type_to_ix[col]]], dtype=torch.int, device=self.device).expand(batch_size,1)
            type_embedding = self.embeddings['type_embeddings'](type_ix_t)
            decoder_input_tensor.append(type_embedding)
        
        decoder_input_tensor = torch.cat(decoder_input_tensor,1) #shape: (batch_size, sequence_len, embedding_dim)

        return decoder_input_tensor

    def fill_pretrained_item_embeddings(self, pretrained_embeddings):
        for feature_type in self.features:
            for col in self.features[feature_type]['features_order']:
                tmp_emb = pretrained_embeddings.item_embeddings[col]
                self.embeddings.item_embeddings[col] = nn.Embedding.from_pretrained(tmp_emb.weight)


def score_positiveness(batch_data, score_tensor, stats_df, features, min_anchor_positiveness=0.05, max_anchor_positiveness=None, c_scaling=1, contexts=None):
    
    device = get_torch_device()
    if contexts != None:
        tmp = stats_df.loc[contexts]
    else:
        tmp = stats_df.loc[batch_data[features['context_features']['features_order']].to_records(index=False).tolist()]
    c = tmp['mean'].values*c_scaling
    min_anchors = tmp['min'].values
    min_g = np.log((1/min_anchor_positiveness)-1)/(c-min_anchors)
    min_l = c*min_g

    min_g = torch.tensor(min_g, device=device)
    min_l = torch.tensor(min_l, device=device)
    p = 1/(1 + torch.exp( min_l - min_g*score_tensor ))

    if max_anchor_positiveness != None:
        max_anchors = tmp['max'].values
        max_g = np.log((1/max_anchor_positiveness)-1)/(c-max_anchors)
        max_l = c*max_g

        max_g = torch.tensor(max_g, device=device)
        max_l = torch.tensor(max_l, device=device)
        new_p = 1/(1 + torch.exp( max_l - max_g*score_tensor ))
        p = torch.minimum(p, new_p)

    return p


def positive_curve(x, y, eiw=1, dfw=0.005):
    # eiw (estimator ignorance weight) : higher value means we don't care about estimator
    # dfw (data fidelity weight) : higher value means we want reconstruction of good strategy
    # (high eiw, high dfw): target a good strategy even if it has bad score estimation
    # (high eiw, low dfw): target a good strategy WITH good estimation
    # (low eiw, high dfw): target either good strategy OR well estimated other
    # (low eiw, low dfw): target any well estimated other (almost ignore data)
    return torch.exp((y-eiw)/eiw)+dfw*torch.pow(x,2)*(torch.pow(y,2)+1)


def average_curve(x, y, eiw=0.5, aaw=2):
    # eiw (estimator ignorance weight) : higher value means we don't care about estimator
    # aaw (average avoidance weight) : higher value means we want to avoid reconstructing neutral strategy
    return torch.exp((y-eiw)/eiw) + aaw/(x+1) 


def negative_curve(x, y, nfw=40, new=1):
    # nfw (negative force weight) : higher value means want to avoid negative strategy more
    return (nfw*torch.exp(new*y))/(x+0.1) + y*torch.exp(new*y)


def L(reconstruction_l, score_l, p, peiw=1, pdfw=0.005, aeiw=0.5, aaw=2, nfw=40, new=1, interpolating=True):
    pos = torch.clamp(2*p-1, 0)
    neg = torch.clamp(1-2*p, 0)
    avg = 2*(1-torch.max(p, 1-p))

    x = reconstruction_l
    y = score_l

    if interpolating:
        return pos*positive_curve(x, y, eiw=peiw, dfw=pdfw) + avg*average_curve(x, y, eiw=aeiw, aaw=aaw) + neg*negative_curve(x, y, nfw=nfw, new=new)
    else:
        pos_ix = pos>=0.5
        avg_ix = avg>=0.5

        total_loss = negative_curve(x, y, nfw=nfw, new=new)
        total_loss[pos_ix] = positive_curve(x[pos_ix], y[pos_ix], eiw=peiw, dfw=pdfw)
        total_loss[avg_ix] = average_curve(x[avg_ix], y[avg_ix], eiw=aeiw, aaw=aaw)

        return total_loss
    

def save_loss_surfaces_plot(args, path, save_file=True):

    def POS_CURVE(x, y):
        x = torch.tensor(x)
        y = torch.tensor(y)
        z = positive_curve(x, y, eiw=args.peiw, dfw=args.pdfw)
        return z.numpy()

    def AVG_CURVE(x, y):
        x = torch.tensor(x)
        y = torch.tensor(y)
        z = average_curve(x, y, eiw=args.aeiw, aaw=args.aaw) 
        return z.numpy()

    def NEG_CURVE(x, y):
        x = torch.tensor(x)
        y = torch.tensor(y)
        z = negative_curve(x, y, new=args.new, nfw=args.nfw)
        return z.numpy()
    
    N = 100
    X_VAL = np.linspace(0, 8, N)
    Y_VAL = np.linspace(0, 1, N)
    X1, Y1 = np.meshgrid(X_VAL, Y_VAL)

    POS_Z = POS_CURVE(X1, Y1)
    AVG_Z = AVG_CURVE(X1, Y1)
    NEG_Z = NEG_CURVE(X1, Y1)

    fig = go.Figure(
        data=[
            go.Surface(x=X1, y=Y1, z=POS_Z, opacity=0.5, colorscale='greens_r'),
            go.Surface(x=X1, y=Y1, z=AVG_Z, opacity=0.5, colorscale='blues_r'),
            go.Surface(x=X1, y=Y1, z=NEG_Z, opacity=0.5, colorscale='reds_r'),
            ],
    )
    fig.update_layout(
        title='Loss Curves',
        scene_camera_eye=dict(x=-1, y=-2, z=1),
        scene = dict(
            xaxis_title = 'Reconstruction Loss',
            xaxis = dict(range=[0,8]),
            yaxis_title = 'Positiveness Loss',
            yaxis = dict(range=[0,1]),
            zaxis_title = 'Global Loss',
            zaxis = dict(range=[0,6]),
        ),
        height = 1000
    )
    if save_file:
        fig.write_html(path)
    return fig, path


def make_ASGAR(args, strategy_estimator, aligner_model, features):
    ffn_expansion = 1
    positional_encodings = False
    n_sampling_choices = 0
    pretrained_embeddings = strategy_estimator.embeddings if args.embeddings_from_estimator == 1 else None
    weight_decay = 1e-5
    
    model = ASGAR(features, 
                args.emb_size, 
                args.transformer_heads, 
                args.transformer_layers, 
                ffn_expansion=ffn_expansion, 
                dropout=args.dropout,
                positional_encodings=positional_encodings, 
                n_sampling_choices=n_sampling_choices, 
                pretrained_embeddings=pretrained_embeddings, 
                aligner_model=aligner_model, 
                type_embedding_weight=args.type_embedding_weight,
                wide_ffns=args.wide_ffns, 
                variational=args.variational, 
                cross_attn_mask_encoder_context=args.cross_attn_mask_encoder_context,
                cross_attn_mask_encoder_instance_q=args.cross_attn_mask_encoder_instance_q,
                use_feature_type_memory_mask=args.use_feature_type_memory_mask,
                context_placement=args.context_placement, 
                quantized_placement=args.quantized_placement,
                instancing_type=args.instancing_type,
                levels=args.fsq_levels,
                n_quantized_tokens=args.n_quantized_tokens,
                candidating_members=args.candidating_members,
                encoder_only=args.encoder_only,
                use_special_embedding=args.use_special_embedding,
                masking_rate=args.masking_rate,
                )
    model = model.to(get_torch_device())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=weight_decay)
    scheduler = None
    if args.lr_scheduler == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, 
            base_lr=args.base_lr, 
            max_lr=args.max_lr, 
            step_size_up=args.lr_scheduler_step_size, 
            cycle_momentum=False, 
            mode='triangular2'
        )
    elif args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor = args.lr_scheduler_factor,
            patience = args.lr_scheduler_patience
        )
    return model, optimizer, scheduler