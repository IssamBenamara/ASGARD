import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import numpy as np
from utils import construct_embedding_module, get_torch_device

class AlignerModule(nn.Module):

    def forward(self, batch, from_logits=False, logits_dict=None, softmax_weight=10):
        pass

    def multi_projection(self, batch, from_logits=False, logits_dict=None, softmax_weight=10):
        pass

    def projection(self, batch, from_logits=False, logits_dict=None, softmax_weight=10):
        pass

    @torch.no_grad()
    def cosine_similarity(self, source, target):
        pass
    

class ProjectionHead(nn.Module):

    def __init__(
        self,
        embedding_dim,
        projection_dim=128,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):

        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPModel(AlignerModule):

    def __init__(
        self,
        strategy_encoder,
        temperature=1.0,
        strategy_embedding=128,
        projection_dim=128,
        double_projection=True,
    ):
        super().__init__()

        self.device = get_torch_device()
        self.strategy_encoder = strategy_encoder
        self.strategy_encoder.eval()
        for p in self.strategy_encoder.parameters():
            p.requires_grad = False

        self.source_projection = ProjectionHead(embedding_dim=strategy_embedding, projection_dim=projection_dim)
        if double_projection:
            self.target_projection = ProjectionHead(embedding_dim=strategy_embedding, projection_dim=projection_dim)
        else:
            self.target_projection = self.source_projection
        self.temperature = temperature

    def forward(self, batch, identity_only=False, add_identity=True, inference=False, from_logits=False, logits_dict=None, softmax_weight=10):

        # Getting Image and Text Features
        source_features = self.strategy_encoder(batch["source"], latent_only=True, from_logits=from_logits, logits_dict=logits_dict, softmax_weight=softmax_weight)
        target_features = self.strategy_encoder(batch["target"], latent_only=True)

        # Getting Image and Text Embeddings (with same dimension)
        source_embeddings = self.source_projection(source_features)
        target_embeddings = self.target_projection(target_features)

        # Calculating the Loss
        logits = (target_embeddings @ source_embeddings.T) / self.temperature

        if inference:
            return source_embeddings, target_embeddings, logits
        
        # Filling the diagonal with 1's because sometimes it is missed by softmax
        identity_mat = torch.eye(source_embeddings.size(0), device=self.device)
        if identity_only:
            targets = identity_mat
        else:
            sources_similarity = source_embeddings @ source_embeddings.T
            targets_similarity = target_embeddings @ target_embeddings.T
            targets = F.softmax(
                (sources_similarity + targets_similarity) / 2 * self.temperature, dim=-1
            )
            if add_identity:
                targets = (1 - identity_mat)*targets + identity_mat
        
        targets_loss = cross_entropy(logits, targets, reduction='none')
        sources_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (sources_loss + targets_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    

class CLIPCosineLoss(nn.Module):

    def __init__(
        self,
        state_dict_path,
        strategy_encoder,
        temperature=1.0,
        strategy_embedding=64,
        projection_dim=128,
        double_projection=True,
        weight=2.0,

    ):
        super().__init__()

        self.device = get_torch_device()
        self.clip_model = CLIPModel(strategy_encoder, 
                                    strategy_embedding=strategy_embedding,
                                    temperature=temperature, 
                                    projection_dim=projection_dim, 
                                    double_projection=double_projection).to(self.device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.load_state_dict(torch.load(state_dict_path, map_location=self.device))
        self.weight = weight

    def forward(self, original_ix_tensors, generated_ix_tensors, from_logits=False, logits_dict=None, softmax_weight=10):
        

        source_embeddings, target_embeddings, _ = self.clip_model(
            {
                "source":original_ix_tensors,
                "target":generated_ix_tensors,
            },
            inference = True,
            from_logits = from_logits,
            logits_dict = logits_dict,
            softmax_weight = softmax_weight,
        )
        
        return self.weight * torch.pow(F.cosine_similarity(source_embeddings, target_embeddings).unsqueeze(1) - 1, 2)


class TripletModel(AlignerModule):

    def __init__(
        self,
        strategy_encoder,
        margin=1.0,
        strategy_embedding=128,
        projection_dim=128,
    ):
        super().__init__()

        self.device = get_torch_device()
        self.strategy_encoder = strategy_encoder
        self.strategy_encoder.eval()
        for p in self.strategy_encoder.parameters():
            p.requires_grad = False

        self.projection_dim = projection_dim
        self.projection_head = ProjectionHead(embedding_dim=strategy_embedding, projection_dim=projection_dim)
        self.margin = margin
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=self.positivized_cosine, margin=self.margin, reduction='none')


    def forward(self, batch, from_logits=False, logits_dict=None, softmax_weight=10):
        
        source_embeddings, positive_target_embeddings, negative_target_embeddings = self.multi_projection(batch, from_logits=from_logits, logits_dict=logits_dict, softmax_weight=softmax_weight)
        loss = self.triplet_loss(source_embeddings, positive_target_embeddings, negative_target_embeddings)
        return loss

    def multi_projection(self, batch, from_logits=False, logits_dict=None, softmax_weight=10):

        # Getting Features
        # Source has extra parameters to allow for gradient flow if needed
        source_features = self.strategy_encoder(batch["source"], latent_only=True, from_logits=from_logits, logits_dict=logits_dict, softmax_weight=softmax_weight)
        positive_target_features = self.strategy_encoder(batch["positive_target"], latent_only=True)
        negative_target_features = self.strategy_encoder(batch["negative_target"], latent_only=True)

        # Getting Embeddings (with same dimension)
        source_embeddings = self.projection_head(source_features)
        positive_target_embeddings = self.projection_head(positive_target_features)
        negative_target_embeddings = self.projection_head(negative_target_features)

        return source_embeddings, positive_target_embeddings, negative_target_embeddings

    def projection(self, batch, from_logits=False, logits_dict=None, softmax_weight=10):

        source_features = self.strategy_encoder(batch, latent_only=True, from_logits=from_logits, logits_dict=logits_dict, softmax_weight=softmax_weight)
        source_embeddings = self.projection_head(source_features)

        return source_embeddings

    def positivized_cosine(self, x, y):
        return 1.0 - F.cosine_similarity(x, y)
    
    @torch.no_grad()
    def cosine_similarity(self, source, target):
        source_embeddings = self.projection(source)
        target_embeddings = self.projection(target)
        return F.cosine_similarity(source_embeddings, target_embeddings).detach().cpu().numpy()
    

class AlignerLoss(nn.Module):

    def __init__(
        self,
        aligner_model,
        weight=2.0,
    ):
        super().__init__()

        self.device = get_torch_device()
        self.aligner_model = aligner_model
        self.aligner_model.eval()
        for p in self.aligner_model.parameters():
            p.requires_grad = False
        self.weight = weight

    def forward(self, generated_ix_tensors, original_ix_tensors, softmax_weight=10):
        source_embeddings = self.aligner_model.projection(generated_ix_tensors)
        target_embeddings = self.aligner_model.projection(original_ix_tensors)
        return self.weight * torch.pow(F.cosine_similarity(source_embeddings, target_embeddings).unsqueeze(1) - 1, 2)
    
    def from_logits(self, logits_dict, target_ix_tensors, softmax_weight=10):
        # target_ix_tensors doesn't do anything here, logits dict is what's taken into account
        source_embeddings = self.aligner_model.projection(target_ix_tensors, from_logits=True, logits_dict=logits_dict, softmax_weight=softmax_weight)
        target_embeddings = self.aligner_model.projection(target_ix_tensors)
        return self.weight * torch.pow(F.cosine_similarity(source_embeddings, target_embeddings).unsqueeze(1) - 1, 2)
