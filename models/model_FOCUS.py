# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
from os.path import join as pjoin
from .model_utils import *
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from torch.nn import functional as F
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

class TextEncoder(nn.Module):
    def __init__(self, conch_model):
        super().__init__()
        self.transformer = conch_model.text.transformer
        self.positional_embedding = conch_model.text.positional_embedding
        self.ln_final = conch_model.text.ln_final
        self.text_projection = conch_model.text.text_projection
        # Get dtype from one of the model's parameters
        self.dtype = next(conch_model.parameters()).dtype

    def forward(self, prompts, tokenized_prompts):
        # Rest of the code remains the same
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[:, 0] @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, classnames, conch_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = ""
        dtype = next(conch_model.parameters()).dtype
        ctx_dim = conch_model.text.ln_final.weight.shape[0]
        
        # Get the tokenizer
        self.tokenizer = get_tokenizer()

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            # Use the correct tokenize function with both arguments
            prompt = tokenize(self.tokenizer, [ctx_init])
            with torch.no_grad():
                embedding = conch_model.text.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if False:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        # Process class names
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [name for name in classnames]
        
        # Use the correct tokenize function with both arguments
        tokenized_prompts = tokenize(self.tokenizer, prompts)
        
        with torch.no_grad():
            embedding = conch_model.text.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        # Use the tokenizer's encode method for getting lengths
        self.name_lens = [len(self.tokenizer.encode(name, 
                                                   max_length=127,
                                                   truncation=True)) 
                         for name in classnames]
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=1,
            )
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError
        return prompts

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)




class FOCUS(nn.Module):
    def __init__(self, config, num_classes=3):
        super(FOCUS, self).__init__()
        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.window_size = config.window_size
        self.sim_threshold = config.sim_threshold

        # Flags từ config
        self.use_prompt = getattr(config, "use_prompt", False)
        self.use_KAVTC = getattr(config, "use_KAVTC", False)
        self.use_SVTC = getattr(config, "use_SVTC", False)
        self.use_CrossAgg = getattr(config, "use_CrossAgg", False)

        self.L = 768
        self.D = 512
        self.L_max = config.max_context_length

        # Feature encoder base - ALWAYS needed
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.LayerNorm(self.D),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        # Initialize text components if any text-related module is used
        if self.use_prompt or self.use_KAVTC or self.use_SVTC or self.use_CrossAgg:
            conch_model_cfg = 'conch_ViT-B-16'
            conch_checkpoint_path = 'ckpts/conch.pth'
            conch_model, _ = create_model_from_pretrained(conch_model_cfg, conch_checkpoint_path)
            _ = conch_model.eval()

            # Use text projection dimension for consistency
            self.text_dim = conch_model.text.text_projection.shape[1]  # 512
            self.prompt_learner = PromptLearner(config.text_prompt, conch_model.float())
            self.text_encoder = TextEncoder(conch_model.float())
            
            # Project visual features to match text dimension
            self.feature_projector = nn.Linear(self.D, self.text_dim)
            self.classifier_dim = self.text_dim
        else:
            self.text_dim = None
            self.feature_projector = None
            self.classifier_dim = self.D

        # Cross-attention components
        if self.use_CrossAgg:
            num_heads = 8
            self.head_dim = self.text_dim // num_heads
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.text_dim,
                num_heads=num_heads,
                batch_first=True
            )

        # MIL aggregation
        self.attention_weights = nn.Linear(self.classifier_dim, 1)
        
        # Classifier
        self.classifier = nn.Linear(self.classifier_dim, num_classes)

    def adaptive_token_selection(self, features, text_features, debug=False):
        """
        KAVTC: Knowledge-enhanced Adaptive Visual Token Compression
        """
        batch_size, n_tokens, feature_dim = features.shape
        
        if debug:
            print(f"KAVTC input: {features.shape}")
        
        # Compute relevance scores using text guidance
        # features: [B, N, D], text_features: [C, D] 
        text_mean = text_features.mean(dim=0, keepdim=True)  # [1, D]
        text_mean = text_mean.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 1, D]
        
        # Compute similarity between each token and text
        features_norm = F.normalize(features, dim=-1)  # [B, N, D]
        text_norm = F.normalize(text_mean, dim=-1)  # [B, 1, D]
        
        # Similarity scores: [B, N]
        similarity = torch.bmm(features_norm, text_norm.transpose(-1, -2)).squeeze(-1)
        
        # Adaptive selection: keep top-k tokens per sample
        keep_ratio = 0.7  # Keep 70% of tokens
        num_keep = max(1, int(n_tokens * keep_ratio))
        
        # Select top tokens for each sample
        _, top_indices = torch.topk(similarity, num_keep, dim=1)
        top_indices, _ = torch.sort(top_indices, dim=1)  # Maintain order
        
        # Gather selected tokens
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_keep)
        selected_features = features[batch_indices, top_indices]  # [B, num_keep, D]
        
        if debug:
            print(f"KAVTC output: {selected_features.shape} (kept {num_keep}/{n_tokens} tokens)")
        
        return selected_features

    def spatial_token_compression(self, features, text_features=None, debug=False):
        """
        SVTC: Sequential Visual Token Compression
        """
        batch_size, n_tokens, feature_dim = features.shape
        
        if debug:
            print(f"SVTC input: {features.shape}")
        
        compressed_features = []
        
        for b in range(batch_size):
            tokens = features[b]  # [N, D]
            
            if n_tokens <= 2:
                compressed_features.append(tokens)
                continue
            
            # Compute pairwise similarities
            tokens_norm = F.normalize(tokens, dim=-1)
            sim_matrix = torch.mm(tokens_norm, tokens_norm.T)  # [N, N]
            
            # Remove self-similarity
            sim_matrix.fill_diagonal_(0)
            
            # Find redundant tokens
            avg_sim = sim_matrix.mean(dim=1)  # Average similarity to other tokens
            threshold = avg_sim.mean() + 0.5 * avg_sim.std()
            
            # Keep tokens with low average similarity (more unique)
            keep_mask = avg_sim < threshold
            
            # Ensure we keep at least 30% of tokens
            min_keep = max(1, int(0.3 * n_tokens))
            if keep_mask.sum() < min_keep:
                _, indices = torch.topk(-avg_sim, min_keep)  # Keep least similar
                keep_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
                keep_mask[indices] = True
            
            compressed_tokens = tokens[keep_mask]
            compressed_features.append(compressed_tokens)
        
        # Pad to same length
        max_len = max(feat.shape[0] for feat in compressed_features)
        padded_features = []
        
        for feat in compressed_features:
            if feat.shape[0] < max_len:
                padding = torch.zeros(max_len - feat.shape[0], feature_dim, 
                                    device=feat.device, dtype=feat.dtype)
                feat = torch.cat([feat, padding], dim=0)
            padded_features.append(feat)
        
        result = torch.stack(padded_features, dim=0)
        
        if debug:
            print(f"SVTC output: {result.shape}")
        
        return result

    def forward(self, x_s, x_l, label, debug=False):
        """
        Fixed forward pass with proper feature flow
        """
        # Encode visual features
        features = self.feature_encoder(x_l.float())  # [N, D=512]
        
        if debug:
            print(f"Encoded features: {features.shape}")
        
        # Add batch dimension for single sample
        if features.dim() == 2:
            features = features.unsqueeze(0)  # [1, N, D]
        
        # Get text features if needed
        text_features = None
        if self.use_prompt or self.use_KAVTC or self.use_SVTC or self.use_CrossAgg:
            prompts = self.prompt_learner()  # [C, seq_len, dim]
            # FIX: Don't slice the text features - use all class embeddings
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)  # [C, text_dim]
            
            # Project visual features to text dimension
            features = self.feature_projector(features)  # [B, N, text_dim]
            
            if debug:
                print(f"Text features: {text_features.shape}")
                print(f"Projected features: {features.shape}")

        # Apply KAVTC
        if self.use_KAVTC and text_features is not None:
            features = self.adaptive_token_selection(features, text_features, debug)

        # Apply SVTC  
        if self.use_SVTC:
            features = self.spatial_token_compression(features, text_features, debug)

        # Apply Cross-modal Attention
        if self.use_CrossAgg and text_features is not None:
            batch_size = features.shape[0]
            
            # Expand text features for each sample in batch
            text_expanded = text_features.unsqueeze(0).expand(batch_size, -1, -1)  # [B, C, D]
            
            # Cross attention: visual features attend to text features
            attended_features, _ = self.cross_attention(
                query=features,      # [B, N, D] - visual queries
                key=text_expanded,   # [B, C, D] - text keys  
                value=text_expanded  # [B, C, D] - text values
            )
            
            # Combine with original features
            alpha = 0.5
            features = alpha * features + (1 - alpha) * attended_features
            
            if debug:
                print(f"Cross-attention output: {features.shape}")

        # MIL Aggregation with attention
        attention_scores = self.attention_weights(features)  # [B, N, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, N, 1]
        
        # Weighted aggregation
        aggregated_features = torch.sum(attention_weights * features, dim=1)  # [B, D]
        
        if debug:
            print(f"Aggregated features: {aggregated_features.shape}")

        # Classification
        logits = self.classifier(aggregated_features)  # [B, num_classes]
        
        # Handle single sample case
        if logits.shape[0] == 1 and label.dim() == 0:
            label = label.unsqueeze(0)
        
        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        
        if debug:
            print(f"Logits: {logits.shape}, Loss: {loss.item():.4f}")
            print("=" * 50)
        
        return Y_prob, Y_hat, loss

    def get_attention_weights(self, x_s, x_l):
        """
        Get attention weights for visualization
        """
        with torch.no_grad():
            features = self.feature_encoder(x_l.float())
            if features.dim() == 2:
                features = features.unsqueeze(0)
                
            if hasattr(self, 'feature_projector') and self.feature_projector is not None:
                features = self.feature_projector(features)
                
            attention_scores = self.attention_weights(features)
            attention_weights = F.softmax(attention_scores, dim=1)
            
        return attention_weights.squeeze(0).squeeze(-1)  # [N]