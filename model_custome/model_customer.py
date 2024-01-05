import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBranch(nn.Module):
    def __init__(self, input_channels, output_channels, patch_size):
        super(CNNBranch, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=patch_size, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        return x

class LinearProjection(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearProjection, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear(x)
        return x


class MultiScaleTransformer(nn.Module):
    def __init__(self, input_size, num_heads):
        super(MultiScaleTransformer, self).__init__()
        self.num_heads = num_heads
        self.self_attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads)
        self.fc = nn.Linear(input_size, input_size)
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x):
        # Dynamically determine the embedding dimension
        input_size = x.size(-1)
        
        # Adjust the number of heads based on the input size
        num_heads = min(self.num_heads, input_size // 64)
        
        # Update the embedding dimension for both attention and linear layer
        self.self_attention.embed_dim = input_size
        self.self_attention.num_heads = num_heads
        self.fc.in_features = input_size

        x = self.self_attention(x, x, x)[0]
        x = F.relu(self.fc(x))
        x = self.layer_norm(x)
        return x



class CrossAttention(nn.Module):
    def __init__(self, input_size):
        super(CrossAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=1)
        self.fc = nn.Linear(input_size, input_size)
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x1, x2):
        # Linear projections for both inputs
        x1_linear = self.fc(x1.view(x1.size(0), -1))
        x2_linear = self.fc(x2.view(x2.size(0), -1))

        # Concatenate the linear projections along the feature dimension
        x_combined = torch.cat((x1_linear, x2_linear), dim=1)

        # Apply cross-attention
        cross_att = self.cross_attention(x_combined, x_combined, x_combined)[0]

        # Additional processing
        x = F.relu(self.fc(cross_att))
        x = self.layer_norm(x)
        return x

class FrequencyFilter(nn.Module):
    def __init__(self, input_size):
        super(FrequencyFilter, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return F.relu(self.fc(x))

class CrossModalityFusion(nn.Module):
    def __init__(self, input_size):
        super(CrossModalityFusion, self).__init__()
        self.fc = nn.Linear(input_size * 2, 1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return torch.sigmoid(self.fc(x))

class M2TRWithCrossAttention(nn.Module):
    def __init__(self, input_channels, patch_size_s, patch_size_l, linear_projection_size, transformer_input_size, num_heads):
        super(M2TRWithCrossAttention, self).__init__()

        self.s_branch = CNNBranch(input_channels, 64, patch_size_s)
        self.l_branch = CNNBranch(input_channels, 64, patch_size_l)

        self.linear_projection_s = LinearProjection(64, linear_projection_size)
        self.linear_projection_l = LinearProjection(64, linear_projection_size)

        self.cross_attention = CrossAttention(linear_projection_size)

        self.multi_scale_transformer = MultiScaleTransformer(transformer_input_size, num_heads)
        self.cross_attention = CrossAttention(transformer_input_size )

        self.frequency_filter = FrequencyFilter(transformer_input_size)
        self.cross_modality_fusion = CrossModalityFusion(transformer_input_size)

        # Add a final linear layer for binary classification
        self.final_linear = nn.Linear(transformer_input_size * 2, 1)  # Update the input size here
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_s = self.s_branch(x)
        x_l = self.l_branch(x)
        # print("x_s", x_s)
        # print("x_l", x_l)
        # Apply adaptive average pooling to ensure a consistent input size
        x_s_pooled = F.adaptive_avg_pool2d(x_s, (1, 1))
        x_l_pooled = F.adaptive_avg_pool2d(x_l, (1, 1))
        # print("x_l_pooled", x_s_pooled)
        # print("x_l_pooled", x_l_pooled)
        x_s_linear = self.linear_projection_s(x_s_pooled.view(x_s.size(0), -1))
        x_l_linear = self.linear_projection_l(x_l_pooled.view(x_l.size(0), -1))
        # print("linear_projection_s", x_s_linear)
        # print("linear_projection_l", x_l_linear)

        # Apply cross-attention to connect linear_projection_s and linear_projection_l
        x_combined = self.cross_attention(x_s_linear, x_l_linear)
        print("x_combined", x_combined)
        # Pass the combined matrix to MultiScaleTransformer
        x_transformer = self.multi_scale_transformer(x_combined)
        x_frequency_filter = self.frequency_filter(x_transformer)
        x_cross_modality_fusion = self.cross_modality_fusion(x_transformer, x_frequency_filter)

        # Concatenate the output of cross_attention and frequency_filter
        # x_combined = torch.cat((x_cross_attention, x_frequency_filter), dim=1)

        # Final linear layer and sigmoid activation for binary classification
        x_final = self.sigmoid(self.final_linear(x_combined))
        return x_final

# # Example instantiation of the model
# model = M2TRWithCrossAttention(input_channels=3, patch_size_s=7, patch_size_l=64,
#                                 linear_projection_size=256, transformer_input_size=256, num_heads=4)

# # In ra cấu trúc của mô hình
# print(model)