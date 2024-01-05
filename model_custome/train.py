import os
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from model_customer import M2TRWithCrossAttention
import torch 

# Định nghĩa đường dẫn đến thư mục train
train_data_path = '/Users/lap01743/Downloads/WorkSpace/Capstone_project/CVPR_2023/face_attribute_attack/data/train'

# Áp dụng các biến đổi cho ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Đổi kích thước ảnh về 224x224 (tùy chọn kích thước)
    transforms.ToTensor(),
    # Thêm các biến đổi khác nếu cần
])

# Tạo dataset từ thư mục train
train_dataset = datasets.ImageFolder(train_data_path, transform=transform)

# Tạo DataLoader
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Khởi tạo mô hình và bộ tối ưu hóa
model = M2TRWithCrossAttention(input_channels=3, patch_size_s=7, patch_size_l=64,
                                linear_projection_size=256, transformer_input_size=256, num_heads=4)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

# Huấn luyện mô hình
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_images, batch_labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels.float().view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}')


# Lưu trữ mô hình sau khi huấn luyện
torch.save(model.state_dict(), 'm2tr_model.pth')
