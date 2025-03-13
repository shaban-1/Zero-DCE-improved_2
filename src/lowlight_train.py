import os
import time
import torch
import torch.optim
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import dataloader
import model
import Myloss


def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        m.weight.data.normal_(0.0, 0.02)
    elif 'BatchNorm' in classname:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    DCE_net = model.enhance_net_nopool().to(device)
    DCE_net.apply(weights_init)

    if config.load_pretrain:
        DCE_net.load_state_dict(torch.load(config.pretrain_dir, map_location=device))
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)

    criterion = Myloss.CustomLoss()
    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    DCE_net.train()
    writer = SummaryWriter()

    for epoch in range(config.num_epochs):
        start_time = time.time()
        loss_list = []

        print(f"\nEpoch {epoch + 1}/{config.num_epochs}:")
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}", unit="batch")

        for iteration, img_lowlight in progress_bar:
            img_lowlight = img_lowlight.to(device)

            enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)
            loss = criterion(enhanced_image, img_lowlight, A)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            loss_list.append(loss.item())

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            if (iteration + 1) % config.snapshot_iter == 0:
                writer.add_scalar('Loss/train', loss, 500 * epoch + iteration + 1)
                torch.save(DCE_net.state_dict(), os.path.join(config.snapshots_folder, f"Epoch{epoch + 1}.pth"))

        avg_loss = np.mean(loss_list)
        epoch_time = time.time() - start_time
        speed = len(train_loader) / epoch_time

        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s ({speed:.2f} batches/sec), Avg Loss: {avg_loss:.4f}")

        writer.add_scalar('Loss/train-epoch-mean', avg_loss, epoch + 1)

    writer.close()


if __name__ == "__main__":
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots/pre-train.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)