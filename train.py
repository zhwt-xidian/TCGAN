import numpy as np
import torch
from torch import optim, nn
from visual_tools import plot_two_subplots
from options.ADFECG_parameter import parser
from data.ADFECGDB_dataloader import get_ADFECGDB_dataloader
from model import Generator, Discriminator
from MyLoss import BCELoss
import matplotlib.pyplot as plt
import os
import time

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------Hyper-parameters----------------------------------------------
# get hyper-parameters
args = parser.parse_args()
dataroot = args.root_dir
batch_size = args.batch_size
overlap = args.overlap
G_lr = args.G_lr
D_lr = args.D_lr
input_size = args.input_size
Gkernel_size = args.Gkernel_size
dropout = args.dropout
input_length = args.input_length
Dkernel_size = args.Dkernel_size
epochs = args.epochs
save_model_path = args.save_model_path
save_valiation_dir = args.save_valiation_dir
save_train_dir = args.save_train_dir
save_interval = args.save_interval
# ----------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------load dataset-----------------------------------------------------
train_dataloader, test_dataloader = get_ADFECGDB_dataloader(dataroot, batch_size=batch_size, overlap=overlap)
print("the dataset has been loaded successfully!")
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------build network object----------------------------------------------
generator = Generator(input_size=input_size, Gkernel_size=Gkernel_size, dropout=dropout).to(device)
discriminator = Discriminator(input_length=input_length, stride=1, Dkernel_size=Dkernel_size).to(device)
# ----------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------Optimizer------------------------------------------------------------
d_optimizer = optim.Adam(discriminator.parameters(), D_lr)
g_optimizer = optim.Adam(generator.parameters(), G_lr)
# ----------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------loss function -------------------------------------------------------
loss_func1 = nn.L1Loss()
loss_func1 = loss_func1.to(device)
loss_func2 = BCELoss()
loss_func2 = loss_func2.to(device)
# ----------------------------------------------------------------------------------------------------------------------

# =================================================Train Model==========================================================
t1 = time.time()
D_Loss = []
G_Loss = []
for epoch in range(epochs):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(train_dataloader)
    for step, (AECG, FECG) in enumerate(train_dataloader):
        AECG = AECG.to(device)
        FECG = FECG.to(device)


        #------------------------------------------------
        #                   train D Network
        #------------------------------------------------
        d_optimizer.zero_grad()
        real_output = discriminator(FECG)
        d_real_loss = loss_func2(real_output, torch.ones_like(real_output))
        print("real_output", real_output)
        d_real_loss.backward()
        gen_FECG = generator(AECG)
        fake_output = discriminator(gen_FECG.detach())
        print("fake_output", fake_output)
        print("==============================================================")
        d_fake_loss = loss_func2(fake_output, torch.zeros_like(fake_output))
        d_fake_loss.backward()
        d_loss = d_real_loss + d_fake_loss
        d_optimizer.step()

        #-----------------------------------------------------------
        #                      train G Network
        #-----------------------------------------------------------
        g_optimizer.zero_grad()
        gen_FECG = generator(AECG)
        fake_output = discriminator(gen_FECG)
        g_loss1 = loss_func2(fake_output, torch.ones_like(fake_output))
        g_loss2 = loss_func1(gen_FECG.reshape(-1), FECG.reshape(-1))
        g_loss = 0.3*g_loss1 + 0.7*g_loss2
        g_loss.backward()
        g_optimizer.step()

        with torch.no_grad():
            d_epoch_loss += d_loss.item()
            g_epoch_loss += g_loss.item()
            print("Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}".format(epoch, epochs,
                                                                                       step, count,
                                                                                       d_loss.item(),
                                                                                       g_loss.item()))
            if step % 100 == 0:
                y1 = gen_FECG.clone()
                y2 = AECG.clone()
                y3 = FECG.clone()
                y1 = y1.squeeze().cpu()
                y2 = y2.squeeze().cpu()
                y3 = y3.squeeze().cpu()
                x = np.linspace(0, 1, 200)
                y1 = y1.numpy()
                y2 = y2.numpy()
                y3 = y3.numpy()

                plt.figure(step)
                plot_two_subplots(x=np.linspace(0, 1, 200), y1=y1[0], y2=y2[0], y3=y3[0],
                                  title1='Gen_FECG', title2='AECG', title3='FECG', label1='Gen_FECG', label2='AECG', label3='FECG')
                fileN = "train"+str(epoch)+"_"+str(step)+".png"
                save_path = os.path.join(save_train_dir, fileN)
                plt.savefig(save_path)

    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_Loss.append(d_epoch_loss)
        G_Loss.append(g_epoch_loss)
        print('Epoch:', epoch)

    # save Model
    generator.eval()
    FSE_list = []
    Gen_FECG_list = []
    Loss_valiation = []
    loss_epoch_valiation = 0
    count_eval = len(test_dataloader)

    with torch.no_grad():
        for n_step, (Orign_AECG, FSE) in enumerate(test_dataloader):
            Orign_AECG = Orign_AECG.to(device)
            FSE = FSE.to(device)
            output = generator(Orign_AECG)
            Gen_FECG_list.append(output)
            FSE_list.append(FSE)
            loss_epoch_valiation += loss_func1(output, FSE)
            loss_epoch_valiation /= count_eval
            Loss_valiation.append(loss_epoch_valiation)
            # save test results
            if n_step % 100 == 0:
                y1 = output.clone()
                y2 = Orign_AECG.clone()
                y3 = FSE.clone()
                y1 = y1.squeeze().cpu()
                y2 = y2.squeeze().cpu()
                y3 = y3.squeeze().cpu()
                y1 = y1.detach().numpy()
                y2 = y2.detach().numpy()
                y3 = y3.detach().numpy()

                plt.figure(n_step)
                plot_two_subplots(x=np.linspace(0, 1, 200), y1=y1, y2=y2, y3=y3,
                            title1='Gen_FECG', title2='Orign_AECG', title3='FSE', label1='Gen_FECG', label2='Orign_AECG',
                            label3='FSE')
                fileN = "valiation" + str(epoch) + "_" + str(n_step) + ".png"
                save_path = os.path.join(save_valiation_dir, fileN)
                plt.savefig(save_path)

    loss_epoch_valiation /= count_eval
    Loss_valiation.append(loss_epoch_valiation)

    Gen_FECG_list = [tensor.cpu().numpy().flatten() for tensor in Gen_FECG_list]
    Gen_FECG_list = np.concatenate(Gen_FECG_list)
    FSE_list = [tensor.cpu().numpy().flatten() for tensor in FSE_list]
    FSE_list = np.concatenate(FSE_list)

    if epoch % 10 == 0:
        fileSigName = ["GenFECG_" + str(epoch) + ".csv", "FSE_" + str(epoch) + ".csv"]
        valiation_GenFECG_PATH = os.path.join(save_valiation_dir, fileSigName[0])
        valiation_FSE_PATH = os.path.join(save_valiation_dir, fileSigName[1])
        np.savetxt(valiation_GenFECG_PATH, Gen_FECG_list, delimiter=',')
        np.savetxt(valiation_FSE_PATH, FSE_list, delimiter=',')

    if epoch % save_interval == 0:
        generator_name = "generator" + str(epoch) + ".pth"
        discriminator_name = "discriminator" + str(epoch) + ".pth"
        generator_path = os.path.join(save_model_path, generator_name)
        discriminator_path = os.path.join(save_model_path, discriminator_name)
        torch.save(generator, generator_path)
        torch.save(discriminator, discriminator_path)