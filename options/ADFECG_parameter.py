import argparse

# These options are used to train model.
parser = argparse.ArgumentParser(description='The hyper parameters')

# -------------------------hyper-parameter-------------------------------
# Generator Network
parser.add_argument('--input_size', type=int, default=1, help='the number of input channel')
parser.add_argument('--Gkernel_size', type=int, default=3, help='kernel size of Generator')
parser.add_argument('--dropout', type=float, default=0.2, help='[0,1]')
# Discriminator Network
parser.add_argument('--input_length', type=int, default=200)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--Dkernel_size', type=int, default=3)

# DataSet
parser.add_argument('--which_dataset', type=str, default='ADFECGDB')
parser.add_argument('--root_dir', type=str, default="./Dataset/ADFECGDB")
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--overlap', type=float, default=0.5)

# train
parser.add_argument('--G_lr', type=float, default=0.001)
parser.add_argument('--D_lr', type=float, default=0.00005)
parser.add_argument('--epochs', type=int, default=500)

# save
parser.add_argument('--save_model_path', type=str, default="./checkpoint/ADFECGDB")
parser.add_argument('--save_interval', type=int, default=10)
parser.add_argument('--save_train_dir', type=str, default="./Result/ADFECGDB/Results_train")
parser.add_argument('--save_valiation_dir', type=str, default="./Result/ADFECGDB/Results_valiation")