import argparse
import csv
import os
import random
import time

import lossfunc.lossf as lossf
import matplotlib.pyplot as plt
import models.ReLURaDOBase as models
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dataloader.dataloader import load_dataloader
from sklearn import preprocessing
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from utils.save_fig import save_images

# CLASS_MAP = {"CN": 0, "AD": 1, "MCI":2}
CLASS_MAP = {"CN": 0, "AD": 1, "MCI":2, "PD": 1}
SEED_VALUE = 103

def parser():
    parser = argparse.ArgumentParser(description="example")
    parser.add_argument("--model", type=str, default="RaDOGAGA_improve_Base++")
    # parser.add_argument("--model", type=str, default="RaDOGAGA_w_original_Base++")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=700)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--log", type=str, default="z-1400")
    parser.add_argument("--n_train", type=float, default=0.8)
    parser.add_argument("--train_or_loadnet", type=str, default="train") # train or loadnet
    parser.add_argument("--beta_d1", type=float, default=1.0)
    parser.add_argument("--beta_d2", type=float, default=3.0)
    parser.add_argument("--beta_kl", type=float, default=1.0)
    parser.add_argument("--beta_cossim", type=float, default=7000.0)
    # parser.add_argument("--conv_model", type=str, default="(8, [[8,1,2],[16,1,2],[32,2,2],[64,2,2]])")
    # parser.add_argument("--conv_model", type=str, default="(16, [[16,1,2],[32,1,2],[64,2,2],[128,2,2]])")
    parser.add_argument("--conv_model", type=str, default="(32, [[32,1,2],[64,1,2],[128,2,2]]), \"this model Activate Func is ReLU\"")
    # parser.add_argument("--conv_model", type=str, default="(32, [[32,1,2],[64,1,2],[128,2,2]]),use leakyReLU function")
    parser.add_argument("--latent_size", type=str, default="z_1400")
    parser.add_argument("--last_path", type=str, default="rec1cos7k_Aug_b32_ADCN_20231224_1710/")# rec1p1_cos7k_Aug_batch32_proto1400_2step_AD_CN
    # parser.add_argument("--class", type=list, default=["AD", "CN", "MCI"])
    # parser.add_argument("--aug_method", type=str, default="degree=10")
    parser.add_argument("--save_image", type=str, default="YES") # save image NO or YES
    parser.add_argument("--data_augment", type=bool, default=True) # Augmentation exe? or Non?
    # parser.add_argument("--data_augment", type=bool, default=False) # Augmentation exe? or Non?
    parser.add_argument("--dataset_kind", type=str, default="SkullStripping")
    parser.add_argument("--fc_layer", type=str, default="wo_fc")
    parser.add_argument("--fold_number", type=int, default=4)
    parser.add_argument("--class_set", type=set, default={"CN","AD"})
    parser.add_argument("--day", type=str, default="20231117_18:00:00")
    parser.add_argument("--import_model", type=str, default="import models.ReLURaDOBase as models")

    # parser.add_argument("--fc_layer", type=str, default="with_fc")
    args = parser.parse_args()
    return args


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # この行をFalseにすると再現性はとれるが、速度が落ちる
    return


fix_seed(SEED_VALUE)


def write_csv(epoch, train_loss, val_loss, path):
    with open(path, "a") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, val_loss])


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() and True else "cpu")
    label_encoder = preprocessing.LabelEncoder()
    targets = list(range(0, len(CLASS_MAP), 1))
    label_encoder.fit(targets)

    args = parser()
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-research",
        # track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "d1_w": args.beta_d1,
            "d2_w": args.beta_d2,
            "cos_w": args.beta_cossim,
            "architecture": "RaDOGAGA & improve Baseline++",
            "dataset": "ADNI2",
            "epochs": args.epoch,
        }
    )
    # 12, [[12,1,2],[24,1,2],[32,1,2],[48,2,2]])
    # conv 12 → 24 → 32  ##  conv 24 → 48 → 64  ##  32 → 64 → 128   ##  64 → 128 → 256
    # net = models.ResNetVAE(8, [[8,1,2],[16,1,2],[32,2,2]]).to(device)
    net = models.ResNetVAE(32, [[32,1,2],[64,1,2],[128,2,2]]).to(device)
    # net = models.ResNetVAE(12, [[12,1,2],[24,1,2],[32,2,2],[48,2,2]])
    # net = models.ResNetVAE(64, [[64,1,2],[128,1,2],[256,2,2]])
    log_path = "./logs/"+ args.dataset_kind +"/"+ args.latent_size +"_RaDOwBase++_"+ args.fc_layer +"/"+ args.last_path

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(log_path + "imgs/", exist_ok=True)
    os.makedirs(log_path + "prams/", exist_ok=True)
    os.makedirs(log_path + "val_imgs/", exist_ok=True)
    os.makedirs(log_path + "csv/", exist_ok=True)
    # save args
    with open(log_path + "my_args.txt", "w") as f:
        f.write("{}".format(args))

#   ここでデータをロードする
    fold_number = args.fold_number
    class_set = args.class_set
    train_loader, val_loader, train_dataset, val_dataset = load_dataloader(args.n_train, args.batch_size, args.data_augment, fold_number, class_set)

    path2 = log_path + "train_result.csv"
    with open(path2, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    # hyperparameter
    epochs = args.epoch
    lr = args.lr
    d1_w = args.beta_d1
    d2_w = args.beta_d2
    kl_w = args.beta_kl
    cos_w = args.beta_cossim

    optimizer = optim.Adam(net.parameters(), lr) # print(optimizer)
    scaler = GradScaler(enabled=True)

    # net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    net = net.to(device)
    print(f"This experiment={log_path}")
    print(f"device={device}  d1_weight={d1_w}  d2_weight={d2_w}  kl_weight={kl_w}  cos_sim_weight={cos_w}")
    print(f"conv model = {args.conv_model}  use data = {args.class_set}   Day:{args.day}")
    net = net.apply(init_weights_he_relu)
    train_losses, val_losses = [], []
    train_mse_losses, train_kl_losses, val_mse_losses, val_kl_losses = [], [], [], []
    train_cossim_losses, val_cossim_losses = [], []
    train_se_losses, val_se_losses, train_d2_losses, val_d2_losses = [], [], [], []
    train_acc_losses, val_acc_losses = [], []
    criterion = nn.CrossEntropyLoss()  ##  Note that `CrossEntropyLoss()` = `LogSoftmax` + `NLLLoss`
    start_time = time.time()
    # softmax = nn.Softmax(dim=1)
    val_f1_losses, val_macro_f1_losses = [], []
    # cos_w = 0.0
    print(f"cos_w={cos_w:.3f}")
    for epoch in range(epochs):
        loop_start_time = time.time()
        train_run_mse = 0.0
        train_run_se = 0.0
        train_run_d2 = 0.0
        train_run_kl = 0.0
        train_run_cossim = 0.0
        train_run_loss = 0.0
        counter = 0
        train_acc = 0.0
        net.train()
        for inputs, labels in train_loader:
            labels = torch.tensor(label_encoder.transform(labels))
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with autocast(enabled=True):
                x_rec_mu, x_rec_z, mu, logvar, cos_sim = net(inputs) #  forward: return  x_rec_mu, x_rec_z, mu, logvar, cos_sim
                cos_loss = criterion(cos_sim, labels)
                cos_loss = cos_loss * cos_w
                loss, se, mse, kl, d1, d2= lossf.RaDOBase_loss(x_rec_mu, x_rec_z, mu, logvar, cos_loss, inputs, d1_w, d2_w, kl_w)

            if epoch % 10 == 0:
                if counter == 0:
                    print(f"train data labels = {labels}")
                    print(f"train data cos_sim[0]={cos_sim[0]}")
                    print(f"train data cos_sim[1]={cos_sim[1]}")
                    print(f"train data cos_sim[2]={cos_sim[2]}")

            yhat = torch.argmax(cos_sim, axis=1)
            train_acc += (yhat == labels).sum().item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_run_loss += loss.item()
            train_run_mse += mse.item()
            train_run_se += se.item()
            train_run_d2 += d2.item()
            train_run_kl += kl.item()
            train_run_cossim += cos_loss.item()
            counter += 1

        train_acc /= len(train_loader.dataset)
        train_run_loss /= len(train_loader)
        train_run_mse /= len(train_loader)
        train_run_se /= len(train_loader)
        train_run_d2 /= len(train_loader)
        train_run_kl /= len(train_loader)
        train_run_cossim /= len(train_loader)

        train_losses.append(train_run_loss)
        train_mse_losses.append(train_run_mse)
        train_se_losses.append(train_run_se)
        train_d2_losses.append(train_run_d2)
        train_kl_losses.append(train_run_kl)
        train_cossim_losses.append(train_run_cossim)
        train_acc_losses.append(train_acc)
# __________________________________________________________________________________ #

        net.eval()
        val_run_loss = 0.0
        val_run_mse = 0.0
        val_run_se = 0.0
        val_run_d2 = 0.0
        val_run_kl = 0.0
        val_run_cossim = 0.0
        test_acc = 0.0
        counter = 0
        with torch.inference_mode():
            for inputs, labels in val_loader:
                labels = torch.tensor(label_encoder.transform(labels))
                inputs = inputs.to(device)
                labels = labels.to(device)
                with autocast(enabled=True):
                    x_rec_mu, x_rec_z, mu, logvar, cos_sim = net(inputs)
                    cos_loss = criterion(cos_sim, labels)
                    # cos_loss = criterion(cos_sim, targets)
                    cos_loss = cos_loss * cos_w
                    loss, se, mse, kl, d1, d2= lossf.RaDOBase_loss(x_rec_mu, x_rec_z, mu, logvar, cos_loss, inputs, d1_w, d2_w, kl_w)
                    # loss = d1_w*d1 + d2_w*d2 + kl_w*kl + cos_loss * cos_w

                if epoch % 10 == 0:
                    if counter == 0:
                        print(f"val data labels={labels}")
                        # print(f"cos_sim[0].shape = {cos_sim[0].shape}")
                        print(f"val data cos_sim[0]={cos_sim[0]}")
                        print(f"val data cos_sim[1]={cos_sim[1]}")
                        print(f"val data cos_sim[2]={cos_sim[2]}")

                yhat = torch.argmax((cos_sim), axis=1)
                test_acc += (yhat == labels).sum().item()
                val_run_loss += loss.item()
                val_run_mse += mse.item()
                val_run_se += se.item()
                val_run_d2 += d2.item()
                val_run_kl += kl.item()
                val_run_cossim += cos_loss.item()
                counter += 1

            if epoch % 5 == 0:
                train_output_cpu, output_cpu_val = [], []
                train_loader_iter = iter(train_loader)
                image, _ = next(train_loader_iter)
                image = image.to(device)
                rec, _, _, _, _ = net(image)
                image = image.cpu()
                for train_rec in rec:
                    train_output_cpu.append(train_rec.detach().cpu())
                save_images(image, train_output_cpu, epoch, log_path, train=True)

                val_loader_iter = iter(val_loader)
                val_image, _ = next(val_loader_iter)
                val_image = val_image.to(device)
                rec, _, _, _, _ = net(val_image)
                val_image = val_image.cpu()
                for val_rec in rec:
                    output_cpu_val.append(val_rec.detach().cpu())
                save_images(val_image, output_cpu_val, epoch, log_path, train=False)

            True_labels = []
            pred_list = []
            for i, (val_voxel, val_label) in enumerate(val_dataset):
                input_voxel = torch.tensor(val_voxel.reshape(1,1,80,112,80))
                _, _, z, _, cos_sim = net.forward(input_voxel.to(device))
                # pred = softmax(cos_sim)
                yhat = torch.argmax(cos_sim, axis=1)
                # ___________________________ #
                True_labels.append(val_label)
                pred_list.append(yhat[0].to('cpu').numpy())
            f1 = f1_score(True_labels, pred_list)
            macro_f1 = f1_score(True_labels, pred_list, average='macro')


        test_acc /= len(val_loader.dataset)
        val_run_loss /= len(val_loader)
        val_run_mse /= len(val_loader)
        val_run_se /= len(val_loader)
        val_run_d2 /= len(val_loader)
        val_run_kl /= len(val_loader)
        val_run_cossim /= len(val_loader)

        val_f1_losses.append(f1)
        val_macro_f1_losses.append(macro_f1)
        val_losses.append(val_run_loss)
        val_mse_losses.append(val_run_mse)
        val_se_losses.append(val_run_se)
        val_d2_losses.append(val_run_d2)
        val_kl_losses.append(val_run_kl)
        val_cossim_losses.append(val_run_cossim)
        val_acc_losses.append(test_acc)

        savename = f"prams/Imp_RaDOBase++_epo{epoch}.pth"
        torch.save(net.to('cpu').state_dict(), log_path + savename)
#       torch.save(model.state_dict(), log_path + f"softintroVAE_weight_epoch{str(epoch)}.pth")
        net = net.to(device)

        now_time = time.time()
        print(f"Epo[{epoch+1}/{epochs}] Train[mse:{train_run_mse:.5f} se:{train_run_se:.1f} d2:{train_run_d2:.1f} kl:{train_run_kl:.1f} cos:{train_run_cossim:.2f} t_acc:{train_acc:0.3f} loss:{train_run_loss:.1f}]  "
              f"Val[mse:{val_run_mse:.5f} se:{val_run_se:.1f} d2:{val_run_d2:.1f} kl:{val_run_kl:.1f} cos:{val_run_cossim:.2f} loss:{val_run_loss:.1f} v_acc:{test_acc:0.3f}  f1:{f1:.7f} macro_f1:{macro_f1:.7f}]  "
              f"{(now_time - loop_start_time):.0f}s/epo sum:{(now_time - start_time)/60:.0f}分")

        wandb.log({
            'epoch': epoch,
            'Train mse': train_run_mse,
            'Train se': train_run_se,
            'Train d2': train_run_d2,
            'Train kl': train_run_kl,
            'Train cossim': train_run_cossim,
            'Train loss': train_run_loss,
            'Val mse': val_run_mse,
            'Val se': val_run_se,
            'Val d2': val_run_d2,
            'Val kl': val_run_kl,
            'Val cossim': val_run_cossim,
            'Val loss': val_run_loss,
        })


    write_figres(log_path + "/loss.txt", train_losses, val_losses)
    write_kl_losses_onlyvae(log_path + "/train_losses.txt", train_mse_losses, train_kl_losses)
    write_kl_losses_onlyvae(log_path + "/val_losses.txt", val_mse_losses, val_kl_losses)
    write_cos_loss(log_path + "/cos_loss.txt", train_cossim_losses, val_cossim_losses)
    write_cos_losses(log_path + "/cos_losses.txt", train_cossim_losses, val_cossim_losses)
    write_all_losses(
        log_path + "/all_losses.txt",
        train_mse_losses, train_se_losses, train_d2_losses, train_kl_losses, train_cossim_losses, train_acc_losses, train_losses,
        val_mse_losses,     val_se_losses,   val_d2_losses,   val_kl_losses,   val_cossim_losses,   val_acc_losses,   val_losses,
        val_f1_losses, val_macro_f1_losses,
    )

    # ================= ここからは、Wstar_layerのみを学習する ==================
    net.train()
    cos_w = args.beta_cossim
    print(f"cos_w={cos_w:.3f}")
    for name, param in net.named_parameters():
        if not name == 'encoder.Wstar_layer.weight':
            param.requires_grad = False
        print(name, f"| grad:{param.requires_grad}")
    optimizer = torch.optim.Adam([param for param in net.parameters() if param.requires_grad], lr=5e-4)
    scaler = GradScaler(enabled=True)
    epochs = 50
    for epoch in range(epochs):
        loop_start_time = time.time()
        train_run_cossim = 0.0
        train_run_loss = 0.0
        counter = 0
        train_acc = 0.0

        for inputs, labels in train_loader:
            labels = torch.tensor(label_encoder.transform(labels))
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with autocast(enabled=True):
                x_rec_mu, x_rec_z, mu, logvar, cos_sim = net(inputs) #  forward: return  x_rec_mu, x_rec_z, mu, logvar, cos_sim
                cos_loss = criterion(cos_sim, labels)
                loss = cos_loss * cos_w

            if epoch % 10 == 0:
                if counter == 0:
                    print(f"train data labels = {labels}")
                    print(f"train data cos_sim[0]={cos_sim[0]}")
                    print(f"train data cos_sim[1]={cos_sim[1]}")
                    print(f"train data cos_sim[2]={cos_sim[2]}")

            yhat = torch.argmax(cos_sim, axis=1)
            train_acc += (yhat == labels).sum().item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_run_cossim += cos_loss.item()
            counter += 1

        train_acc /= len(train_loader.dataset)
        train_run_loss /= len(train_loader)
        train_run_cossim /= len(train_loader)

        train_losses.append(train_run_loss)
        train_cossim_losses.append(train_run_cossim)
        
        net.eval()
        val_run_loss = 0.0
        val_run_cossim = 0.0
        test_acc = 0.0
        counter = 0
        with torch.inference_mode():
            for inputs, labels in val_loader:
                labels = torch.tensor(label_encoder.transform(labels))
                inputs = inputs.to(device)
                labels = labels.to(device)
                with autocast(enabled=True):
                    x_rec_mu, x_rec_z, mu, logvar, cos_sim = net(inputs)
                    cos_loss = criterion(cos_sim, labels)
                    loss = cos_loss * cos_w

                if epoch % 10 == 0:
                    if counter == 0:
                        print(f"val data labels={labels}")
                        # print(f"cos_sim[0].shape = {cos_sim[0].shape}")
                        print(f"val data cos_sim[0]={cos_sim[0]}")
                        print(f"val data cos_sim[1]={cos_sim[1]}")
                        print(f"val data cos_sim[2]={cos_sim[2]}")

                yhat = torch.argmax((cos_sim), axis=1)
                test_acc += (yhat == labels).sum().item()
                val_run_loss += loss.item()
                val_run_cossim += cos_loss.item()
                counter += 1

        test_acc /= len(val_loader.dataset)
        val_run_loss /= len(val_loader)
        val_run_cossim /= len(val_loader)

        val_losses.append(val_run_loss)
        val_cossim_losses.append(val_run_cossim)


        if epoch % 5 == 0:
            savename = f"prams/Improve_RaDOBase++_finetune_epo{epoch}.pth"
            torch.save(net.to('cpu').state_dict(), log_path + savename)
    #       torch.save(model.state_dict(), log_path + f"softintroVAE_weight_epoch{str(epoch)}.pth")
            net = net.to(device)

        now_time = time.time()
        print(f"Epoch[{epoch+1}/{epochs}] Train[cossim:{train_run_cossim:.2f} train_acc:{train_acc:0.3f} loss:{train_run_loss:.1f}]  "
              f"Val[cossim:{val_run_cossim:.2f} val_acc:{test_acc:0.3f} loss:{val_run_loss:.1f}]  "
              f"1epo:{(now_time - loop_start_time):.0f}秒 total:{(now_time - start_time)/60:.0f}分")


    if epochs != 0:
        net = net.to('cpu')
        torch.save(net.state_dict(), log_path + "last_resnetvae_weight.pth")
        print("saved ResNetVAE param and ", end="")

    print("finished !")
    # train_result.result_ae(train_loss, val_loss, log_path)
    torch.save(net.state_dict(), log_path + "resnetvae_weight.pth")

    wandb.finish()



def init_weights_he(m):
    if type(m) is nn.Conv3d or type(m) is nn.ConvTranspose3d:
        nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu") # leaky_relu or relu
    return

def init_weights_he_relu(m):
    if type(m) is nn.Conv3d or type(m) is nn.ConvTranspose3d:
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu") # leaky_relu or relu
    return


def write_fig(path, trainE, valE, trainD, valD):
    with open(path, "w") as f:
        for t,v,td,vd in zip(trainE, valE, trainD, valD):
            f.write("trainE=%s\n" % str(t))
            f.write("valE===%s\n" % str(v))
            f.write("trainD=%s\n" % str(td))
            f.write("valD===%s\n" % str(vd))
    return

def write_kl_losses(path, kls_real, kls_fake, kls_rec, rec_errs):
    with open(path, "w") as f:
        for t,v,td,vd in zip(kls_real, kls_fake, kls_rec, rec_errs):
            f.write("kls_real==%s\n" % str(t))
            f.write("kls_fake==%s\n" % str(v))
            f.write("kls_rec===%s\n" % str(td))
            f.write("rec_errs==%s\n" % str(vd))
    return


def write_kl_losses_onlyvae(path, mse_losses, kl_losses):
    with open(path, "w") as f:
        for t,v in zip(mse_losses, kl_losses):
            f.write("mse_loss==%s\n" % str(t))
            f.write("kl_loss===%s\n" % str(v))
    return


def write_cos_losses(path, t_cos_loss, v_cos_loss):
    with open(path, "w") as f:
        for t,v in zip(t_cos_loss, v_cos_loss):
            f.write("train_cos_loss=%s    " % str(t))
            f.write("val__cos_loss=%s\n"    % str(v))
    return


def write_cos_loss(path, t_cos_loss, v_cos_loss):
    with open(path, "w") as f:
        counter = 0
        for t,v in zip(t_cos_loss, v_cos_loss):
            f.write(f"epoch{counter}: train_cos_loss={t}  val_cos_loss={v}\n")
            counter += 1
    return

def write_all_losses(
    path, 
    t_mse_losses, t_se_losses, t_d2_losses, t_kl_losses, t_cossim_losses, t_acc_losses,train_losses,
    v_mse_losses, v_se_losses, v_d2_losses, v_kl_losses, v_cossim_losses, v_acc_losses,  val_losses,
    v_f1_losses, v_macro_f1_losses,
):
    with open(path, "w") as f:
        counter = 0
        for t1,t2,t3,t4,t5,t6,t7,v1,v2,v3,v4,v5,v6,v7,v8,v9 in zip(
            t_mse_losses, t_se_losses, t_d2_losses, t_kl_losses, t_cossim_losses, t_acc_losses, train_losses,
            v_mse_losses, v_se_losses, v_d2_losses, v_kl_losses, v_cossim_losses, v_acc_losses,   val_losses,
            v_f1_losses, v_macro_f1_losses,
        ):
            f.write(f"epo{counter}:Train[mse:{t1:.5f} se:{t2:.1f} d2:{t3:.1f} kl:{t4:.1f} cossim:{t5:.2f} pred_acc:{t6:0.3f} sum_loss:{t7:0.3f}]\n")
            f.write(f"   {counter}:  Val[mse:{v1:.5f} se:{v2:.1f} d2:{v3:.1f} kl:{v4:.1f} cossim:{v5:.2f}  val_acc:{v6:0.3f} sum_loss:{v7:0.3f}]\n")
            f.write(f"   {counter}:  Val[f1:{v8:.7f} macro_f1:{v9:.7f}]\n\n")
            counter += 1
    return

def write_figres(path, train, val):
    with open(path, "w") as f:
        for t,v in zip(train, val):
            f.write("train=%s\n" % str(t))
            f.write("val===%s\n" % str(v))
    return


if __name__ == "__main__":
    main()
