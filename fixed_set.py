import torcheeg.transforms as torcheeg_transforms
from torcheeg.datasets import CSVFolderDataset
from torcheeg.models import EEGNet
from torch.utils.data import Subset, DataLoader
import numpy as np
from torcheeg.model_selection import KFoldCrossSubject, LeaveOneSubjectOut, KFold
from torchmetrics.classification import Precision, Recall, Accuracy, F1Score, AUROC
import mlflow
from torchinfo import summary

from make_data.seed import seed_everything
from make_data.utils import _read_fn
from make_data.eeg_dataset import split_train_test_from_csvdataset
from make_data.torcheeg_transforms import MappingPatient
from Domain_stuff.DataUtils import InfiniteDataLoader
from Domain_stuff.Cross_Subject_Dataset import CrossSubjectDataset
from Domain_stuff.random_subject import random_subdict
from Domain_stuff.MLDG import *
from models.Simple_STSincNet import Simple_STSincNet, classifier
from models.STSincNet import STSincNet
from models.DeepConvNet import DeepConvNet
from models.ShallowConvNet import ShallowConvNet
from models.ProtoNet import ProtoNet, SharedProNet, euclidean_distance
from train_vali_test.test import supervised_test, center_loss_test
from train_vali_test.train import supervised_train, center_loss_train
from train_vali_test.EarlyStopping import EarlyStopping

from models.OxcarNet import OxcarNet

# channel names
OXCAR_CHN_NAME = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'F7', 'F8', 
                  'T3', 'T4', 'T5', 'T6', 'O1', 'O2', 'Fz', 'Cz', 'Pz', 'A1', 'A2']
# Grid pos
OXCAR_POS_DICT = {
    'Fp1': [0, 3], 'Fp2': [0, 5], 'F7': [1, 1], 'F3': [2, 2], 'Fz': [2, 4],
    'F4': [2, 6], 'F8': [1, 7], 'T3': [4, 0], 'C3': [4, 2], 'Cz': [4, 4],
    'C4': [4, 6], 'T4': [4, 8], 'T5': [7, 1], 'P3': [6, 2], 'Pz': [6, 4], 
    'P4': [6, 6], 'T6': [7, 7], 'O1': [8, 3], 'O2': [8, 5]
}

if __name__ == "__main__":
    # set seed
    seed_everything(666)

    # use MLflow to manage
    mlflow.set_tracking_uri("your url")
    tracking_uri = mlflow.get_tracking_uri()
    # print(f'current uri:{tracking_uri}')
    experiment_name = "your experiment name"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)

    # read dataset
    online_transform = torcheeg_transforms.Compose([
        torcheeg_transforms.ToTensor(),
        torcheeg_transforms.MeanStdNormalize(axis=0),
        torcheeg_transforms.To2d()
    ])
    offline_transform = torcheeg_transforms.Compose([
        torcheeg_transforms.PickElectrode(
            torcheeg_transforms.PickElectrode.to_index_list(['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'F7', 'F8', 'T3', 
                                                    'T4', 'T5', 'T6', 'O1', 'O2', 'Fz', 'Cz', 'Pz'], OXCAR_CHN_NAME)
            ),
        torcheeg_transforms.MeanStdNormalize(axis=0),
        # torcheeg_transforms.ToInterpolatedGrid(OXCAR_POS_DICT),
    ])
    label_transofrm = torcheeg_transforms.Compose([
        torcheeg_transforms.Select(key=['subject_id', 'label']),
        MappingPatient(PATIENT_MAP)
        
    ])
    dataset = CSVFolderDataset(csv_path='your path',
    # dataset = CrossSubjectDataset(csv_path='/home/ZRK/sda_data/zrkdata/oxcar_pred_dataset/remote_meta_info.csv',
                           read_fn = _read_fn,
                           io_path = 'your path',
                           online_transform=online_transform,
                           offline_transform=offline_transform,
                           label_transform=label_transofrm,
                           num_worker=4,
                           domain_dict=PATIENT_MAP)
    
    train_set, test_set = split_train_test_from_csvdataset(dataset, TRAIN_SUBJECTS, TEST_SUBJECTS)
    print(len(train_set))
    print(len(test_set))
    # k_fold = LeaveOneSubjectOut(split_path='./partioninfo/LeaveOneSubjectOut_nowdelete')
    # for i, (train_set, test_set) in enumerate(k_fold.split(dataset)):
    device = 'cuda:1'
    # model = EEGNet(chunk_size=256, num_electrodes=19, num_classes=2).to(device)
    # model = ShallowConvNet(n_channels=19, n_classes=2).to(device)
    
    # model = DeepConvNet(n_channels=19, n_classes=2).to(device)
    
    # feature_extractor = STSincNet(num_classes=2, num_electrodes=19, dilated_factor=64, chunk_size=256).to(device)
    for dialated_factor in [16, 32, 64, 128]:
        for sinc_length_factor in [16, 8, 4, 2, 1]:
            # dialated_factor = 32
            # sinc_length_factor = 16
            # print(dialated_factor)
            # print(sinc_length_factor)
            model = nn.Sequential(
                # STSincNet(num_classes=2, num_electrodes=19, dilated_factor=dialated_factor, sinc_length_factor=sinc_length_factor, chunk_size=256, dropout=0.25),
                OxcarNet(num_classes=2, num_electrodes=19, dilated_factor=dialated_factor, sinc_length_factor=sinc_length_factor, chunk_size=256, dropout=0.25),
                classifier(dialated_factor, 2)
            ).to(device)
            model_name = model.__class__.__name__
            num_epochs = 50
            learning_rate = 0.001
            weight_decay = 0.01
            # alpha = 0.1

            criterion = torch.nn.CrossEntropyLoss()
            test_criterion = torch.nn.CrossEntropyLoss()
            # center_loss_criterion = CenterLoss(28, 512, device)
            # center_loss_optimizer = torch.optim.SGD(center_loss_criterion.parameters(), lr=0.5)
            # params = list(model.parameters())+list(center_loss_criterion.parameters())
            params = model.parameters()
            optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
            # optimizer = torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9, nesterov=True)
            # optimizer = torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)

            train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

            # test_names = list(set(list(test_set.info['subject_id'])))
            test_names = test_set[0][1][0]

            train_accuracy = Accuracy(task= 'binary').to(device)
            test_accuracy = Accuracy(task= 'binary').to(device)
            test_auroc = AUROC(task='binary').to(device)
            test_recall = Recall(task='binary').to(device)
            test_precision = Precision(task='binary').to(device)
            test_f1 = F1Score(task='binary').to(device)

            # use earlystopping strategy
            checkpoint_path = 'your path'
            # checkpoint_path = '/home/ZRK/sda_data/zrkdata/oxcar_pred_model/'+model_name+'/'+model_name+'_fixed_test'+'.pth'
            early_stopping = EarlyStopping(checkpoint_path, patience=20, verbose=True)
            
            with mlflow.start_run(experiment_id=experiment_id):
            #     # mlflow.log_param("initilization", "xavier")
                mlflow.log_param("model", model_name)
                # mlflow.log_param("test_names", test_names)
                mlflow.log_param("label", test_set[0][1][1])
                mlflow.log_param("train_patients", TRAIN_SUBJECTS)
                mlflow.log_param("test_patients", TEST_SUBJECTS)
                mlflow.log_param("dialated_factor", dialated_factor)
                mlflow.log_param("sinc_length_factor", sinc_length_factor)
                for epoch in range(num_epochs):
                        # train
                        train_loss, total_train_predicted, total_train_labels = supervised_train(model, train_loader, device, criterion, optimizer)
                        # train_loss, total_train_predicted, total_train_labels = center_loss_train(model, train_loader, device, criterion, optimizer, center_loss_criterion, alpha)
                        
                        print(f'train_loss:{train_loss}')

                        # compute train metrics
                        train_accuracy_value = train_accuracy(total_train_predicted, total_train_labels)

                        # test
                        test_loss, total_test_predicted, total_test_labels = supervised_test(model, test_loader, device, test_criterion)
                        # test_loss, total_test_predicted, total_test_labels = center_loss_test(model, test_loader, device, test_criterion, center_loss_criterion, alpha)

                        print(f'test_loss:{test_loss}')
                        # compute test metrics
                        test_accuracy_value = test_accuracy(total_test_predicted, total_test_labels)
                        test_auroc_value = test_auroc(total_test_predicted, total_test_labels)
                        test_recall_value = test_recall(total_test_predicted, total_test_labels)
                        test_precision_value = test_precision(total_test_predicted, total_test_labels)
                        test_f1_value = test_f1(total_test_predicted, total_test_labels)
                        
                        # if test_accuracy_value>0.8:
                        #     break

                        # record metrics
                        mlflow.log_metrics({"train_loss": train_loss,
                                            "test_loss": test_loss,
                                            "train_accuracy": train_accuracy_value*100,
                                            "test_accuracy": test_accuracy_value*100,
                                            "test_auroc": test_auroc_value*100,
                                            "test_recall": test_recall_value*100,
                                            "test_precision": test_precision_value*100,
                                            "test_f1": test_f1_value*100}, step=epoch)

                        early_stopping(val_loss=test_loss, model=model)
                            
                        # need recover
                        if early_stopping.early_stop:
                            print('Early Stopping!!!')
                            break
                        # need delete
                        # early_stopping.save_checkpoint(val_loss=test_loss,model=model)
                        # torch.save(model.state_dict(), checkpoint_path)

                        train_accuracy.reset()
                        test_accuracy.reset()
                        test_auroc.reset()
                        test_recall.reset()
                        test_precision.reset()
                        test_f1.reset()

                        print('yes')

                
                # Use Saved model to test
                if model_name == 'Sequential':
                    # feature_extractor = STSincNet(num_classes=2, num_electrodes=19, dilated_factor=64, chunk_size=256).to(device)
                    saved_model = nn.Sequential(
                        # STSincNet(num_classes=2, num_electrodes=19, dilated_factor=dialated_factor, sinc_length_factor=sinc_length_factor, chunk_size=256, dropout=0.25),
                        OxcarNet(num_classes=2, num_electrodes=19, dilated_factor=dialated_factor, sinc_length_factor=sinc_length_factor, chunk_size=256, dropout=0.25),
                        classifier(dialated_factor, 2)
                    ).to(device)
                elif model_name == 'EEGNet':
                    saved_model = EEGNet(chunk_size=256, num_electrodes=19, num_classes=2).to(device)
                else:
                    network = globals()[model_name]
                    saved_model = network(n_channels=19, n_classes=2).to(device)
                saved_model.load_state_dict(torch.load(checkpoint_path))
                test_loss, total_test_predicted, total_test_labels = supervised_test(saved_model, test_loader, device, test_criterion)
                
                # compute test metrics
                test_accuracy_value = test_accuracy(total_test_predicted, total_test_labels)
                test_auroc_value = test_auroc(total_test_predicted, total_test_labels)
                test_recall_value = test_recall(total_test_predicted, total_test_labels)
                test_precision_value = test_precision(total_test_predicted, total_test_labels)
                test_f1_value = test_f1(total_test_predicted, total_test_labels)

                print('=====================result==================================')
                print(f'accuracy: {test_accuracy_value*100:.2f}%')
                print(f'auroc: {test_auroc_value*100:.2f}%')
                print(f'recall: {test_recall_value*100:.2f}%')
                print(f'precision: {test_precision_value*100:.2f}%')
                print(f'f1score: {test_f1_value*100:.2f}%')
                # record metrics
                mlflow.log_metrics({
                    "final_accuracy": test_accuracy_value*100,
                    "final_auroc": test_auroc_value*100,
                    "final_recall": test_recall_value*100,
                    "final_precision": test_precision_value*100,
                    "final_f1score": test_f1_value*100
                })

        #     break
        # break