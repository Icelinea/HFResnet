import torch
import torch.nn as nn
import torch.optim as optim

from utils import basicInit, dataInit, calculate_metrics
# from time import sleep


def test(device):
    # init
    args, model = basicInit()
    _, _, test_dataloader = dataInit(args)
    
    model_name = args.checkpoint_path + 'model_.pth'    # assign name by hand
    model.load_state_dict(torch.load(model_name))
    model.to(device)

    # test
    model.eval()
    emotions_id = list(args.emotions2id.values())

    test_preds = []
    test_labels = []

    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        test_preds.extend(predicted.tolist())
        test_labels.extend(labels.tolist())

    precision, recall, f1 = calculate_metrics(test_preds, test_labels, emotions_id)

    # print
    print('\n\n----- test dataset using model {} ------'.format(model_name))
    for i in range(args.num_classes):
        print('Test - Emotion {} Precision: {:.3f} Recall: {:.3f} F1: {:3f}' \
                .format(args.id2emotions[i], precision[i], recall[i], f1[i]))


def train(device):
    # init
    args, model = basicInit()
    model.to(device)
    train_dataloader, eval_dataloader, _ = dataInit(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # basic info
    emotions_id = list(args.emotions2id.values())

    train_precision = []
    train_recall = []
    train_f1 = []

    eval_precision = []
    eval_recall = []
    eval_f1 = []
    eval_overall_loss = 114514

    # train and eval print
    print('\n\n----- training start -----')
    for epoch in range(args.num_epoches):
        # train
        model.train()
        train_preds = []
        train_labels = []

        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.tolist())
            train_labels.extend(labels.tolist())

        precision, recall, f1 = calculate_metrics(train_preds, train_labels, emotions_id)
        train_precision.append(precision)
        train_recall.append(recall)
        train_f1.append(f1)

        # eval
        model.eval()
        eval_preds = []
        eval_labels = []
        eval_loss = 0

        for images, labels in eval_dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            eval_preds.extend(predicted.tolist())
            eval_labels.extend(labels.tolist())

        eval_loss /= len(eval_dataloader.dataset)

        precision, recall, f1 = calculate_metrics(eval_preds, eval_labels, emotions_id)
        eval_precision.append(precision)
        eval_recall.append(recall)
        eval_f1.append(f1)

        # print
        print('\n\n----- [{}] / [{}] ------'.format(epoch + 1, args.num_epoches))
        for i in range(args.num_classes):
            print('Train - Emotion {} Precision: {:.3f} Recall: {:.3f} F1: {:3f}' \
                  .format(args.id2emotions[i], train_precision[epoch][i], train_recall[epoch][i], train_f1[epoch][i]))
            print('Eval - Emotion {} Precision: {:.3f} Recall: {:.3f} F1: {:3f}' \
                  .format(args.id2emotions[i], eval_precision[epoch][i], eval_recall[epoch][i], eval_f1[epoch][i]))
            
        # checkpoint
        if eval_overall_loss > eval_loss:
            eval_overall_loss = eval_loss
            save_path = args.checkpoint_path + 'model_{:3f}.pth'.format(eval_overall_loss)
            torch.save(model.state_dict(), save_path)

    print('\n\n----- training finished -----')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(device)
    test(device)