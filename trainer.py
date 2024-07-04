import random
import time
from utils import *
from evaluation import evaluate_ranking
from model import Model


def training(dataLoader, config, device):
    if config.isTrain:
        numUsers = dataLoader.numTrain
        numItems = dataLoader.numItemsTrain
    else:
        numUsers = dataLoader.numTrainVal
        numItems = dataLoader.numItemsTest

    if numUsers % config.batch_size == 0:
        numBatch = numUsers // config.batch_size
    else:
        numBatch = numUsers // config.batch_size + 1
    idxList = [i for i in range(numUsers)]

    model = Model(config, numItems).to(device)

    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2)
    elif config.opt == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr, weight_decay=config.l2)

    for epoch in range(config.epochs):
        random.seed(1234)
        random.shuffle(idxList)
        timeEpStr = time.time()
        epochLoss = 0

        for batch in range(numBatch):
            start = config.batch_size * batch
            end = min(numUsers, start + config.batch_size)

            batchList = idxList[start:end]

            samples, userhis, target = generateBatchSamples(dataLoader, batchList, config, isEval=0)

            samples = torch.from_numpy(samples).type(torch.LongTensor).to(device)
            userhis = torch.from_numpy(userhis).type(torch.FloatTensor).to(device)
            target = torch.from_numpy(target).type(torch.FloatTensor).to(device)

            scores = model.forward(samples, userhis, device)
            loss = -(torch.log(scores) * target + torch.log(1 - scores) * (1 - target)).sum(-1).mean()  # sum
            epochLoss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epochLoss = epochLoss / float(numBatch)
        timeEpEnd = time.time()

        if epoch % config.evalEpoch == 0:
            timeEvalStar = time.time()
            print("start evaluation")

            recall, ndcg = evaluate_ranking(model, dataLoader, config, device, config.isTrain)

            timeEvalEnd = time.time()

            output_str = "Epoch %d \t recall@10=%.6f, recall@20=%.6f," \
                         "ndcg@10=%.6f, ndcg@20=%.6f, [%.1f s]" % (
                             epoch + 1, recall[0], recall[1], ndcg[0], ndcg[1],
                             timeEvalEnd - timeEvalStar)

            print("time: %.1f, loss: %.3f" % (timeEpEnd - timeEpStr, epochLoss))
            print(output_str)

        else:
            print("time: %.1f, loss: %.3f" % (timeEpEnd - timeEpStr, epochLoss))
