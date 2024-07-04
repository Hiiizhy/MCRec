import numpy as np
import torch
import torch.nn.functional as F

activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': torch.tanh, 'sigm': torch.sigmoid}


def generateBatchSamples(dataLoader, batchIdx, config, isEval):
    samples, sampleLen, userhis, target = dataLoader.batchLoader(batchIdx, config.isTrain, isEval)

    imaxLenSeq = max([len(userLen) for userLen in sampleLen])
    if imaxLenSeq > config.maxLenSeq:
        maxLenSeq = imaxLenSeq
    else:
        maxLenSeq = config.maxLenSeq

    imaxLenBas = max([max(userLen) for userLen in sampleLen])
    if imaxLenBas > config.maxLenBas:
        maxLenBas = imaxLenBas
    else:
        maxLenBas = config.maxLenBas

    paddedSamples = []
    targetList = []
    for user in samples:
        trainU = user[:-1]
        testU = user[-1]
        targetList.append(testU)

        paddedU = []
        for eachBas in trainU:
            paddedBas = [config.padIdx] * (maxLenBas - len(eachBas)) + eachBas
            paddedU.append(paddedBas)
        paddedU = [[config.padIdx] * maxLenBas] * (maxLenSeq - len(paddedU)) + paddedU
        paddedSamples.append(paddedU)

    return np.asarray(paddedSamples), userhis, target
