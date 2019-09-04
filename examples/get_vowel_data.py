import numpy as np
import csv
import os

if __name__ == "__main__":
    # Download data
    url1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/JapaneseVowels-mld/ae.train"
    os.system(f"""wget -c --read-timeout=5 --tries=0 ""{url1}""""")
    url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/JapaneseVowels-mld/ae.test"
    os.system(f"""wget -c --read-timeout=5 --tries=0 ""{url2}""""")
    url3 = "https://archive.ics.uci.edu/ml/machine-learning-databases/JapaneseVowels-mld/size_ae.train"
    os.system(f"""wget -c --read-timeout=5 --tries=0 ""{url3}""""")
    url4 = "https://archive.ics.uci.edu/ml/machine-learning-databases/JapaneseVowels-mld/size_ae.test"
    os.system(f"""wget -c --read-timeout=5 --tries=0 ""{url4}""""")

    # Make directory
    if not os.path.isdir('vowel'):
        os.mkdir('vowel')

    # Load train size
    size_train = np.loadtxt('size_ae.train')

    # Load train file
    subjArray = []
    with open('ae.train', newline='') as f:
        reader = csv.reader(f)
        data = []
        blockArray = []
        i = 0
        for row in reader:
            if row == []:
                arr = np.array(data)
                blockArray.append(arr)
                data = []
                if(len(blockArray) == size_train[i]):
                    subjArray.append(np.array(blockArray))
                    blockArray = []
                    i = i + 1
            else:
                data.append(row[0].split())
    f.close()

    # Set train/val size
    trainBlocks = int(np.floor(size_train[0]*0.55))
    valBlocks = int(size_train[0] - trainBlocks)

    # Find rescaling factors
    maxVal = -10
    minVal = +10
    for subj in subjArray:
        for i in range(trainBlocks):
            maxValTemp = np.max(subj[i].astype(np.float))
            if(maxValTemp > maxVal):
                maxVal = maxValTemp
            minValTemp = np.min(subj[i].astype(np.float))
            if(minValTemp < minVal):
                minVal = minValTemp

    # Save training data 
    with open('./vowel/TrainData.csv', 'w') as f:
        fields = ['val{}'.format(x) for x in range(12)]
        writer = csv.writer(f, dialect='excel', lineterminator='\n')
        writer.writerow(fields)
        for subj in subjArray:
            for i in range(trainBlocks):
                datatemp = np.interp(subj[i].astype(np.float64), (minVal, maxVal), (0, 1))
                writer.writerows(datatemp)
        f.close()

    # Save training label
    label = np.reshape([[x]*trainBlocks for x in range(size_train.shape[0])], (trainBlocks*size_train.shape[0],1))
    np.savetxt('./vowel/TrainLabel.csv', label, fmt='%d', header='label0', comments='')

    # Save training sampling
    sampling = []
    for subj in subjArray:
        for i in range(trainBlocks):
            sampling.append([len(subj[i]), 0])
    np.savetxt('./vowel/TrainSampling.csv', sampling, fmt='%d', delimiter=',', header='sample0,sample1', comments='')
    
    # Save validation data
    with open('./vowel/ValData.csv', 'w') as f:
        fields = ['val{}'.format(x) for x in range(12)]
        writer = csv.writer(f, dialect='excel', lineterminator='\n')
        writer.writerow(fields)
        for subj in subjArray:
            for i in range(valBlocks):
                datatemp = np.interp(subj[i+trainBlocks].astype(np.float64), (minVal, maxVal), (0, 1))
                writer.writerows(datatemp)
        f.close()
    
    # Save validation label
    label = np.reshape([[x]*valBlocks for x in range(size_train.shape[0])], (valBlocks*size_train.shape[0],1))
    np.savetxt('./vowel/ValLabel.csv', label, fmt='%d', header='label0', comments='')

    # Save validation sampling
    sampling = []
    for subj in subjArray:
        for i in range(valBlocks):
            sampling.append([len(subj[i+trainBlocks]), 0])
    np.savetxt('./vowel/ValSampling.csv', sampling, fmt='%d', delimiter=',', header='sample0,sample1', comments='')

    # Load test size
    size_test = np.loadtxt('size_ae.test', dtype=np.uint8)

    # Load test file
    subjArray = []
    with open('ae.test', newline='') as f:
        reader = csv.reader(f)
        data = []
        blockArray = []
        i = 0
        for row in reader:
            if row == []:
                arr = np.array(data)
                blockArray.append(arr)
                data = []
                if(len(blockArray) == size_test[i]):
                    subjArray.append(np.array(blockArray))
                    blockArray = []
                    i = i + 1
            else:
                data.append(row[0].split())
    f.close()

    # Save test data 
    with open('./vowel/TestData.csv', 'w') as f:
        fields = ['val{}'.format(x) for x in range(12)]
        writer = csv.writer(f, dialect='excel', lineterminator='\n')
        writer.writerow(fields)
        for subj in subjArray:
            for block in subj:
                datatemp = np.interp(block.astype(np.float64), (minVal, maxVal), (0, 1))
                writer.writerows(datatemp)
        f.close()

    # Save test label
    label = np.array([val for sublist in [[x]*size_test[x] for x in range(size_test.shape[0])] for val in sublist]).transpose()
    np.savetxt('./vowel/TestLabel.csv', label, fmt='%d', header='label0', comments='')

    # Save validation sampling
    sampling = []
    for subj in subjArray:
        for block in subj:
            sampling.append([len(block), 0])
    np.savetxt('./vowel/TestSampling.csv', sampling, fmt='%d', delimiter=',', header='sample0,sample1', comments='')
