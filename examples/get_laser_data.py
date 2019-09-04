import os
import numpy as np

if __name__ == "__main__":
    # Get laser data
    url1 = "http://web.cecs.pdx.edu/~mcnames/DataSets/SantaFeA.dat"
    os.system(f"""wget -c --read-timeout=5 --tries=0 ""{url1}""""")
    url2 = "http://web.cecs.pdx.edu/~mcnames/DataSets/SantaFeA2.dat"
    os.system(f"""wget -c --read-timeout=5 --tries=0 ""{url2}""""")

    # Load data
    laser = np.concatenate((np.loadtxt('SantaFeA.dat'), np.loadtxt('SantaFeA2.dat')))

   

    # Make directory
    if not os.path.isdir('./laser'):
        os.mkdir('./laser')

    # Generate indexes
    train_idx = [0, int(np.ceil(laser.shape[0]*0.55))]
    val_idx = [train_idx[1], int(np.ceil(laser.shape[0]*0.75))]
    test_idx = [val_idx[1], laser.shape[0]-1]
    print(train_idx)
    print(val_idx)
    print(test_idx)

    # Separate data
    train_data = laser[train_idx[0]:train_idx[1]+2]
    val_data = laser[val_idx[0]:val_idx[1]+2]
    test_data = laser[test_idx[0]:test_idx[1]+2]


 	# Rescale between -1 and 1
    # laser = np.interp(laser, (np.min(laser), np.max(laser)), (-1, 1))
    rescaleMax = np.max(np.concatenate((train_data, val_data)))
    rescaleMin = np.min(np.concatenate((train_data, val_data)))

    train_data = np.interp(train_data, (rescaleMin, rescaleMax), (-1,1))
    val_data = np.interp(val_data, (rescaleMin, rescaleMax), (-1,1))
    test_data = np.interp(test_data, (rescaleMin, rescaleMax), (-1,1))



    # Save data
    np.savetxt('./laser/TrainData.csv', train_data[0:-1], header='val0', comments='')
    np.savetxt('./laser/TrainLabel.csv', train_data[1::], header='label0', comments='')
    np.savetxt('./laser/ValData.csv', val_data[0:-1], header='val0', comments='')
    np.savetxt('./laser/ValLabel.csv', val_data[1::], header='label0', comments='')
    np.savetxt('./laser/TestData.csv', test_data[0:-1], header='val0', comments='')
    np.savetxt('./laser/TestLabel.csv', test_data[1::], header='label0', comments='')
    np.savetxt('./laser/FullData.csv', laser, header='val0', comments='')
    