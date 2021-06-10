import h5py
import matplotlib.pyplot as plt
import numpy as np
import math

#for use with creating images from scratch, returns maxes[x,y,z] and mins[x,y,z]
def get_max_min(h5File):
    maxes = []
    mins = []
    for k in range(len(h5File.keys())):
        evt_id = list(h5File.keys())[k]
        event = h5File[evt_id]
        evt_data = np.zeros((len(event),3))
        i=0
        for e in event:
            a = list(e)
            evt_data[i] = a[:3]
            i+=1
        if len(evt_data) != 0:
            maxes.append(list(np.amax(evt_data, axis=0)))
            mins.append(list(np.amin(evt_data, axis=0)))
    maxes = np.asarray(maxes)
    mins = np.asarray(mins)
    final_max = np.amax(maxes, axis =0)
    final_min = np.amin(mins, axis =0)
    return final_min, final_max

#concatenates 2 datasets into 1 dataset for 3 class classification
def combineDatasets(h5File, h5File2):
    fileName = input("Enter file name (.h5) for new dataset: ")
    hf = h5py.File(fileName, 'w')
    images = []
    targets = []
    data1 = h5File['imgs'][:]
    targets1 = h5File['targets'][:]
    data2 = h5File2['imgs'][:]
    targets2 = h5File2['targets'][:]
    for k in range(len(data1)):
        images.append(data1[k])
        targets.append(targets1[k])
    for i in range(len(data2)):
        images.append(data2[i])
        if(targets2[i]==1):
            targets.append(2)
        else:
            targets.append(targets2[i])
    hf.create_dataset('imgs', data = images)
    hf.create_dataset('targets', data = targets)

#stacks plots, takes in the h5Files of images to be concatenated
def makeStackedDataset(h5File,h5File2):
    #plt.figure(figsize=(10,10))
    fileName = input("Enter file name (.h5) for new dataset: ")
    hf = h5py.File(fileName, 'w')
    images = []
    data1 = h5File['imgs'][:]
    data2 = h5File2['imgs'][:]
    if len(data1) != len(data2):
        print("Incompatible data file lengths, cannot create merged dataset.")
    else:
        targets = h5File['targets'][:]
        for k in range(len(data1)):
            event = np.concatenate((data1[k], data2[k]))
            images.append(event)
        hf.create_dataset('imgs', data = images)
        hf.create_dataset('targets', data = targets)

#data vis matrix code using matplotlib to plot
def plotData(h5File):
    plt.figure(figsize=(10,10))
    for k in range(len(h5File.keys())):
        evt_id = list(h5File.keys())[k]
        event = h5File[evt_id]
        evt_data = np.zeros((len(event),3))
        #print(evt_data)
        num_id = int(evt_id[7:len(evt_id)-1])
        #print(num_id)
        i = 0
        for e in event:
            #print(list(e))
            a = list(e)
            evt_data[i] = a[:3]
            #print (evt_data[i])
            i += 1
        plt.subplot(5, 5, k + 1)
        plt.scatter(evt_data[:,1],evt_data[:,2], c=None, s=11, marker='.')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.xlim(-300,300)
        plt.ylim(450,1050)
        if num_id%2==0:
            l = "Beam"
        else:
            l = "Decay"
        plt.xlabel(l+": Event "+str(num_id))
    plt.show()

#DATA VIS MATRIX CODE, DOESN'T SAVE ANYTHING
def imageMatrix(h5File,mins,maxes,xy):
    plt.figure(figsize=(10,10))
    scalar = [0,0,0]
    for i in range(3):
        if abs(mins[i]) > abs(maxes[i]):
            scalar[i] = abs(mins[i])
        else:
            scalar[i] = abs(maxes[i])
    for k in range(len(h5file.keys())):
        evt_id = list(h5File.keys())[k]
        raw_event = h5File[evt_id]
        event = np.full((128,128),255)
        num_id = int(evt_id[7:len(evt_id)-1])
        raw_event_list = list(raw_event)
        #print(evt_id)
        for entry in raw_event_list:
            a = list(entry)
            x = a[0]
            y = a[1]
            z = a[2]
            #print("Original: "+str(x)+" "+str(y))
            x = round(((x/scalar[0])*64))+63
            #print(x)
            y = round(((y/scalar[1])*64))+63
            #z-=550
            z = round(((z/scalar[2])*64))+63
            if xy:
                event[y][x]-=1
            else:
                event[z][y] -=1
        if xy:
            event = np.flipud(event)
        else:
            event = np.flipud(event)
        plt.subplot(5, 5, k + 1)
        plt.imshow(event, cmap = "bone")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if num_id%2==0:
            l = "Beam"
        else:
            l = "Decay"
        plt.xlabel(l+": Event "+str(num_id))
        plt.savefig("greyscale_matrix.png")

#saves a matrix of images (requires already processed image matrix)
def saveMatrix(h5File):
    plt.figure(figsize=(10,10))
    for k in range(len(h5file.keys())):
        event = h5File['imgs'][k]
        plt.subplot(5, 5, k + 1)
        plt.imshow(event, cmap = "bone")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
    plt.savefig('image_matrix.png')

#makes black and white image Dataset, takes the h5File of data, a tuple of mins [x,y,z],
# a tuple of maxes [x,y,z] and a boolean (xy) that creates xy projection when true and yz
#when false
def makeImageDataset(h5File, mins, maxes, xy):
    fileName = input("Enter file name (.h5) for new dataset: ")
    hf = h5py.File(fileName, 'w')
    images = []
    targets = []
    scalar = [0.0,0.0,0.0]
    for i in range(3):
        if abs(mins[i]) > abs(maxes[i]):
            scalar[i] = abs(mins[i])
        else:
            scalar[i] = abs(maxes[i])
    for k in range(len(h5file.keys())):
        evt_id = list(h5File.keys())[k]
        raw_event = h5File[evt_id]
        event = np.full((128,128),255)
        num_id = int(evt_id[7:len(evt_id)-1])
        raw_event_list = list(raw_event)
        for entry in raw_event_list:
            a = list(entry)
            x = a[0]
            y = a[1]
            z = a[2]
            x = round(((x/scalar[0])*64))+63
            #print(x)
            y = round(((y/scalar[1])*64))+63
            #z-=550
            z = round(((z/scalar[2])*64))+63
            if xy:
                event[y][x]-=1
            else:
                event[z][y] -=1
        if xy:
            event = np.flipud(event)
        else:
            event = np.flipud(event)
        images.append(event)
        if num_id%2==0:
            targets.append(0)
            #beam
        else:
            targets.append(1)
    #decay
    hf.create_dataset('imgs', data=images)
    hf.create_dataset('targets', data=targets)
