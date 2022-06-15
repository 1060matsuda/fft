from os import times
from tokenize import cookie_re
from turtle import color
from xmlrpc.client import boolean
from numpy.core.shape_base import atleast_2d
from numpy.lib import type_check
from scipy import fftpack
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import signal

HPC_OR_LOCAL = "HPC"
# "HPC" -> no graph
# "LOCAL" -> graph

#detects the first downward zerocross point
def getDownwardZeroCrossIndex(vector1d):
    downCount = 0
    searchIndex = 1
    """
    for i in range(10):
        #downcount counter until datapoint 10
        searchIndex = searchIndex + i
        if data[searchIndex] - data[searchIndex-1] < 0:
            downCount=downCount + 1
        else:
            downCount = 0
    """
    while True:
        searchIndex = searchIndex + 1
        if vector1d[searchIndex]<0 and vector1d[searchIndex-1]>0:
            for i in range(10):
                if vector1d[searchIndex-5+i] < vector1d[searchIndex-5+i+1]:
                    downCount = 0
                else:
                    downCount = downCount+1
            if downCount == 10:
                #print(searchIndex)
                return searchIndex

#detects the first upward zerocross point
def getUpwardZeroCrossIndex(vector1d):
    upCount = 0
    searchIndex = 1
    """
    for i in range(10):
        #downcount counter until datapoint 10
        searchIndex = searchIndex + i
        if data[searchIndex] - data[searchIndex-1] < 0:
            upCount=upCount + 1
        else:
            upCount = 0
    """
    while True:
        searchIndex = searchIndex + 1
        if vector1d[searchIndex]>0 and vector1d[searchIndex-1]<0:
            for i in range(10):
                if vector1d[searchIndex-5+i] > vector1d[searchIndex-5+i+1]:
                    upCount = 0
                else:
                    upCount = upCount+1
            if upCount == 10:
                #print(searchIndex)
                return searchIndex

# Reads the csv file where time series data of detector coorinate is saved.
def loadCsvOutput(csvData):
    ndarrayData = np.loadtxt(csvData, delimiter=",")

    # Several timesteps are recorded twice in the raw csv file. 
    # Below is the script which deletes the double-recorded timesteps.
    timeAtTheRowAbove = -100
    delRows = []
    for i in range(len(ndarrayData)):
        if timeAtTheRowAbove == ndarrayData[i,0]:
            delRows.append(i)
        timeAtTheRowAbove = ndarrayData[i,0]
    ndarrayData = np.delete(ndarrayData,delRows,0)

    return ndarrayData

# Withdraws the time series of the detector's x coorinate
def getDetector(array):
    return array[:,2]

# Withdraws the time series of the source's x coorinate
def getSource(array):
    return array[:,1]

# trims the wave within the specified range
def trim(vector1d, trimStartTimestep, trimRange):
    trimmedArray = vector1d[trimStartTimestep:trimStartTimestep+trimRange]
    return trimmedArray

# trims the wave within the specified range and then offsets the wave so that the normal position can be x=0. 
def trimAndOffset(vector1d, trimStartTimestep, trimRange):
    trimmedArray = vector1d[trimStartTimestep:trimStartTimestep+trimRange]
    offsettedArray = trimmedArray - vector1d[0]
    return offsettedArray

# performs fft. output[0] = power, output[1] = freq. both output are recognized as complex.
def fftWithWindow(FFTData, hannORhamming:str):
    dataPoints = len(FFTData)
    if hannORhamming == "hann":
        windowFunction = signal.hann(dataPoints)
    elif hannORhamming == "hamming":
        windowFunction = signal.hamming(dataPoints)
    else:
        raise Exception("Window is not/wrongly specified. Either hann / hamming is right.")
    acf = 1/(sum(windowFunction)/dataPoints)
    #print("acf")
    #print(acf)
    waveToTransform = acf*windowFunction*FFTData
    if HPC_OR_LOCAL == "LOCAL":
        plt.plot(waveToTransform)
        plt.show()
    FFT_power = np.fft.fft(waveToTransform,n=None,norm=None) 
    FFT_freq = np.fft.fftfreq(dataPoints, d=timeStep*(10**-12))
    return np.stack([FFT_power, FFT_freq])

# calculates beta.
def getBeta(Amp1,Amp2,Lambda,DeltaX):
    beta = 8*Amp2*Lambda*Lambda/DeltaX/Amp1/Amp1/np.pi/np.pi/2/2
    return beta

# DO NOT USE THIS FUNCTION. currently under development. incomplete.
def zeroPadding(data):
    zeros = np.zeros(len(data))
    paddedData = np.hstack((zeros,data,zeros))
    acf = (sum(np.abs(data)) / len(data)) / (sum(np.abs(paddedData)) / len(paddedData))
    return acf*paddedData

# searches the 1d array data for the input value,
# and returns the index of the array where the nearest value of the input value is contained.
def getIndexOfNearestValue(data, value):
    index = np.argmin(np.abs(np.array(data) - value))
    return index

# searches the FFT-performed 1d array of frequencies, and returns the index of 1st - 6th harmonics.
def getIndexUpToSixthHarmonic(data, frequency):
    index1 = getIndexOfNearestValue(data, frequency)
    index2 = getIndexOfNearestValue(data, frequency*2)
    index3 = getIndexOfNearestValue(data, frequency*3)
    index4 = getIndexOfNearestValue(data, frequency*4)
    index5 = getIndexOfNearestValue(data, frequency*5)
    index6 = getIndexOfNearestValue(data, frequency*6)
    return np.array([index1, index2,index3,index4,index5,index6], dtype=np.int64)

# reads the output csv.
data1 = loadCsvOutput("output.csv")

# data2 is always needed.
# If you perform pulse-inversion, specify the inversed wave form output.
# If you don't, specify the same filename as data1.
data2 = loadCsvOutput("output.csv")

data= (data1+data2)/2

timeStep = data1[1,0] - data1[0,0]

# Casts each row of the superimposed data to 1-dimensional vector.
time = data[:,0]
x_detec = data[:,2]
x_source = data[:,1]
x_edge = data[:,3]
v_source = data[:,4]
f_source = data[:,5]

# Wave Period. 1000 / (Input Frequency[GHz]). in other words FREQUENCY[GHz]*T[ps] = 1000.
T = 2
inputFreq = 10**12 / T #[Hz]

# wave amplitude at the source
source_amp = 0.1 #[Å]

# Nc: How many cycles to window
Nc = 3

# Ns: Number of data points in one cycle
Ns = T/timeStep

# N: Total Number of data points in thw windowed region
N = int(Nc*Ns)

zeroCrossTimeStep = getDownwardZeroCrossIndex(data1[:,2]-data1[0,2])
#zeroCrossTimeStep = getUpwardZeroCrossIndex(data1[:,2]-data1[0,2])
arrivalTimeStep = zeroCrossTimeStep - T/timeStep/2
windowStartTimeStep = zeroCrossTimeStep

delta_x = x_detec[0] - x_source[0]

waveVelocity = delta_x*(10**(-10)) / ((arrivalTimeStep * timeStep)*(10**(-12)))

waveLength = waveVelocity * T * (10**(-12))

trimmedXD = trimAndOffset(x_detec, windowStartTimeStep, N)
trimmedTime = trim(time, windowStartTimeStep, N)

#FFT. transformedArray: [0]=power, [1]=freq
FFTedData = fftWithWindow(trimmedXD, "hann") #window = "hann" or "hamming"
#FFTedData = fftWithWindow(zeroPadding(trimmedXD), "hann")
absFFTData = np.abs(FFTedData)

#higher harmonics amplitude[arb]
harmonicsIndex = getIndexUpToSixthHarmonic(absFFTData[1],inputFreq)
A1 = absFFTData[0][harmonicsIndex[0]]
A2 = absFFTData[0][harmonicsIndex[1]]
A3 = absFFTData[0][harmonicsIndex[2]]
A4 = absFFTData[0][harmonicsIndex[3]]
A5 = absFFTData[0][harmonicsIndex[4]]
A6 = absFFTData[0][harmonicsIndex[5]]

a1s = source_amp*10**-10
a2 = 2*A2/int(len(trimmedXD))*10**-10
beta = getBeta(a1s,a2,waveLength,delta_x*(10**-10))

f = open("beta.txt", "w")
f.write(str(beta))
f.close

#drawings
if HPC_OR_LOCAL == "LOCAL":
    fig = plt.figure()
    print("x_detec1")
    plt.plot(getDetector(data1)-getDetector(data1)[0])
    plt.axvline(x=windowStartTimeStep, color="red")
    plt.axvline(x=windowStartTimeStep+N, color="red")
    plt.show()

    print("timeStep Δt is:")
    print(timeStep)

    print("superimposed wave form")
    plt.plot(x_detec-x_detec[0])
    plt.show()

    print("trimmed and offseted form of superimposed wave")
    plt.plot(trimmedTime/timeStep, trimmedXD)
    plt.show()

    print()
    print("timeStep Δt is:")
    print(timeStep)

    print("wave velocity v[m/s] is:")
    print(waveVelocity)

    print("wave length λ is:")
    print(waveLength)

    fig, ax = plt.subplots()
    ax.plot(absFFTData[1,1:int(N/2)], absFFTData[0,1:int(N/2)])
    #ax.plot(absFFTData[1,1:int(N/20)], absFFTData[0,1:int(N/20)])
    ax.set_xlabel("Freqency [Hz]")
    ax.set_ylabel("Amplitude")
    ax.grid()
    plt.show()

    print("beta:")
    print(beta)

    x_bar = np.array([1, 2, 3, 4, 5, 6])
    y_bar = np.array([A1, A2, A3, A4, A5, A6])
    #print(x_bar)
    #print(y_bar)
    plt.bar(x_bar, y_bar, width=0.2)
    plt.show()
