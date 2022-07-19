import numpy as np


class MonteCarlo:
    def __init__(self, probability):
        self.simulation = []
        self.simulatedOutcome = []
        self.probL = probability
        self.simulationNumber = 1000000



    def simulate(self):
        for i in range(self.simulationNumber):
            probSet = np.random.rand(9)
            self.simulation.append(probSet)
        return self.simulation

    def combinedProbability(self):
        length = len(self.probL)
        for i in self.simulation:
            count = 0
            for j in range(length):
                if i[j] < self.probL[j]:
                    count = count + 1
            self.simulatedOutcome.append(count)

    def findMajorProb(self):
        num = len([i for i in self.simulatedOutcome if i >= 5])
        return "{:2}".format(num / self.simulationNumber * 100)



