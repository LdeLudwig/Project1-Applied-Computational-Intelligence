# Computational Intelligence - Project 1a - Fuzzy System
# Lucas Xavier and Simon Nygren
# 2023-10-06

import pandas as pd
import simpful as sf
import random

TestMe = pd.read_csv('Proj1_TestS.csv', sep=',', decimal='.')

# Define the range for randomization for each input variable for the dataset to be generated
min_MU, max_MU = 0, 1.0
min_PL, max_PL = 0, 1.0
min_OB, max_OB = 0, 1.0
min_LA, max_LA = 0, 1.0

# Number of data points in the generated dataset
num_data_points = 1000

# Initialize Fuzzy System 1 and define linguistic variables
FS1 = sf.FuzzySystem()

S1_1= sf.FuzzySet(points=[[0, 1], [0.3, 1], [0.5, 0]], term="low")
S1_2 = sf.FuzzySet(points=[[0.3, 0], [0.5, 1], [0.6, 1], [0.8, 0]], term="medium")
S1_3 = sf.FuzzySet(points=[[0.6, 0], [0.8, 1], [1, 1]], term="high")
Latency = sf.LinguisticVariable([S1_1, S1_2, S1_3])
FS1.add_linguistic_variable("Latency", Latency)

S2_1= sf.FuzzySet(points=[[0, 1], [0.3, 1], [0.5, 0]], term="low")
S2_2 = sf.FuzzySet(points=[[0.3, 0], [0.5, 1], [0.7, 1], [0.8, 0]], term="medium")
S2_3 = sf.FuzzySet(points=[[0.65, 0], [0.8, 1], [1, 1]], term="high")
ProcessorLoad = sf.LinguisticVariable([S2_1, S2_2, S2_3])
FS1.add_linguistic_variable("ProcessorLoad", ProcessorLoad)

S3_1= sf.FuzzySet(points=[[0, 1], [0.2, 1], [0.5, 0]], term="bad")
S3_2 = sf.FuzzySet(points=[[0.2, 0], [0.4, 1], [0.6, 1], [0.8, 0]], term="okay")
S3_3 = sf.FuzzySet(points=[[0.5, 0], [0.8, 1], [1, 1]], term="good")
ProcessLocally = sf.LinguisticVariable([S3_1, S3_2, S3_3])
FS1.add_linguistic_variable("ProcessLocally", ProcessLocally)

# Construct Rule Base 1
FS1.add_rules(["IF (Latency IS low) AND (ProcessorLoad IS low) THEN (ProcessLocally IS okay)",
               "IF (Latency IS low) AND (ProcessorLoad IS medium) THEN (ProcessLocally IS bad)",
               "IF (Latency IS low) AND (ProcessorLoad IS high) THEN (ProcessLocally IS bad)",
               "IF (Latency IS medium) AND (ProcessorLoad IS low) THEN (ProcessLocally IS good)",
               "IF (Latency IS medium) AND (ProcessorLoad IS medium) THEN (ProcessLocally IS okay)",
               "IF (Latency IS medium) AND (ProcessorLoad IS high) THEN (ProcessLocally IS bad)",
               "IF (Latency IS high) AND (ProcessorLoad IS low) THEN (ProcessLocally IS good)",
               "IF (Latency IS high) AND (ProcessorLoad IS medium) THEN (ProcessLocally IS good)",
               "IF (Latency IS high) AND (ProcessorLoad IS high) THEN (ProcessLocally IS bad)"
               ])


# Initialize Fuzzy System 2 and define linguistic variables
FS2 = sf.FuzzySystem()

S4_1 = sf.FuzzySet(points=[[0, 1], [0.2, 1], [0.5, 0]], term="low")
S4_2 = sf.FuzzySet(points=[[0.2, 0], [0.3, 1], [0.6, 1], [0.8, 0]], term="medium")
S4_3 = sf.FuzzySet(points=[[0.6, 0], [0.8, 1], [1, 1]], term="high")
OutBandwidth = sf.LinguisticVariable([S4_1, S4_2, S4_3])
FS2.add_linguistic_variable("OutBandwidth", OutBandwidth)


S5_1= sf.FuzzySet(points=[[0, 1], [0.2, 1], [0.5, 0]], term="low")
S5_2 = sf.FuzzySet(points=[[0.2, 0], [0.3, 1], [0.6, 1], [0.8, 0]], term="medium")
S5_3 = sf.FuzzySet(points=[[0.6, 0], [0.8, 1], [1, 1]], term="high")
MemoryUsage = sf.LinguisticVariable([S5_1, S5_2, S5_3])
FS2.add_linguistic_variable("MemoryUsage", MemoryUsage)


S6_1= sf.FuzzySet(points=[[0, 1], [0.15, 1], [0.40, 0]], term="bad")
S6_2 = sf.FuzzySet(points=[[0.15, 0], [0.4, 1], [0.6, 1], [0.8, 0]], term="okay")
S6_3 = sf.FuzzySet(points=[[0.6, 0], [0.8, 1], [1, 1]], term="good")
InputDataCloud = sf.LinguisticVariable([S6_1, S6_2, S6_3])
FS2.add_linguistic_variable("InputDataCloud", InputDataCloud)


# Construct Rule Base 2
FS2.add_rules(["IF (OutBandwidth IS low) AND (MemoryUsage IS low) THEN (InputDataCloud IS okay)",
               "IF (OutBandwidth IS low) AND (MemoryUsage IS medium) THEN (InputDataCloud IS bad)",
               "IF (OutBandwidth IS low) AND (MemoryUsage IS high) THEN (InputDataCloud IS bad)",
               "IF (OutBandwidth IS medium) AND (MemoryUsage IS low) THEN (InputDataCloud IS good)",
               "IF (OutBandwidth IS medium) AND (MemoryUsage IS medium) THEN (InputDataCloud IS okay)",
               "IF (OutBandwidth IS medium) AND (MemoryUsage IS high) THEN (InputDataCloud IS bad)",
               "IF (OutBandwidth IS high) AND (MemoryUsage IS low) THEN (InputDataCloud IS good)",
               "IF (OutBandwidth IS high) AND (MemoryUsage IS medium) THEN (InputDataCloud IS good)",
               "IF (OutBandwidth IS high) AND (MemoryUsage IS high) THEN (InputDataCloud IS bad)"
               ])

# Initialize Fuzzy System 3 and define linguistic variables
FS3 = sf.FuzzySystem()

S7_1 = sf.FuzzySet(points=[[0, 1], [0.3, 1], [0.4, 0]], term="low")
S7_2 = sf.FuzzySet(points=[[0.2, 0], [0.4, 1], [0.6, 1], [0.8, 0]], term="medium")
S7_3 = sf.FuzzySet(points=[[0.6, 0], [0.7, 1], [1, 1]], term="high")
ProcessLocallyInput = sf.LinguisticVariable([S7_1, S7_2, S7_3])
FS3.add_linguistic_variable("ProcessLocallyInput", ProcessLocallyInput)

S8_1 = sf.FuzzySet(points=[[0, 1], [0.3, 1], [0.4, 0]], term="low")
S8_2 = sf.FuzzySet(points=[[0.2, 0], [0.4, 1], [0.6, 1], [0.8, 0]], term="medium")
S8_3 = sf.FuzzySet(points=[[0.6, 0], [0.7, 1], [1, 1]], term="high")
InputDataCloudInput = sf.LinguisticVariable([S8_1, S8_2, S8_3])
FS3.add_linguistic_variable("InputDataCloudInput", InputDataCloudInput)


S9_1 = sf.FuzzySet(points=[[-1, 1], [-0.6, 1], [-0.2, 0]], term="decrease")
S9_2 = sf.FuzzySet(points=[[-0.4, 0], [0, 1], [0.3, 1], [0.9, 0]], term="maintain")
S9_3 = sf.FuzzySet(points=[[0.4, 0], [0.9, 1], [1, 1]], term="increase")
CLP_Variation = sf.LinguisticVariable([S9_1, S9_2, S9_3])
FS3.add_linguistic_variable("CLP_Variation", CLP_Variation)

# Construct Rule Base 3
FS3.add_rules(["IF (ProcessLocallyInput IS low) AND (InputDataCloudInput IS low) THEN (CLP_Variation IS decrease)",
               "IF (ProcessLocallyInput IS low) AND (InputDataCloudInput IS medium) THEN (CLP_Variation IS decrease)",
               "IF (ProcessLocallyInput IS low) AND (InputDataCloudInput IS high) THEN (CLP_Variation IS decrease)",
               "IF (ProcessLocallyInput IS medium) AND (InputDataCloudInput IS low) THEN (CLP_Variation IS decrease)",
               "IF (ProcessLocallyInput IS medium) AND (InputDataCloudInput IS medium) THEN (CLP_Variation IS maintain)",
               "IF (ProcessLocallyInput IS medium) AND (InputDataCloudInput IS high) THEN (CLP_Variation IS increase)",
               "IF (ProcessLocallyInput IS high) AND (InputDataCloudInput IS low) THEN (CLP_Variation IS decrease)",
               "IF (ProcessLocallyInput IS high) AND (InputDataCloudInput IS medium) THEN (CLP_Variation IS increase)",
               "IF (ProcessLocallyInput IS high) AND (InputDataCloudInput IS high) THEN (CLP_Variation IS increase)"
               ])

# Predefine empty list for result values
CLPVarResults = []

# Construct TestMe Result File
for i in range(len(TestMe)):
    MU, PL, OB, LA = TestMe.iloc[i, 0], TestMe.iloc[i, 1], TestMe.iloc[i, 4], TestMe.iloc[i, 5]
    FS1.set_variable("Latency", LA)
    FS1.set_variable("ProcessorLoad", PL)
    CLP_C1 = FS1.inference()['ProcessLocally']

    FS2.set_variable("OutBandwidth", OB)
    FS2.set_variable("MemoryUsage", MU)
    CLP_C2 = FS2.inference()['InputDataCloud']

    FS3.set_variable("ProcessLocallyInput", CLP_C1)
    FS3.set_variable("InputDataCloudInput", CLP_C2)
    CLP_C3 = FS3.inference()['CLP_Variation']

    CLPVarResults.append(CLP_C3)

df = pd.DataFrame(CLPVarResults, columns=["CLPVariation"])
df.to_csv("TestResult.csv", index=False)

# Predefine empty list for the generated data
Dataset = []

# Generate data to train NN, save as fuzzy_dataset.csv
for _ in range(num_data_points):
    # Randomly generate values for features within the defined range
    MU = random.uniform(min_MU, max_MU)
    PL = random.uniform(min_PL, max_PL)
    OB = random.uniform(min_OB, max_OB)
    LA = random.uniform(min_LA, max_LA)
    INT = random.uniform(min_MU, max_MU) #these features generated to have the equal inputs to NN
    ONT = random.uniform(min_MU, max_MU) #using the min and max MU because is the same range
    
    # randomly generate values for variation rates of each features
    VMU = random.uniform(min_MU, max_MU)
    VPL = random.uniform(min_PL, max_PL)
    VOB = random.uniform(min_OB, max_OB)
    VLA = random.uniform(min_LA, max_LA)
    VINT = random.uniform(min_MU, max_MU) 
    VONT = random.uniform(min_MU, max_MU)
    
    FS1.set_variable("Latency", LA)
    FS1.set_variable("ProcessorLoad", PL)
    CLP_C1 = FS1.inference()['ProcessLocally']

    FS2.set_variable("OutBandwidth", OB)
    FS2.set_variable("MemoryUsage", MU)
    CLP_C2 = FS2.inference()['InputDataCloud']

    FS3.set_variable("ProcessLocallyInput", CLP_C1)
    FS3.set_variable("InputDataCloudInput", CLP_C2)
    CLP_C3 = FS3.inference()['CLP_Variation']

    Dataset.append([MU, PL, INT, ONT, OB, LA, VMU, VPL, VOB, VLA, VINT, VONT, CLP_C3])

df = pd.DataFrame(Dataset, columns=["MemoryUsage", "ProcessorLoad","InpNetThroughput","OutNetThroughput", "OutBandwidth", "Latency","V_MemoryUsage","V_ProcessorLoad","V_InpNetThroughput","V_OutNetThroughput","V_OutBandwidth","V_Latency", "CLPVariation"])
df.to_csv("fuzzy_dataset.csv", index=False)
