This dataset is a collection of episodes (full play throughs) of the following 7 atari games:
- BeamRider
- Breakout
- Enduro
- Pong
- Qbert
- Seaquest
- SpaceInvaders

Each game has ~200k (normal) frames collected using the OpenAI baseline implementation of A2C. 
Files are stored in hdf5 format and are loaded as numpy arrays. Images (states) are in HWC, RBG, uint8 (0-255) format. 

Files can be loaded in python using the h5py module:

-- pip install h5py
-- pip install numpy

with the following python code:
---------------------------------------------------------------
import h5py

path = "./episode.hdf5"
file = h5py.File(path, 'r')
states = file['state'][...] 
actions = file['action'][...]
labels = file['label'][...]   # for ANOMALY - state labels
tlabels = file['tlabel'][...] # for ANOMALY - transition labels
---------------------------------------------------------------

Details of each collection of episodes the corresponding meta.txt files.

RAW: 
Contains ~100k frames and associated actions for each game.

CLEAN: 
Contains ~100k frames and associated actions for each game (different to RAW) - the clean version of ANOMALY.

ANOMALY:
Contains ~100k frames for each game, artificial anomalies have been introduced into CLEAN. Anomalies include, freezing, flickering and visual artefacts. Anomalous and normal states/transitions are labelled 1 and 0 respectively. 

META DATA:

------------- STATE SPACE --------------
BeamRiderNoFrameskip-v4: 	Discrete(210, 160, 3)
BreakoutNoFrameskip-v4: 	Discrete(210, 160, 3)
EnduroNoFrameskip-v4: 		Discrete(210, 160, 3)
PongNoFrameskip-v4: 		Discrete(210, 160, 3)
QbertNoFrameskip-v4: 		Discrete(210, 160, 3)
SeaquestNoFrameskip-v4: 	Discrete(210, 160, 3)
SpaceInvadersNoFrameskip-v4: 	Discrete(210, 160, 3)
----------------------------------------

ACTION META DATA:

------------- ACTION SPACE --------------
BeamRiderNoFrameskip-v4: 	Discrete(9)
BreakoutNoFrameskip-v4: 	Discrete(4)
EnduroNoFrameskip-v4: 		Discrete(9)
PongNoFrameskip-v4: 		Discrete(6)
QbertNoFrameskip-v4: 		Discrete(6)
SeaquestNoFrameskip-v4: 	Discrete(18)
SpaceInvadersNoFrameskip-v4: 	Discrete(6)
----------------------------------------
