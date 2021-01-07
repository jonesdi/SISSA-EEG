#coding=utf-8
import psychopy
import pandas as pd
import random as rn
import numpy as np
import itertools
import collections
import random 
import os
import time
import pickle

from psychopy import visual, core, event, gui, parallel, prefs
from psychopy.hardware import keyboard
from utils import draw, format_instr, print_instr, create_run_splits

#########################
##### settings  #########
#########################

# Setting the parallel port for EEG triggers

port_number = 888
#outputPort = parallel.ParallelPort(port_number)
#outputPort.setData(0)

### subject, actual runs to be ran (out of 32), refresh rate of the screen
gui = psychopy.gui.Dlg()
gui.addField("Subject ID:")
gui.addField('Refresh rate')
gui.show()

subjectId = int(gui.data[0])
refresh = int(gui.data[1])

#prefs.general['winType'] = 'pyglet'

### make window
### TO DO: adapting screen size to whatever size so as to avoid crashes/bugs
#win = visual.Window(size=[1920,1080], fullscr=False, color=[-1,-1,-1], units='pix', winType='pyglet', screen=0, checkTiming=True)
win = visual.Window(size=[1920,1080], fullscr=False, color=[-1,-1,-1], units='pix', checkTiming=True)
loadingMessage = 'L\'esperimento si sta caricando...'
textStimulus = visual.TextStim(win, loadingMessage, color=[.8,.8,.8], pos=[0,0], ori=0, wrapWidth=1080, height=30)

### Printing the loading message

textStimulus.autoDraw = True

### Setting the presentation time
presentationFrames = 4 # 4 frames on a 144hz screen = 27ms 
minuteCounter = {step : value for step, value in enumerate([abs(i) for i in range(-59, 1) for k in range(refresh)])}

### Printing out to output the screen setting, for debugging

print('Frame rate detected: {}'.format(win.getActualFrameRate()))
actualFrameMs = win.getMsPerFrame(nFrames=180)[2]
predictedFrameMs = win.monitorFramePeriod
print('Predicted milliseconds per frame: {}'.format(predictedFrameMs*1000.))
print('Actual milliseconds per frame: {}'.format(actualFrameMs))
print('Actual milliseconds per-stimulus in this experiment: {} = {}'.format(presentationFrames, presentationFrames*actualFrameMs))

### Setting up the keyboard
keysMapping = [{'d' : 'animal', 'k' : 'object'}, {'d' : 'object', 'k' :  'animal'}]
EngToIta = {'animal' : 'Animale', 'object' : 'Oggetto inanimato'}

### Setting and creating the output folder

timeNow = time.strftime('%d_%b_%Hh%M', time.gmtime())
outputIdentifier = '{}_refresh_rate_{}'.format(timeNow, refresh)
subjectPath = os.path.join('results', outputIdentifier, 'sub_{:02}'.format(subjectId))
os.makedirs(subjectPath, exist_ok=True) 

### load the stimulus set
Stimuli = pd.read_csv('stimuli_final.csv', delimiter=';')

### Creating the n runs, which allow to obtain n/2 trials per stimulus 

runs = 32
final_runs = create_run_splits(runs)
with open(os.path.join(subjectPath, 'runs_split.pkl'), 'wb') as runsPickle:
    pickle.dump(final_runs, runsPickle)
print(final_runs.keys())

#with open(os.path.join('results/30_Nov_13h46_refresh_rate_144/sub_01/runs_split.pkl'), 'rb') as input_file:
    #final_runs = pickle.load(input_file)

### Messages

### Defining the general mask
mask = '##########'
#mask.height = 40

### Defining the main instructions

instrIntro1 = 'In questo esperimento, vedrai una serie di parole presentate sullo schermo del computer. \n\n Compariranno una alla volta, e per un tempo molto breve. \n\nQualche volta sarà difficile vederle: non preoccuparti, è tutto previsto dall\'esperimento. Dopo che la parola sarà apparsa, passerà qualche momento di vuoto in cui ti chiediamo di pensare al significato della parola stessa. Subito dopo, ti chiederemo di dirci se la parola si riferiva a: \n\n - un animale \n\n - un oggetto inanimato. \n\n\n [Premi la barra spaziatrice per continuare]'

# taking away the color coding
#instrIntro2 = format_instr(win, text='Cerca di rispondere il più velocemente possibile, usando il tasto D (rosso) per gli animali e il tasto K (verde) per gli oggetti inanimati. \n Talvolta i tasti corrispondenti cambieranno, verrai informato prima di ogni sezione dell\'esperimento. \n \n [Premi la barra spaziatrice per continuare]')
instrIntro2 = 'Cerca di rispondere il più velocemente possibile, premendo: \n\n- il tasto D per gli animali\n\n- il tasto K per gli oggetti inanimati. \n\n Talvolta i tasti si invertiranno, ma te lo diremo prima di ogni sezione.\n\n\n [Premi la barra spaziatrice per continuare]'
instrIntro3 = 'Dopo che avrai premuto il tasto, ti chiederemo anche quanto sicura/o sei della tua risposta. \n\nCome dicevamo, la parola potrebbe essere difficile da vedere in qualche occasione, per cui a volte sarai più sicura/o della tua risposta, a volte meno. \n\nVedrai la domanda sullo schermo. Per indicare quanto sei sicuro/a, potrai usare i tasti:\n\n - 1 (per niente sicura/o)\n\n- 2 (abbastanza sicura/o)\n\n - 3 (molto sicura/o). \n\n\n [Premi la barra spaziatrice per continuare]'
instrIntro4 = 'È importante che cerchi di rispondere alla domanda animale/oggetto inanimato anche quando ti sembrerà di non aver letto la parola per nulla. \n\nIn quei casi, fidati del tuo intuito, anche se sarai poco sicura/o della tua risposta. \n\n\n [Premi la barra spaziatrice per fare una prova]'
#instrReminder = format_instr(win, text='I tasti per questa sessione sono: \n\n - D per gli animali \n\n - K per gli oggetti inanimati \n\n\n [Premi la barra spaziatrice]')
instrRandomMapping = lambda dKey, kKey : 'I tasti per questa sessione sono: \n\n - D per {} \t\t\t - K per {} \n\n\n [Premi la barra spaziatrice]'.format(dKey.upper(), kKey.upper())
questionStimulus = lambda dKey, kKey : '{} (D)\t\toppure\t\t{} (K)?\n'.format(dKey.upper(), kKey.upper())
instrGoOn = '[Premi la barra spaziatrice per procedere con la prossima parola] \n'
question = 'Dicci per favore quanto sei sicura/o della tua risposta, premendo: \n\n - 1 (per niente sicura/o)\n\n- 2 (abbastanza sicura/o)\n\n - 3 (molto sicura/o)'
AfterRest = 'Procediamo! \n\n\n [Premi la barra spaziatrice]'
Finished = 'Questa era l\'ultima sessione, e l\'esperimento ora è finito. \n\n\nGrazie per aver partecipato!'
StartExample = 'Cominciamo con qualche parola di prova! \n\n\n [Premi la barra spaziatrice]'

instrMain = 'Ora procediamo con l\'esperimento vero e proprio! Da ora in avanti, non riceverai più alcuna valutazione corretto/sbagliato sulle tue scelte. \n\nSe hai delle domande, ora è il momento di farle. \n\n\n [Altrimenti, premi la barra spaziatrice]'

############################
#### Show instructions #####
############################

### show main instruction


textStimulus.text = instrIntro1
win.flip()
event.waitKeys(keyList=['space'])

textStimulus.text = instrIntro2
win.flip()
event.waitKeys(keyList=['space'])

textStimulus.text = instrIntro3
win.flip()
event.waitKeys(keyList=['space'])

textStimulus.text = instrIntro4
win.flip()
event.waitKeys(keyList=['space'])
    
### Show example instructions

textStimulus.text = StartExample
win.flip()
event.waitKeys(keyList=['space'])

trialKeysMapping = keysMapping[0]
instrMessage = instrRandomMapping(EngToIta[[v for k, v in trialKeysMapping.items()][0]], EngToIta[[v for k, v in trialKeysMapping.items()][1]])

textStimulus.text = instrMessage
win.flip()
event.waitKeys(keyList=['space'])

#####################
#start example trials
#####################

# Selecting random words from the left-out filler stimuli
randomIndices = random.sample([k for k in range(30, 35)], k=2) + random.sample([k for k in range(35, 40)], k=2)

### Starting the 4 pre-experiment trials
for exampleNum, exampleIndex in enumerate(randomIndices):

    exampleWord = Stimuli['word'][exampleIndex]

    clock = core.Clock()
    while clock.getTime()< 0.5:
        textStimulus.text = mask
        win.flip()
    clock = core.Clock()
    while clock.getTime()< 0.023:
        textStimulus.text = exampleWord
        win.flip()
    clock = core.Clock()
    while clock.getTime()< 1.5:
        textStimulus.text = mask
        win.flip()

    ### Waiting for an answer and then collecting it

            
    questionMessage = questionStimulus(EngToIta[[v for k, v in trialKeysMapping.items()][0]], EngToIta[[v for k, v in trialKeysMapping.items()][1]])
    textStimulus.text = questionMessage
    win.flip()
    responses = event.waitKeys(keyList=['d','k'])
    
    responseKey = responses[0]

    #ask how sure you are
    
    textStimulus.text = question
    win.flip()
    responseSureTemp = event.waitKeys(keyList=['1','2','3'])

    ### feedback
    outcome = 'Corretto' if trialKeysMapping[responseKey] == Stimuli['category'][exampleIndex] else 'Sbagliato'
    outcomeExample = lambda outcome, word: '{}!\n\nLa parola era \'{}\''.format(outcome, word)
    textStimulus.text = outcomeExample(outcome, exampleWord)
    win.flip()
    core.wait(1)

    ### press space to go to next trial
    if exampleNum<3:
        
        textStimulus.text = instrGoOn
        win.flip()
        event.waitKeys(keyList=['space'])

#########################
#### start main exp #####
#########################

textStimulus.text = instrMain

for runNum in (range(1, 33)):  # 32 runs
    
    runResults = collections.defaultdict(list)

    trialKeysMapping = keysMapping[random.choice([0, 1])]
    
    instrMessage = instrRandomMapping(EngToIta[[v for k, v in trialKeysMapping.items()][0]], EngToIta[[v for k, v in trialKeysMapping.items()][1]])
    textStimulus.text = instrMessage
    win.flip()
    event.waitKeys(keyList=['space'])

    ### start trials
    for trialIndex, trialStimulus in enumerate(final_runs[runNum]):

        trialWord = Stimuli['word'][trialStimulus]

        clock = core.Clock()
        while clock.getTime()< 0.5:
            textStimulus.text = mask
            win.flip()
        #outputPort.setData(int(trialStimulus)+1) # Sending the EEG trigger, opening the parallel port with the trialNum number
        clock = core.Clock()
        #while clock.getTime()< 0.02:
        for i in range(4):
            textStimulus.text = trialWord
            win.flip()
        stimulusDuration = clock.getTime() # stores stimulus presentation duration
        #outputPort.setData(0) # Closing the parallel port
        clock = core.Clock()
        while clock.getTime()< 1.5:
            textStimulus.text = mask
            win.flip()

        ### Waiting for an answer and then collecting it

            
        questionMessage = questionStimulus(EngToIta[[v for k, v in trialKeysMapping.items()][0]], EngToIta[[v for k, v in trialKeysMapping.items()][1]])
        textStimulus.text = questionMessage
        win.flip()
        
        clock = core.Clock()
        responses = event.waitKeys(keyList=['d','k'], timeStamped=clock)
        responseKey, responseTime = responses[0][0], responses[0][1]

        ### ask how sure you are
        textStimulus.text = question
        win.flip()
        responseCertainty = event.waitKeys(keyList=['1','2','3'])[0]
        
        ### Checking whether the response is correct or not
        trialCategory = Stimuli['category'][trialStimulus]
        predictionOutcome = 'correct' if trialCategory == trialKeysMapping[responseKey] else 'wrong'
        
        ### Updating the results dictionary with: word, group, trigger code/word index, correct/wrong prediction, response time, stimulus duration
        runResults[trialIndex+1] = [trialWord.lower(), Stimuli['group'][trialStimulus], trialStimulus, predictionOutcome, responseTime, responseCertainty, stimulusDuration]

        #press space to go to next trial, or the next run if that's the end
        if trialIndex<19:

            textStimulus.text = instrGoOn
            win.flip()
            event.waitKeys(keyList=['space'])
    
    #assert len(runResults.items()) == 20 # debugging: checking all went OK

    ### Saving the output
    runDataFrame = pd.DataFrame([[k] + v for k, v in runResults.items()], columns=['Trial number', 'Word', 'Group','Trigger code', 'Prediction outcome', 'Response time', 'Certainty', 'Stimulus duration (ms)']) # turning the dictionary into a pandas data frame
    runDataFrame.to_csv(os.path.join(subjectPath, 'run_{:02}_events_log.csv'.format(runNum)), index=False) # exporting to file the pandas data frame

    ### rest 1 min
    if runNum<32:
        for _ in range(int(refresh*60)): 
            countdown = minuteCounter[_]
            restText = lambda run : 'Fine della sessione {} su 32 - ottimo lavoro!\n\n Ora hai 1 minuto di pausa prima di cominciare la prossima sessione.\n\n 00.{:02}'.format(run, countdown)
            textStimulus.text = restText(runNum)
            win.flip()
            keypress = event.getKeys(keyList=['n'])
            if len(keypress)>0 or _ == int(refresh*60):
            #if _ == int(refresh*60):
                textStimulus.text = AfterRest
                break
    else:
        textStimulus.text = Finished
        win.flip()
        core.wait(5)

win.close()