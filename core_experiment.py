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

from psychopy import visual, core, event, gui, parallel
from utils import draw, format_instr, print_instr, create_run_splits

#########################
##### settings  #########
#########################

# Setting the parallel port for EEG triggers

#port_number = 00000
#parallel.setPortAddress(port_number)

#subject, actual runs to be ran (out of 32), refresh rate and output
gui = psychopy.gui.Dlg()
gui.addField("Subject ID:")
gui.addField("Number of runs:")
gui.addField('Refresh cycles, 144/...')
gui.show()

subjectId = int(gui.data[0])
actual_runs = int(gui.data[1])  
refresh = int(gui.data[2])

keysMapping = {'k' : 'object', 'd' : 'animal'}
timeStamp = core.getAbsTime()

timeNow = time.strftime('%d_%b_%Hh%M', time.gmtime())
outputIdentifier = '{}_refresh_rate_{}'.format(timeNow, gui.data[2])
subjectPath = os.path.join('results', outputIdentifier, 'sub_{:02}'.format(subjectId))
os.makedirs(subjectPath, exist_ok=True) 

#load the stimulus set
Stimuli = pd.read_csv('stimuli_final.csv', delimiter=';')

# Creating the n runs, which allow to obtain n/2 trials per stimulus 

runs = 32
final_runs = create_run_splits(runs)

#########################
##### instructions ######
#########################

#make window and mask
### TO DO: adapting screen size to whatever size so as to avoid crashes/bugs
win = visual.Window(size=[1920,1080], fullscr=False, color=[-1,-1,-1], units='pix')
mask = format_instr(win, text='##########')

#main instructions
instrIntro1 = format_instr(win, text='In questo esperimento, vedrai una serie di parole presentate sullo schermo del computer. \n\n Compariranno una alla volta, e per un tempo molto breve. \n\nQualche volta sarà difficile vederle: non preoccuparti, è tutto previsto dall\'esperimento. Ti chiediamo di dirci se la parola che di volta in volta vedrai si riferisce a: \n\n - un animale \n\n - un oggetto inanimato. \n\n\n [Premi la barra spaziatrice per continuare]')

# taking away the color coding
#instrIntro2 = format_instr(win, text='Cerca di rispondere il più velocemente possibile, usando il tasto D (rosso) per gli animali e il tasto K (verde) per gli oggetti inanimati. \n Talvolta i tasti corrispondenti cambieranno, verrai informato prima di ogni sezione dell\'esperimento. \n \n [Premi la barra spaziatrice per continuare]')
instrIntro2 = format_instr(win, text='Cerca di rispondere il più velocemente possibile, premendo: \n\n- il tasto D per gli animali\n\n- il tasto K per gli oggetti inanimati. \n\n\n [Premi la barra spaziatrice per continuare]')
instrIntro3 = format_instr(win, text='Dopo che avrai premuto il tasto, ti chiederemo anche quanto sicura/o sei della tua risposta. \n\nCome dicevamo, la parola potrebbe essere difficile da vedere in qualche occasione, per cui a volte sarai più sicura/o della tua risposta, a volte meno. \n\nVedrai la domanda sullo schermo. Per indicare quanto sei sicuro/a, potrai usare i tasti:\n\n - 1 (per niente sicura/o)\n\n- 2 (abbastanza sicura/o)\n\n - 3 (molto sicura/o). \n\n\n [Premi la barra spaziatrice per continuare]')
instrIntro4 = format_instr(win, text='È importante che cerchi di rispondere alla domanda animale/oggetto inanimato anche quando ti sembrerà di non aver letto la parola per nulla. \n\nIn quei casi, fidati del tuo intuito, anche se sarai poco sicura/o della tua risposta. \n\n\n [Premi la barra spaziatrice per fare una prova]')
instrReminder = format_instr(win, text='Ricorda, i tasti sono: \n\n - D per gli animali \n\n - K per gli oggetti inanimati \n\n\n [Premi la barra spaziatrice]')
instrGoOn = format_instr(win, text='[Premi la barra spaziatrice per procedere con la prossima parola] \n')
question = format_instr(win, text='Dicci per favore quanto sei sicura/o della tua risposta, premendo: \n\n - 1 (per niente sicura/o)\n\n- 2 (abbastanza sicura/o)\n\n - 3 (molto sicura/o)')
AfterRest = format_instr(win, text='Procediamo! \n\n\n [Premi la barra spaziatrice]')
Finished = format_instr(win, text='Questa era l\'ultima sessione, e l\'esperimento ora è finito. \n\n\nGrazie per aver partecipato!')
StartExample = format_instr(win, text='Cominciamo con qualche parola di prova! \n\n\n [Premi la barra spaziatrice]')


instrMain = format_instr(win, text='Ora procediamo con l\'esperimento vero e proprio! \n\nSe hai delle domande, ora è il momento di farle. \n\n\n [Altrimenti, premi la barra spaziatrice]')

############################
#### Show instructions #####
############################

#show main instruction
print_instr(win, instrIntro1,0)
print_instr(win, instrIntro2,0)
print_instr(win, instrIntro3,0)
print_instr(win, instrIntro4,0.5)

# Show example instructions
print_instr(win, StartExample, 0.5)
print_instr(win, instrReminder, 1)

#####################
#start example trials
#####################

# Selecting random words from the stimuli
randomIndices = random.choices([k for k in range(50)], k=4)

#Starting the 4 pre-experiment trials
for exampleNum, exampleIndex in enumerate(randomIndices):

    exampleWord = Stimuli['word'][exampleIndex]
    exampleStimulus = format_instr(win, text=exampleWord)

    draw(win, mask,int(refresh/2))
    draw(win, exampleStimulus,int(refresh/36))

    # Waiting for an answer and then collecting it
        
    responseNotGiven = True
    while responseNotGiven:
            
        #draw(win, mask,int(refresh*2))
        mask.draw(win=win)
        win.flip()
        responses = event.waitKeys(keyList=['d','k'])
        responseKey = responses[0]
        #responseCatPerRun.append(responseCatTemp)
        responseNotGiven = False

    win.flip()

    #feedback
    outcome = 'Corretto' if keysMapping[responseKey] == Stimuli['category'][exampleIndex] else 'Sbagliato'
    outcomeExample = lambda outcome, word: '{}!\n\nLa parola era \'{}\''.format(outcome, word)
    format_instr(win, text=outcomeExample(outcome, exampleWord)).draw(win=win)
    win.flip()
    core.wait(1)

    #ask how sure you are
    question.draw(win=win)
    win.flip()
    responseSureTemp = event.waitKeys(keyList=['1','2','3'])

    #press space to go to next trial
    if exampleNum<3:
        print_instr(win, instrGoOn, 0.5)

#########################
#### start main exp #####
########################

print_instr(win, instrMain,0.5)

for runNum in range(1, actual_runs+1):
    
    runResults = collections.defaultdict(list)

    print_instr(win, instrReminder, 1)

    #start trials
    for trialIndex, trialStimulus in enumerate(final_runs[runNum]):

        trialWord = Stimuli['word'][trialStimulus]
        word = format_instr(win, text=trialWord)
        
        draw(win, mask,int(refresh/2))
        #parallel.setData(trialNum) # Sending the EEG trigger, opening the parallel port with the trialNum number
        draw(win, word,int(refresh/36))
        #parallel.setData(0) # Closing the parallel port
        clock = core.Clock()
        win.flip()

        # Waiting for an answer and then collecting it
        
        responseNotGiven = True
        while responseNotGiven:
            
            mask.draw(win=win)
            win.flip()
            responses = event.waitKeys(keyList=['d','k'], timeStamped=clock)
            responseKey, responseTime = responses[0][0], responses[0][1]
            responseNotGiven = False

        win.flip()

        #ask how sure you are
        question.draw(win=win)
        win.flip()
        responseCertainty = event.waitKeys(keyList=['1','2','3'])[0]
        
        # Checking whether the response is correct or not
        trialCategory = Stimuli['category'][trialStimulus]
        predictionOutcome = 'correct' if trialCategory == keysMapping[responseKey] else 'wrong'
        
        # Updating the results dictionary with: word, group, trigger code/word index, correct/wrong prediction, response time
        runResults[trialIndex+1] = [trialWord, Stimuli['group'][trialStimulus], trialStimulus, predictionOutcome, responseTime, responseCertainty]

        #press space to go to next trial, or the next run if that's the end
        if trialIndex<19:
            print_instr(win, instrGoOn,0)
    
    assert len(runResults.items()) == 20 # checking all went OK
    runDataFrame = pd.DataFrame([[k] + v for k, v in runResults.items()], columns=['Trial number', 'Word', 'Group','Trigger code', 'Prediction outcome', 'Response time', 'Certainty']) # turning the dictionary into a pandas data frame
    runDataFrame.to_csv(os.path.join(subjectPath, 'run_{:02}_events_log.csv'.format(runNum)), index=False) # exporting to file the pandas data frame

    # rest 1 min or continue on keypress
    if runNum<actual_runs:
        for _ in range(int(refresh*60)):
            restText = lambda run : 'Fine della sessione {} su 32 - ottimo lavoro!\n\n Se vuoi, ora hai fino a 1 minuto di pausa prima di cominciare la prossima sessione.\n\n [Altrimenti, premi la barra spaziatrice]'.format(run)
            Rest = format_instr(win, text=restText(runNum))
            Rest.draw(win=win)
            win.flip()
            keypress = event.getKeys(keyList=['space'])
            if len(keypress)>0 or _ == int(refresh*60):
                print_instr(win, AfterRest,0.5)
                break
    else:
        Finished.draw(win=win)
        win.flip()
        core.wait(5)

win.close()
