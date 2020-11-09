#coding=utf-8
import psychopy
from psychopy import visual, core, event, gui
import pandas as pd
import random as rn
import numpy as np
import itertools

#########################
##### settings  #########
#########################
#subject and output
gui = psychopy.gui.Dlg()
gui.addField("Subject ID:")
gui.addField("Number of runs:")
gui.addField('Refresh cycles, 144/...')
gui.show()
subjectId = gui.data[0]
runs = int(gui.data[1])
timeStamp = core.getAbsTime()
outputFileName_pptResponse = 'out_' + subjectId + '_' + str(timeStamp) + '_refresh' + gui.data[2] + '.csv'
refresh=144

#preallocate ppts' responses
runId = list() #run id
stimuliDF = pd.DataFrame() #word & category
presentationId = list() #word presentation id - according to order of presentation (max 20)
responseCatDF = pd.DataFrame() #response category & response time
responseSure = list() #response sure

response = pd.DataFrame() #overall result


#set exmaples
examples = [['pesce','animal'],['cellulare','object'],['uccello','animal'],['specchio','object']]
examples = pd.DataFrame(examples, columns=['word','cat'])
#Stimuli = examples #to use fewer stumuli for test, i use examples for the main part as well

#load the stimulus set
#Stimuli = pd.read_csv('D:/Ekaterina/stimuli_final.csv', delimiter=';')
Stimuli = pd.read_csv('stimuli_final.csv', delimiter=';')

### TO DO: thinking about data splitting/randomization, also depending on the number of runs etc
Stimuli_1half = Stimuli[0:5]
Stimuli_1half = Stimuli_1half.append(Stimuli[10:15])
Fillers_1half = Stimuli[20:25]
Fillers_1half = Fillers_1half.append(Stimuli[35:40])
Set_1 = pd.concat([Stimuli_1half, Fillers_1half])

Stimuli_2half = Stimuli[5:10]
Stimuli_2half = Stimuli_2half.append(Stimuli[15:20])
Fillers_2half = Stimuli[25:30]
Fillers_2half = Fillers_1half.append(Stimuli[40:45])
Set_2 = pd.concat([Stimuli_2half, Fillers_2half])

runs = 16 # 20 words per run: 5 animals, 5 objects, 10 fillers

targets = [l for l in Stimuli if l[2] == 'target']
fillers = [l for l in Stimuli if l[2] != 'target']


NumStimuli=20
NumTargets=10

#########################
##### instructions ######
#########################

#make window and mask
### TO DO: adapting screen size to whatever size so as to avoid crashes/bugs
win = visual.Window(size=[1920,1080], fullscr=False, color=[-1,-1,-1], units='pix')
mask = visual.TextStim(win, text='##########', color=[.8,.8,.8], pos=[0,0], ori=0)

#main instructions
instrIntro1 = visual.TextStim(win, text='In questo esperimento, vedrai una serie di parole presentate sullo schermo del computer. \n Esse compariranno una alla volta, e per un tempo molto breve. Qualche volta sarà difficile vederle: non preoccuparti, è tutto previsto dall\'esperimento. \n Ti chiediamo di dirci se la parola che di volta in volta vedrai si riferisce a un animale oppure a un oggetto inanimato. \n \n [Premi la barra spaziatrice per continuare]', color=[.8,.8,.8], pos=[0,0], ori=0)
instrIntro2 = visual.TextStim(win, text='Cerca di rispondere il più velocemente possibile, usando il tasto D (rosso) per gli animali e il tasto K (verde) per gli oggetti inanimati. \n Talvolta i tasti corrispondenti cambieranno, verrai informato prima di ogni sezione dell\'esperimento. \n \n [Premi la barra spaziatrice per continuare]', color=[.8,.8,.8], pos=[0,0], ori=0)
instrIntro3 = visual.TextStim(win, text='Dopo che avrai premuto il tasto, ti chiederemo anche quanto sicuro sei della tua risposta: come dicevamo, la parola potrebbe essere difficile da vedere in qualche occasione, per cui a volte sarai più sicura/o della tua risposta, a volte meno. Vedrai la domanda sullo schermo. La scala di sicurezza va da 1 (per niente sicuro) a 3 (molto sicuro). \n \n [Premi la barra spaziatrice per continuare]', color=[.8,.8,.8], pos=[0,0], ori=0)
instrIntro4 = visual.TextStim(win, text='È importante che cerchi di rispondere alla domanda animale/oggetto inanimato anche quando ti sembrerà di non aver letto la parola per nulla: in quei casi, fidati della tua intuizione, anche se sarai molto poco sicuro della tua risposta. \n \n Premi la barra spaziatrice per cominciare con qualche esempio', color=[.8,.8,.8], pos=[0,0], ori=0)
question = visual.TextStim(win, text='Dicci per favore quanto sei sicuro dell tua risposta, da 1 (per niente sicuro) a 3 (molto sicuro) usando i tasti 1,2,3', color=[.8,.8,.8])
Rest = visual.TextStim(win, text='Pausa 1 minuto \n Se vuoi continuare adesso, premi la barra', color=[.8,.8,.8])
AfterRest = visual.TextStim(win, text='Procediamo \n Premi la barra', color=[.8,.8,.8])
Finished = visual.TextStim(win, text='Grazie per la partecipazione!', color=[.8,.8,.8])

#example instructions
instrCorrect = visual.TextStim(win, text='Corretto!', color=[.8,.8,.8])
instrWrong = visual.TextStim(win, text='Sbagliato!', color=[.8,.8,.8])
instrMain = visual.TextStim(win, text='Procediamo ora con la sezione principale \n Premi la barra', color=[.8,.8,.8])

#keys mapping to choose from
instrMapping = ['Rosso = animale, verde = oggetto','Rosso = oggetto, verde = animale']
Keys = pd.DataFrame(columns=['instr','d','k'], index=['1','2']) #dataframe of keymapping and corresponding instructions
Keys['instr'] = instrMapping
Keys['d'] = ['animal','object']
Keys['k'] = ['object','animal']
MappingHistory=list()

#########################
##### functions #########
#########################

def print_instr(instr_type, wait):
    instr_type.draw(win=win)
    win.flip()
    event.waitKeys(keyList=['space'])
    win.flip()
    core.wait(wait)

def draw(stimulus, cycles):
    for _ in range(cycles):
        stimulus.draw(win=win)
        win.flip()

#########################
#### start examples #####
#########################

#show main instruction
print_instr(instrIntro1,0)
print_instr(instrIntro2,0)
print_instr(instrIntro3,0)
print_instr(instrIntro4,0.5)

#keys randomizing
MappingRand = pd.DataFrame.sample(Keys)
#keys instruction
Mapping = MappingRand['instr'][0]
MappingKeys = visual.TextStim(win, text='{} \n Premi la barra'.format(Mapping), color=[.8,.8,.8])
print_instr(MappingKeys,1)

#start example trials
for exampleNum in range (4):

    example = visual.TextStim(win, text=examples['word'][exampleNum], color=[.8,.8,.8], pos=[0,0], ori=0)

    draw(mask,int(refresh/2))
    draw(example,int(refresh/36))
    clock = core.Clock()
    draw(mask,int(refresh*1.5))

    #give category response
    responseNotGiven = True
    while responseNotGiven:
        responseCatTemp = event.getKeys(keyList=['d','k'], timeStamped=clock)
        if len(responseCatTemp)==1:
            responseNotGiven = False

    #feedback
    if MappingRand[responseCatTemp[0][0]][0] == examples['cat'][exampleNum]:
        instrCorrect.draw(win=win)
        win.flip()
        core.wait(1)
    else:
        instrWrong.draw(win=win)
        win.flip()
        core.wait(1)

    #ask how sure you are
    question.draw(win=win)
    win.flip()
    responseSureTemp = event.waitKeys(keyList=['1','2','3'])

    #press space to go to next trial
    if exampleNum<3:
        instrGoOn = visual.TextStim(win, text='Premi la barra per procedere con la prossima parola. \n {}'.format(Mapping), color=[.8,.8,.8])
        print_instr(instrGoOn, 0.5)

print_instr(instrMain,0.5)
#########################
#### start main exp #####
########################

for runNum in range(runs):

    responseCatPerRun=list() #temporary category+time response for each run

    #show how many runs remain
    if runNum>0:
        instrRun = visual.TextStim(win, text='Rimangono {} / {} sezioni. \n Premi la barra'.format(runs-runNum,runs), color=[.8,.8,.8])
        print_instr(instrRun,1)

    #key randomizing. show key mapping for the run
    MappingRand = pd.DataFrame.sample(Keys)
    Mapping = MappingRand['instr'][0]
    MappingKeys = visual.TextStim(win, text='Attenzione! I tasti per questa sezione sono: \n {} \n Premi la barra'.format(Mapping), color=[.8,.8,.8])
    print_instr(MappingKeys,1)
    MappingHistory.append(Mapping)

    ### TO DO: see above, randomize only once, making sure that stimuli are not presented more than once in each run

    #randomize the stimuli
    if runNum % 2 == 0:
        Set = Set_1
    else:
        Set = Set_2
    
    Set = Set.reset_index(drop=True)
    StimuliRand = Set.reindex(rn.sample(range(NumStimuli), NumStimuli))
    StimuliRand.index = range(NumStimuli) 

    #start trials
    for trialNum in range(NumStimuli):

        #temporary response for the trial
        responseCatTemp = ()
        responseSureTemp = ()

        word = visual.TextStim(win, text=StimuliRand['word'][trialNum], color=[.8,.8,.8], pos=[0,0], ori=0)
        
        if StimuliRand['group'][trialNum]=='target':
            presentationId.append(trialNum)

        draw(mask,int(refresh/2))
        draw(word,int(refresh/36))
        clock = core.Clock()
        draw(mask,int(refresh*2))

        ## TO DO: clean up this part
        #press a key for category (prevents no response). here i decided that if the ppt accidentally presses the key twice, i take the second response, assuming that the ppt wanted to correct a mistake
        responseNotGiven = True
        while responseNotGiven:
            responseCatTemp = event.getKeys(keyList=['d','k'], timeStamped=clock)
            if len(responseCatTemp)==1:
                responseNotGiven = False
                if StimuliRand['group'][trialNum]=='target':
                    responseCatPerRun.append(responseCatTemp)
            elif len(responseCatTemp)>1:
                responseNotGiven = False
                if StimuliRand['group'][trialNum]=='target':
                    responseCatPerRun.append(responseCatTemp[1])

        win.flip()
        core.wait(0.5)

        #ask how sure you are
        if trialNum<NumStimuli-1:
            question.draw(win=win)
            win.flip()
            responseSureTemp = event.waitKeys(keyList=['1','2','3'])
            if StimuliRand['group'][trialNum]=='target':
                responseSure.append(responseSureTemp)
        else: #to wait a little after the last trial
            question.draw(win=win)
            win.flip()
            responseSureTemp = event.waitKeys(keyList=['1','2','3'])
            if StimuliRand['group'][trialNum]=='target':
                responseSure.append(responseSureTemp)
            win.flip()
            core.wait(0.5)

        #press space to go to next trial
        if trialNum<NumStimuli-1:
            instrGoOn = visual.TextStim(win, text='Premi la barra per procedere con la prossima parola. \n {}'.format(Mapping), color=[.8,.8,.8])
            print_instr(instrGoOn,0)

    #save the output for the run
    for _ in range(NumTargets):
        runId.append(str(runNum))
    for _ in range (len(StimuliRand)):
        if StimuliRand['group'][_]=='target':
            stimuliDF = stimuliDF.append(StimuliRand.loc[_])
    stimuliDF.to_csv('stimuli_'+outputFileName_pptResponse, index=False)
    #map the keys of this run with the category and save the response
    responseCatPerRun = list(itertools.chain(*responseCatPerRun))
    responseCatPerRun = pd.DataFrame(responseCatPerRun, columns=['responseCat','responseTime'])
    responseCatPerRun['responseCat'] = responseCatPerRun['responseCat'].map({'d' : MappingRand['d'][0], 'k' : MappingRand['k'][0]})
    responseCatDF = responseCatDF.append(responseCatPerRun)
    responseCatDF.to_csv('resp_'+outputFileName_pptResponse, index=False)

    #rest 1 min or on keypress
    if runNum<runs-1:
        for _ in range(int(refresh*60)):
            Rest.draw(win=win)
            win.flip()
            keypress = event.getKeys(keyList=['space'])
            if len(keypress)>0 or _ == int(refresh*60):
                print_instr(AfterRest,0.5)
                break
    else:
        Finished.draw(win=win)
        win.flip()
        core.wait(1)

win.close()

#########################
#### save the result ####
#########################

#saving results as dataframes
runIdDF = pd.DataFrame(runId)
presentationIdDF = pd.DataFrame(presentationId)
sureDF = pd.DataFrame(responseSure)
#drop indices
runIdDF = runIdDF.reset_index(drop=True)
presentationIdDF = presentationIdDF.reset_index(drop=True)
stimuliDF = stimuliDF.reset_index(drop=True)
responseCatDF = responseCatDF.reset_index(drop=True)
sureDF = sureDF.reset_index(drop=True)
#concatenate into one dataframe
DFlist = [runIdDF, presentationIdDF, stimuliDF, responseCatDF, sureDF]
response = pd.concat(DFlist, axis=1)
response.columns = ['runId','presentationId','category','group','word','responseCat','responseTime','responseSure']
#save
response.to_csv(outputFileName_pptResponse, index=False)
