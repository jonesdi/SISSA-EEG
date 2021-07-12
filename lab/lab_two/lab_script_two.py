import numpy
import os
import psychopy
import random

from psychopy import core, event, gui, parallel, visual

from messages_two import Messages
from utils_two import generate_runs

# Asking for the experimental session information

gui = gui.Dlg()
gui.addField('Subject ID:')
gui.addField('in the lab? (y/n)')
gui.show()

subject_number = int(gui.data[0])
in_the_lab = True if gui.data[1] ==  'y' else False

# Setting the output folder 

out_folder = os.path.join('results', \
                          'sub-{:02}_events'.format(subject_number))
os.makedirs(out_folder, exist_ok=True)

# Setting the parallel port for EEG triggers

if in_the_lab:
    port_number = 888
    outputPort = parallel.ParallelPort(port_number)
    outputPort.setData(0)

# Read words and properties
word_to_trigger = read_words_and_triggers()


# Loading the messages

messages = Messages()

# Generating the runs
words, questions, answers = generate_runs(animals_and_objects)

# Generating the screen
win = visual.Window(size=[1920,1080], fullscr=False, color=[-1,-1,-1], units='pix', checkTiming=True)

# Checking for the current screen's  frame rate

print('Frame rate detected: {}'.format(win.getActualFrameRate()))
actualFrameMs = win.getMsPerFrame(nFrames=180)[2]
predictedFrameMs = win.monitorFramePeriod
print('Predicted milliseconds per frame: {}'.format(predictedFrameMs*1000.))
print('Actual milliseconds per frame: {}'.format(actualFrameMs))

# Time utilities
one_second = int(win.getActualFrameRate())
minuteCounter = {step : value for step, value in enumerate([abs(i) for i in range(-59, 1) for k in range(one_second)])}

# Generating the noise

visualNoiseSize = 256 # Dimension in pixels of visual noise. Must be a power of 2
noiseSize = 256
noiseTexture = numpy.random.rand(visualNoiseSize,visualNoiseSize) #*2-1
visualNoise = visual.GratingStim(win=win, tex=noiseTexture, size=(visualNoiseSize,256), units='pix', \
                                 interpolate=False, \
                                 mask='circle')

# Generating the text stimulus

loadingMessage = 'Now loading...'
textStimulus = visual.TextStim(win, '', \
                               color=[0., 0., 0.], \
                               pos=[0,0], ori=0, wrapWidth=1080, \
                               height=40, font='Calibri')

# Instructions page 1
textStimulus.text = messages.instructions_one
textStimulus.draw()
win.flip()

event.waitKeys(keyList=['space'])

# Instructions page 2
textStimulus.text = messages.instructions_two
textStimulus.draw()
win.flip()

event.waitKeys(keyList=['space'])

# Start trial
textStimulus.text = messages.start_example
textStimulus.draw()
win.flip()

event.waitKeys(keyList=['space'])

# For the trial, one random run, 16/24 random trials
r = random.randint(0, 8)
random_trials = random.sample(list(range(33)), k=8)

# Starting from 0.5, then using the result as a starting point
new_opacity = 0.5

for t in random_trials:
    
    current_word = words[r, t]
    current_question = questions[r, t]
    current_answer = questions[r, t]

    # Fixation cross
    textStimulus.opacity = 1.
    textStimulus.text = '+'
    for i in range(one_second):
        textStimulus.draw()
        win.flip()
    
    textStimulus.opacity = new_opacity
    old_opacity = new_opacity
    textStimulus.text = current_word
    #print(textStimulus.opacity)
        
    # Mask 1
    for i in range(4):
        visualNoise.draw()
        win.flip()
    
    # Word flashing
    t = core.Clock()
    for i in range(2):
        textStimulus.draw()
        win.flip()
    stimulus_length = t.getTime()
    
    # Mask 2
    for i in range(4):
        visualNoise.draw()
        win.flip()
    
    textStimulus.opacity = 1.
    
    # Fixation cross
    textStimulus.text = '+'
    for i in range(one_second):
        textStimulus.draw()
        win.flip()
    
    # Subjective question (based on perceptual awareness scale)
    
    textStimulus.text = messages.pas

    textStimulus.draw()
    win.flip()
    subjective_answer, subjective_time = event.waitKeys(keyList=['1', '2', '3'], timeStamped=True)[0]
    
    # 1-on-1 staircase procedure
    if subjective_answer == '1':
        new_opacity = min(1., old_opacity+.02)
    elif subjective_answer == '2':
        new_opacity = min(1., old_opacity+.01)
    elif subjective_answer == '3':
        new_opacity = max(0., old_opacity-.02)
    
    # Objective question (based on semantic features)
    textStimulus.text = messages.obj_question(current_question)
    
    textStimulus.draw()
    win.flip()
    objective_answer, objective_time = event.waitKeys(keyList=['1', '3'], timeStamped=True)[0]

# Real experiment

textStimulus.text = messages.real_exp
textStimulus.draw()
win.flip()

event.waitKeys(keyList=['space'])

subject_results = list()

#for r in range(1):
for r in range(24):
    
    run_results = list()

    for t in range(33):
    #for t in range(3):
        
        current_word = words[r, t]
        current_question = questions[r, t]
        current_answer = questions[r, t]
        
        if in_the_lab:
            outputPort.setData(0) # Closing the parallel port
        trigger = word_to_trigger[current_word]
    
        # Fixation cross
        textStimulus.opacity = 1.
        textStimulus.text = '+'
        for i in range(one_second):
            textStimulus.draw()
            win.flip()
        
        textStimulus.opacity = new_opacity
        old_opacity = new_opacity
        textStimulus.text = current_word
        #print(textStimulus.opacity)
            
        # Mask 1
        for i in range(4):
            visualNoise.draw()
            win.flip()
        
        # Word flashing
        if in_the_lab:
            outputPort.setData(trigger) # Sending the EEG trigger, opening the parallel port with the trialNum number
        t = core.Clock()
        for i in range(2):
            textStimulus.draw()
            win.flip()
        stimulus_length = t.getTime()
        if in_the_lab:
            outputPort.setData(0) # Closing the parallel port
        
        # Mask 2
        for i in range(4):
            visualNoise.draw()
            win.flip()
        
        textStimulus.opacity = 1.
        # Fixation cross
        textStimulus.text = '+'
        for i in range(one_second):
            textStimulus.draw()
            win.flip()
        
        # Subjective question (based on perceptual awareness scale)
        
        textStimulus.text = messages.pas
    
        textStimulus.draw()
        win.flip()
        subjective_answer, subjective_time = event.waitKeys(keyList=['1', '2', '3'], timeStamped=True)[0]
        
        # 1-on-1 staircase procedure
        #if right_or_wrong == 'wrong':
        if subjective_answer == '1':
            new_opacity = min(1., old_opacity+.02)
        elif subjective_answer == '2':
            new_opacity = min(1., old_opacity-.01)
        elif subjective_answer == '3':
            new_opacity = max(0., old_opacity-.02)
        
        # Objective question (based on semantic features)
        textStimulus.text = messages.obj_question(current_question)
        
        textStimulus.draw()
        win.flip()
        objective_answer, objective_time = event.waitKeys(keyList=['1', '3'], timeStamped=True)[0]
        
        correct_answer = '1' if current_answer == True else '3'
        #print((correct_answer, objective_answer))
        
        right_or_wrong = 'correct' if correct_answer == objective_answer \
                         else 'wrong'
        #print(right_or_wrong)
        
        # Packing up results
        trial_results = [current_word, \
                         current_question, \
                         subjective_answer, \
                         subjective_time, \
                         right_or_wrong, \
                         objective_time, \
                         old_opacity, \
                         stimulus_length]
        #print(trial_results)
        run_results.append(trial_results)
    
    # Writing results to file
    out_file = os.path.join(out_folder, 'sub-{:02}_run-{:02}.events'.format(\
                                         subject_number, r+1))
                                         
    with open(out_file, mode='w', encoding='utf-8') as o:
        o.write('Current word\t'\
                'Current question\t'\
                'PAS score\t'\
                'PAS RT\t'\
                'Accuracy\t'\
                'Objective question RT\t'\
                'Stimulus opacity\t'\
                'Stimulus length\t'\
                '\n')
        for t in run_results:
            for value in t:
                o.write('{}\t'.format(value))
            o.write('\n')
    subject_results.append(run_results)
    
    # Rest one minute
    if r < 23:
        for _ in range(one_second*60): 
            countdown = minuteCounter[_]
            textStimulus.text = messages.rest(r+1, countdown)
            textStimulus.draw()
            win.flip()
            keypress = event.getKeys(keyList=['n'])
            if len(keypress)>0 or _ == int(one_second*60):
                break
        
        # Start the new run
        textStimulus.text = messages.after_rest
        textStimulus.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
        
    # End of experiment
    textStimulus.text = messages.end
    textStimulus.draw()
    win.flip()
    core.wait(5)
