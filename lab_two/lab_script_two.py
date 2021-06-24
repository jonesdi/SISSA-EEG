import psychopy
from numpy import random

from psychopy import clock, event, visual

visualNoiseSize = 256 # Dimension in pixels of visual noise. Must be a power of 2
noiseSize = 256
noiseTexture = random.rand(visualNoiseSize,visualNoiseSize)
#*2-1

win = visual.Window(size=[1920,1080], fullscr=False, color=[-1,-1,-1], units='pix', checkTiming=True)
print('Frame rate detected: {}'.format(win.getActualFrameRate()))
actualFrameMs = win.getMsPerFrame(nFrames=180)[2]
predictedFrameMs = win.monitorFramePeriod
print('Predicted milliseconds per frame: {}'.format(predictedFrameMs*1000.))
print('Actual milliseconds per frame: {}'.format(actualFrameMs))

loadingMessage = 'Now loading...'

textStimulus = visual.TextStim(win, loadingMessage, color=[.8,.8,.8], pos=[0,0], ori=0, wrapWidth=1080, height=40, font='Calibri')
textStimulus.draw()

visualNoise = visual.GratingStim(win=win, tex=noiseTexture, size=(visualNoiseSize,256), units='pix', \
                                 interpolate=False, \
                                 mask='circle')
visualNoise.draw()

t = clock.Clock()

win.clearBuffer()

win.flip()

for rep in range(10):
    textStimulus.opacity = 0.5
    for i in range(30):
        
        visualNoise.draw()
        win.flip()
    t.reset()
    for i in range(2):
        textStimulus.draw()
        win.flip()
    print(t.getTime())
        
    for i in range(30):
        visualNoise.draw()
        win.flip()
        
win.close()
#event.waitKeys(keyList=['space'])
