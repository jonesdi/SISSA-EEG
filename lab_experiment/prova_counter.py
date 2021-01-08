from psychopy import visual, core, event, gui, parallel, prefs
from utils import draw, format_instr, print_instr, create_run_splits


win = visual.Window(size=[1920,1080], fullscr=False, color=[-1,-1,-1], units='pix', winType='pyglet', screen=0, checkTiming=True)

refresh = 60
minuteCounter = {step : value for step, value in enumerate([abs(i) for i in range(-2, 1) for k in range(refresh)])}
AfterRest = format_instr(win, text='Procediamo! \n\n\n [Premi la barra spaziatrice]')
restText = lambda run, countdown : 'Fine della sessione {} su 32 - ottimo lavoro!\n\n Ora hai 1 minuto di pausa prima di cominciare la prossima sessione.\n\n 00.{:02}'.format(run, countdown)

clock = core.Clock()
for _ in range(int(refresh*1)): 
    print(_)
    countdown = minuteCounter[_]
    Rest = format_instr(win, text=restText(1, countdown))
    Rest.draw(win=win)
    win.flip()
    #keypress = event.getKeys(keyList=['space'])
    #if len(keypress)>0 or _ == int(refresh*60):
    if _ == int(refresh*1):
        #print_instr(win, AfterRest,0.5)
        break
print(clock.getTime())