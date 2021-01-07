import pyglet

display = pyglet.canvas.get_display()
screens = display.get_screens()
for screen_index, screen in enumerate(screens):
    details = screen.get_mode()
    rate = [w.strip().replace(')', '') for w in str(details).split(',') if 'rate' in w]
    assert len(rate) == 1
    rate = int(rate[0].split('=')[1])
    print('Screen {} - refresh rate: {} fps'.format(screen_index, rate))