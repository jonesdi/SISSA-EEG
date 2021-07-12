class Messages:
    
    def __init__(self):
        
        self.pas = 'Cosa hai visto?\n\n\n'\
                   '1 - Non ho visto nulla\n\n'\
                   '2 - ho visto delle lettere, ma non una parola\n\n'\
                   '3 - ho visto una parola'
                   
        self.obj_question = lambda question : 'La parola si riferiva a qualcosa che {}?\n\n\n'\
                                              '1 - vero\t\t\t3 - falso'.format(question)
        
        self.start_example = 'Cominciamo con qualche parola di prova! \n\n\n [Premi la barra spaziatrice]'
        
        self.instructions_one = 'In questo esperimento, appariranno molto rapidamente degli stimoli visivi al centro dello schermo.'\
                                '\n'\
                                'Se dovessi vedere una parola, ti chiediamo di immaginare la cosa a cui quella parola si riferisce.'\
                                '\n\n'\
                                '[premi la barra spaziatrice per continuare]'
        
        self.instructions_two = 'Dopodichè, ti verranno fatte due domande riguardanti ciò che hai visto.'\
                                '\n'\
                                'Le domande ti verrano poste una dopo l\'altra, e potrai rispondere senza fretta usando i tasti 1,2 e 3.'\
                                '\n\n'\
                                'Nella prima ti chiederemo se hai riconosciuto qualcosa o meno sullo schermo.'\
                                '\n'\
                                'Nella seconda, invece, ti faremo una domanda sul significato della parola che potresti aver visto.'\
                                'In caso non avessi visto davvero nulla, metti una risposta a caso.'\
                                '\n\n'\
                                '[premi la barra spaziatrice per continuare]'
        
        self.real_exp = 'Ora procediamo con l\'esperimento vero e proprio! '\
                         '\n\n'\
                         'Se hai delle domande, ora è il momento di farle. \n\n\n'\
                         '[Altrimenti, premi la barra spaziatrice]'
                         
        self.end = 'Questa era l\'ultima sessione, e l\'esperimento ora è finito. \n\n\nGrazie per aver partecipato!'
        
        self.rest = lambda run, countdown : 'Fine della sessione {} su 16 - ottimo lavoro!\n\n'\
                                            'Ora hai 1 minuto di pausa prima di cominciare la prossima sessione.\n\n'\
                                            '00.{:02}'.format(run, countdown)
                                            
        self.after_rest = 'Procediamo! \n\n\n [Premi la barra spaziatrice]'