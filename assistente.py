import pyttsx3
import speech_recognition as sr
from playsound import playsound
import random
import datetime
hour = datetime.datetime.now().strftime('%H:%M')
#print(hour)
date = datetime.date.today().strftime('%d/%B/%Y')
#print(date)
date = date.split('/')
#print(date)
import webbrowser as wb
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from modules import carrega_agenda, comandos_respostas
comandos = comandos_respostas.comandos
respostas = comandos_respostas.respostas
#print(comandos)
#print(respostas)

meu_nome = 'Ana'

# MacOS
chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
# Windows
#chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'
# Linux
# chrome_path = '/usr/bin/google-chrome %s'

def search(frase):
    wb.get(chrome_path).open('https://www.google.com/search?q=' + frase)

#search('linguagem python')

MODEL_TYPES = ['EMOÇÃO']

def load_model_by_name(model_type):
    if model_type == MODEL_TYPES[0]:
        model = tf.keras.models.load_model('models/speech_emotion_recognition.hdf5')
        model_dict = sorted(list(['neutra', 'calma', 'feliz', 'triste', 'nervosa', 'medo', 'nojo', 'surpreso']))
        SAMPLE_RATE = 48000
    return model, model_dict, SAMPLE_RATE

#print(load_model_by_name('EMOÇÃO'))
#print(load_model_by_name('EMOÇÃO')[0].summary())

model_type = 'EMOÇÃO'
loaded_model = load_model_by_name(model_type)

def predict_sound(AUDIO, SAMPLE_RATE, plot = True):
    results = []
    wav_data, sample_rate = librosa.load(AUDIO, sr=SAMPLE_RATE)
    #print(wav_data)
    #print(wav_data.shape)
    # https: // librosa.org / doc / main / generated / librosa.effects.trim.html
    clip, index = librosa.effects.trim(wav_data, top_db=60, frame_length=512, hop_length=64)
    splitted_audio_data = tf.signal.frame(clip, sample_rate, sample_rate, pad_end=True, pad_value=0)
    for i, data in enumerate(splitted_audio_data.numpy()):
        #print('Audio split: ', i)
        #print(data)
        #print(data.shape)
        # http://www.midiacom.uff.br/debora/images/disciplinas/2018-2/smm/trabalhos/Apresentacao-MFCC.pdf
        mfccs_features = librosa.feature.mfcc(y = data, sr = sample_rate, n_mfcc=40)
        #print(mfccs_features.shape)
        #print(mfccs_features)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis = 0)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
        #print(mfccs_scaled_features.shape)
        # batch
        mfccs_scaled_features = mfccs_scaled_features[:,:,np.newaxis]
        #print(mfccs_scaled_features.shape)

        predictions = loaded_model[0].predict(mfccs_scaled_features, batch_size=32)
        #print(predictions)
        #print(predictions.sum())
        if plot:
            plt.figure(figsize=(len(splitted_audio_data), 5))
            plt.barh(loaded_model[1], predictions[0])
            plt.tight_layout()
            plt.show()

        predictions = predictions.argmax(axis = 1)
        #print(predictions)
        predictions = predictions.astype(int).flatten()
        predictions = loaded_model[1][predictions[0]]
        results.append(predictions)
        #print(results)

        result_str = 'PARTE ' + str(i) + ': ' + str(predictions).upper()
        #print(result_str)

    count_results = [[results.count(x), x] for x in set(results)]
    #print(count_results)

    #print(max(count_results))
    return max(count_results)

#predict_sound('triste.wav', loaded_model[2], plot=True)

def play_music_youtube(emocao):
    play = False
    if emocao == 'triste' or emocao == 'medo':
        wb.get(chrome_path).open('https://www.youtube.com/watch?v=k32IPg4dbz0&ab_channel=Amelhorm%C3%BAsicainstrumental')
        play = True
    if emocao == 'nervosa' or emocao == 'surpreso':
        wb.get(chrome_path).open('https://www.youtube.com/watch?v=pWjmpSD-ph0&ab_channel=CassioToledo')
        play = True
    return play

#play_music_youtube('triste')
#emocao = predict_sound('triste.wav', loaded_model[2], plot=False)
#print(emocao)
#play_music_youtube(emocao[1])

def speak(audio):
    engine = pyttsx3.init()
    engine.setProperty('rate', 85) # número de palavras por minuto
    engine.setProperty('volume', 1) # min: 0, max: 1
    engine.say(audio)
    engine.runAndWait()

#speak('Testando o sintetizador de voz da assistente')

def listen_microphone():
    microfone = sr.Recognizer()
    with sr.Microphone() as source:
        microfone.adjust_for_ambient_noise(source, duration=0.8)
        print('Ouvindo:')
        audio = microfone.listen(source)
        with open('recordings/speech.wav', 'wb') as f:
            f.write(audio.get_wav_data())
    try:
        frase = microfone.recognize_google(audio, language='pt-BR')
        print('Você disse: ' + frase)
    except sr.UnknownValueError:
        frase = ''
        print('Não entendi')
    return frase

#listen_microphone()

def test_models():
    audio_source = '/Users/jonesgranatyr/Documents/Ensino/IA Expert/Cursos/Assistente Virtual/curso_assistente/recordings/speech.wav'
    prediction = predict_sound(audio_source, loaded_model[2], plot = False)
    return prediction

#print(test_models())

playing = False
mode_control = False
print('[INFO] Pronto para começar!')
playsound('n1.mp3')

while (1):
    result = listen_microphone()

    if meu_nome in result:
        result = str(result.split(meu_nome + ' ')[1])
        result = result.lower()
        #print('Acionou a assistente!')
        #print('Após o processamento: ', result)

        if result in comandos[0]:
            playsound('n2.mp3')
            speak('Até agora minhas funções são: ' + respostas[0])

        if result in comandos[3]:
            playsound('n2.mp3')
            speak('Agora são ' + datetime.datetime.now().strftime('%H:%M'))

        if result in comandos[4]:
            playsound('n2.mp3')
            speak('Hoje é dia ' + date[0] + ' de ' + date[1])

        if result in comandos[1]:
            playsound('n2.mp3')
            speak('Pode falar!')
            result = listen_microphone()
            anotacao = open('anotacao.txt', mode='a+', encoding='utf-8')
            anotacao.write(result + '\n')
            anotacao.close()
            speak(''.join(random.sample(respostas[1], k = 1)))
            speak('Deseja que eu leia os lembretes?')
            result = listen_microphone()
            if result == 'sim' or result == 'pode ler':
                with open('anotacao.txt') as file_source:
                    lines = file_source.readlines()
                    for line in lines:
                        speak(line)
            else:
                speak('Ok!')

        if result in comandos[2]:
            playsound('n2.mp3')
            speak(''.join(random.sample(respostas[2], k = 1)))
            result = listen_microphone()
            search(result)

        if result in comandos[6]:
            playsound('n2.mp3')
            if carrega_agenda.carrega_agenda():
                speak('Estes são os eventos agendados para hoje:')
                for i in range(len(carrega_agenda.carrega_agenda()[1])):
                    speak(carrega_agenda.carrega_agenda()[1][i] + ' ' + carrega_agenda.carrega_agenda()[0][i] + ' agendada para as ' + str(carrega_agenda.carrega_agenda()[2][i]))
            else:
                speak('Não há eventos agendados para hoje a partir do horário atual!')

        if result in comandos[5]:
            mode_control = True
            playsound('n1.mp3')
            speak('Modo de análise de emoção foi ativado!')

        if mode_control:
            analyze = test_models()
            print(f'Notei {analyze} na sua voz')
            if not playing:
                playing = play_music_youtube(analyze[1])

        if result == 'encerrar':
            playsound('n2.mp3')
            speak(''.join(random.sample(respostas[4], k = 1)))
            break
    else:
        playsound('n3.mp3')





















