import train_model
import predict_model
import Music_Player
import winsound
from shutil import move
from threading import Thread
import sms

songs = Music_Player.load_music()
cry_count = 1
silent_count = 1
song = []
flag = 0
songs2 = Music_Player.PriorityQueue()

def chk():
    while True:
        if cry_count==6 and silent_count==6:
            train_model()
            print("Training " + " " + cry_count + " " + silent_count)


status = True

if __name__ == '__main__':
    train_model.train_model()
    if(status==True):
        Thread(target = chk).start()
        print("Threading Started")
        status=False


while True:

    print("Executing the while loop.....")

    label, file  = predict_model.predict()

    if label == 1:
        # Play Music
        if songs.isEmpty():
            songs = songs2
            songs2 = Music_Player.PriorityQueue()
        song = songs.delete()
        print("Baby is Crying")
        print("Playing Song" + song[0])

        winsound.PlaySound("Songs/"+song[0], winsound.SND_FILENAME)

        flag = 1
        # if cry_count == 1:
        sms.send_sms("Baby is Crying", "7293858555")
        if cry_count == 6:
            predict_model.move_files(1)
            cry_count = 1
        move("New.wav", "Baby_Cry"+str(cry_count)+".wav")
        cry_count += 1
        songs2.insert(song[0], song[1])

    else:
        if silent_count == 6:
            predict_model.move_files(0)
            silent_count = 1
        move("New.wav", "Silent_"+ str(silent_count)+".wav")
        silent_count += 1
        if flag == 1:
            song[1] += 1
            print(song)
            songs2.insert(song[0],song[1])
            if not songs2.isEmpty():
                while not songs.isEmpty():
                    item = songs.delete()
                    songs2.insert(item[0], item[1])
            songs = songs2
            songs2 = Music_Player.PriorityQueue()
            flag = 0
        print("Baby is Silent")


