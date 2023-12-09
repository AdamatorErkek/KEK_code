import os
import cv2


DATA_DIR = './data'  #DATA_DIR değişkeni oluşturup ./data dosyasında veriler yerleştirecez
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)    ## bu döngü burada eğer DATA_DIR yoksa onu oluştur döngüsüdür

number_of_classes = 3  # üç tane sınıf yani klasör için değişken oluşturduk
dataset_size = 100 # yüz tane görsel için değişken oluşturduk

cap = cv2.VideoCapture(0) # bildiğimiz yer

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):   # join eklemek için var  # j=1 ise resmi o path içinde data1 olarak kaydedilir
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()